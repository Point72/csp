from typing import Any, Dict, List, Set, Tuple, Type, Union, get_args

import numpy
from pydantic import TypeAdapter, ValidationError

import csp.typing
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.instantiation_type_resolver import UpcastRegistry
from csp.impl.types.numpy_type_util import map_numpy_dtype_to_python_type
from csp.impl.types.pydantic_types import CspTypeVarType, adjust_annotations
from csp.impl.types.typing_utils import CspTypingUtils, TsTypeValidator


class TVarValidationContext:
    """Custom validation context class for handling the special csp TVAR logic."""

    # Note: some of the implementation is borrowed from InputInstanceTypeResolver

    def __init__(
        self,
        forced_tvars: Union[Dict[str, Type], None] = None,
        allow_none_ts: bool = False,
    ):
        # Can be set by a field validator to help track the source field of the different tvar refs
        self.field_name = None
        self._allow_none_ts = allow_none_ts
        self._forced_tvars: Dict[str, Type] = forced_tvars or {}
        self._tvar_type_refs: Dict[str, Set[Tuple[str, Type]]] = {}
        self._tvar_refs: Dict[str, Dict[str, List[Any]]] = {}
        self._tvars: Dict[str, Type] = {}
        self._conflicting_tvar_types = {}

        if self._forced_tvars:
            config = {"arbitrary_types_allowed": True, "strict": True}
            self._forced_tvars = {k: ContainerTypeNormalizer.normalize_type(v) for k, v in self._forced_tvars.items()}
            self._forced_tvar_adapters = {
                tvar: TypeAdapter(List[t], config=config) for tvar, t in self._forced_tvars.items()
            }
            self._forced_tvar_validators = {tvar: TsTypeValidator(t) for tvar, t in self._forced_tvars.items()}
            self._tvars.update(**self._forced_tvars)

    @property
    def tvars(self) -> Dict[str, Type]:
        return self._tvars

    @property
    def allow_none_ts(self) -> bool:
        return self._allow_none_ts

    def add_tvar_type_ref(self, tvar, value_type):
        if value_type is not numpy.ndarray:
            # Need to convert, i.e. [float] into List[float] when passed as a tref
            # Exclude ndarray because otherwise will get converted to NumpyNDArray[float], even for non-float
            # See, i.e. TestParquetReader.test_numpy_array_on_struct_with_field_map
            # TODO: This should be fixed in the ContainerTypeNormalizer
            value_type = ContainerTypeNormalizer.normalize_type(value_type)
        self._tvar_type_refs.setdefault(tvar, set()).add((self.field_name, value_type))

    def add_tvar_ref(self, tvar, value):
        self._tvar_refs.setdefault(tvar, {}).setdefault(self.field_name, []).append(value)

    def resolve_tvars(self):
        # Validate instances against forced tvars
        if self._forced_tvars:
            for tvar, adapter in self._forced_tvar_adapters.items():
                for field_name, field_values in self._tvar_refs.get(tvar, {}).items():
                    # Validate using TypeAdapter(List[t]) in pydantic as it's faster than iterating through in python
                    adapter.validate_python(field_values, strict=True)

            for tvar, validator in self._forced_tvar_validators.items():
                for field_name, v in self._tvar_type_refs.get(tvar, set()):
                    validator.validate(v)

        # Add resolutions for references to tvar types (where type is inferred directly from type)
        for tvar, type_refs in self._tvar_type_refs.items():
            for field_name, value_type in type_refs:
                self._add_t_var_resolution(tvar, field_name, value_type)

        # Add resolutions for references to tvar values (where type is inferred from type of value)
        for tvar, field_refs in self._tvar_refs.items():
            if self._forced_tvars and tvar in self._forced_tvars:
                # Already handled these
                continue
            for field_name, values in field_refs.items():
                for value in values:
                    typ = type(value)
                    if not CspTypingUtils.is_type_spec(typ):
                        typ = ContainerTypeNormalizer.normalize_type(typ)
                    self._add_t_var_resolution(tvar, field_name, typ, value if value is not typ else None)
        self._try_resolve_tvar_conflicts()

    def revalidate(self, model):
        """Once tvars have been resolved, need to revalidate input values against resolved tvars"""
        # Determine the fields that need to be revalidated because of tvar resolution
        # At the moment, that's only int fields that need to be converted to float
        # What does revalidation do?
        #   - It makes sure that, edges declared as ts[float] inside a data structure, i.e. List[ts[float]],
        #     get properly converted from, ts[int]
        #   - It makes sure that scalar int values get converted to float
        #   - It ignores validating a pass "int" type as a "float" type.
        fields_to_revalidate = set()
        for tvar, type_refs in self._tvar_type_refs.items():
            if self._tvars[tvar] is float:
                for field_name, value_type in type_refs:
                    if field_name and value_type is int:
                        fields_to_revalidate.add(field_name)
        for tvar, field_refs in self._tvar_refs.items():
            for field_name, values in field_refs.items():
                if field_name and any(type(value) is int for value in values):  # noqa E721
                    fields_to_revalidate.add(field_name)
        # Do the conversion only for the relevant fields
        for field in fields_to_revalidate:
            value = getattr(model, field)
            annotation = model.__annotations__[field]
            args = get_args(annotation)
            if args and args[0] is CspTypeVarType:
                # Skip revalidation of top-level type var types, as these have been handled via tvar resolution
                continue
            new_annotation = adjust_annotations(annotation, forced_tvars=self.tvars)
            try:
                new_value = TypeAdapter(new_annotation).validate_python(value)
            except ValidationError as e:
                msg = "\t" + str(e).replace("\n", "\n\t")
                raise ValueError(
                    f"failed to revalidate field `{field}` after applying Tvars: {self._tvars}\n{msg}\n"
                ) from None
            setattr(model, field, new_value)
        return model

    def _add_t_var_resolution(self, tvar, field_name, resolved_type, arg=None):
        old_tvar_type = self._tvars.get(tvar)
        if old_tvar_type is None:
            self._tvars[tvar] = self._resolve_tvar_container_internal_types(tvar, resolved_type, arg)
            return
        elif self._forced_tvars and tvar in self._forced_tvars:
            # We must not change types, it's forced. So we will have to make sure that the new resolution matches the old one
            return

        combined_type = UpcastRegistry.instance().resolve_type(resolved_type, old_tvar_type, raise_on_error=False)
        if combined_type is None:
            self._conflicting_tvar_types.setdefault(tvar, []).append(resolved_type)

        if combined_type is not None and combined_type != old_tvar_type:
            self._tvars[tvar] = combined_type

    def _resolve_tvar_container_internal_types(self, tvar, container_typ, arg, raise_on_error=True):
        """This function takes, a container type (i.e. list) and an arg (i.e. 6) and infers the type of the TVar,
        i.e. typing.List[int]. For simple types, this function is a pass-through (i.e. arg is None).
        """
        if arg is None:
            return container_typ
        if container_typ not in (set, dict, list, numpy.ndarray):
            return container_typ
        # It's possible that we provided type as scalar argument, that's illegal for containers, it must specify explicitly typed
        # list
        if arg is container_typ:
            if raise_on_error:
                raise ValueError(f"unable to resolve container type for type variable {tvar}: invalid argument {arg}")
            else:
                return False
        if len(arg) == 0:
            return container_typ
        res = None
        if isinstance(arg, set):
            first_val = arg.__iter__().__next__()
            first_val_t = self._resolve_tvar_container_internal_types(tvar, type(first_val), first_val)
            if first_val_t:
                res = Set[first_val_t]
        elif isinstance(arg, list):
            first_val = arg.__iter__().__next__()
            first_val_t = self._resolve_tvar_container_internal_types(tvar, type(first_val), first_val)
            if first_val_t:
                res = List[first_val_t]
        elif isinstance(arg, numpy.ndarray):
            python_type = map_numpy_dtype_to_python_type(arg.dtype)
            if arg.ndim > 1:
                res = csp.typing.NumpyNDArray[python_type]
            else:
                res = csp.typing.Numpy1DArray[python_type]
        else:
            first_k, first_val = arg.items().__iter__().__next__()
            first_key_t = self._resolve_tvar_container_internal_types(tvar, type(first_k), first_k)
            first_val_t = self._resolve_tvar_container_internal_types(tvar, type(first_val), first_val)
            if first_key_t and first_val_t:
                res = Dict[first_key_t, first_val_t]
        if not res and raise_on_error:
            raise ValueError(f"unable to resolve container type for type variable {tvar}.")
        return res

    def _try_resolve_tvar_conflicts(self):
        for tvar, conflicting_types in self._conflicting_tvar_types.items():
            # Consider the case:
            # f(x : 'T', y:'T', z : 'T')
            # f(1, Dummy(), object())
            # The resolution between x and y will fail, while resolution between x and z will be object. After we resolve all,
            # the tvars resolution should have the most primitive subtype (object in this case) and we can now resolve Dummy to
            # object as well
            resolved_type = self._tvars.get(tvar)
            assert resolved_type, f'"{tvar}" was not resolved'
            for conflicting_type in conflicting_types:
                if (
                    UpcastRegistry.instance().resolve_type(resolved_type, conflicting_type, raise_on_error=False)
                    is not resolved_type
                ):
                    raise ValueError(f"Conflicting type resolution for {tvar}: {resolved_type, conflicting_type}")
