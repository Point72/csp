import collections
import inspect
import typing
from abc import ABCMeta, abstractmethod

import numpy

import csp.typing
from csp.impl.types import tstype
from csp.impl.types.common_definitions import ArgKind, BasketKind, InputDef, OutputDef
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.numpy_type_util import map_numpy_dtype_to_python_type
from csp.impl.types.tstype import AttachType, SnapKeyType, SnapType, TsType, isTsDynamicBasket
from csp.impl.types.typing_utils import CspTypingUtils, FastList
from csp.impl.wiring.edge import Edge


class UpcastRegistry(object):
    _instance = None

    def __init__(self):
        self._type_registry: typing.Dict[typing.Tuple[type, type], type] = {}
        self._add_type_upcast(int, float, float)

    def resolve_type(self, expected_type: type, new_type: type, raise_on_error=True):
        if expected_type == new_type:
            return expected_type
        if expected_type is object or new_type is object:
            return object
        res = None

        if isinstance(expected_type, collections.abc.Hashable) and isinstance(new_type, collections.abc.Hashable):
            res = self._type_registry.get((expected_type, new_type), None)
        if res is None:
            if CspTypingUtils.is_generic_container(expected_type):
                expected_type_base = CspTypingUtils.get_orig_base(expected_type)
                if expected_type_base is new_type:
                    return expected_type_base  # If new_type is Generic and expected type is Generic[T], return Generic
                if CspTypingUtils.is_generic_container(new_type):
                    expected_origin = CspTypingUtils.get_origin(expected_type)
                    new_type_origin = CspTypingUtils.get_origin(new_type)
                    array_types = (csp.typing.Numpy1DArray, csp.typing.NumpyNDArray)
                    if (
                        expected_origin in array_types
                        and new_type_origin in array_types
                        and expected_type.__args__ == new_type.__args__
                    ):
                        return csp.typing.NumpyNDArray[new_type.__args__[0]]
                if raise_on_error:
                    raise TypeError(f"Incompatible types {expected_type} and {new_type}")
                else:
                    return None
            elif CspTypingUtils.is_generic_container(new_type):
                if CspTypingUtils.get_orig_base(new_type) is expected_type:
                    return expected_type
                elif raise_on_error:
                    raise TypeError(f"Incompatible types {expected_type} and {new_type}")
                else:
                    return None

            if inspect.isclass(expected_type) and inspect.isclass(new_type):
                if issubclass(expected_type, new_type):
                    # Generally if B inherits from A, we want to resolve from A, the only exception
                    # is "Generic types". Dict[int, int] inherits from dict but we want the type to be resolved to the generic type
                    # that keeps the type information
                    return expected_type if CspTypingUtils.is_generic_container(expected_type) else new_type
                elif issubclass(new_type, expected_type):
                    return new_type if CspTypingUtils.is_generic_container(new_type) else expected_type
            if raise_on_error:
                raise TypeError(f"Incompatible types {expected_type} and {new_type}")
        return res

    def _add_type_upcast(self, t1: type, t2: type, resolved_type: type):
        assert (t1, t2) not in self._type_registry
        self._type_registry[(t1, t2)] = resolved_type
        self._type_registry[(t2, t1)] = resolved_type

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = UpcastRegistry()
        return cls._instance


UpcastRegistry.instance()


class ContainerTypeVarResolutionError(TypeError):
    def __init__(self, func_name, tvar, tvar_value):
        self._func_name, self._tvar, self._tvar_value = func_name, tvar, tvar_value
        super().__init__(
            f"In function {func_name}: Unable to resolve container type for type variable {tvar} explicit value must have"
            + f" uniform values and be non empty, got: {tvar_value} "
        )

    def __reduce__(self):
        return (ContainerTypeVarResolutionError, (self._func_name, self._tvar, self._tvar_value))


class TypeMismatchError(TypeError):
    @classmethod
    def pretty_typename(cls, typ):
        return CspTypingUtils.pretty_typename(typ)

    @classmethod
    def get_tvar_info_str(cls, tvar_info):
        if tvar_info is None:
            return ""
        else:
            return "({})".format(",".join([f"{k}={cls.pretty_typename(v)}" for k, v in tvar_info.items()]))


class ArgTypeMismatchError(TypeMismatchError):
    def __init__(self, func_name, expected_t, actual_arg, arg_name, tvar_info=None):
        self._func_name, self._expected_t, self._actual_arg, self._arg_name, self._tvar_info = (
            func_name,
            expected_t,
            actual_arg,
            arg_name,
            tvar_info,
        )
        super().__init__(
            f"In function {func_name}: Expected {self.pretty_typename(expected_t)} for "
            + ("return value, " if arg_name is None else f"argument '{arg_name}', ")
            + f"got {actual_arg} ({self.pretty_typename(type(actual_arg))}){self.get_tvar_info_str(tvar_info)}"
        )

    def __reduce__(self):
        return (
            ArgTypeMismatchError,
            (self._func_name, self._expected_t, self._actual_arg, self._arg_name, self._tvar_info),
        )


class ArgContainerMismatchError(TypeMismatchError):
    def __init__(self, func_name, expected_t, actual_arg, arg_name, tvar_info=None):
        self._func_name, self._expected_t, self._actual_arg, self._arg_name = (
            func_name,
            expected_t,
            actual_arg,
            arg_name,
        )
        super().__init__(
            f"In function {func_name}: Expected {self.pretty_typename(expected_t)} for argument '{arg_name}', got {actual_arg} "
            + "instead of generic container type specification"
        )

    def __reduce__(self):
        return (ArgContainerMismatchError, (self._func_name, self._expected_t, self._actual_arg, self._arg_name))


class TSArgTypeMismatchError(TypeMismatchError):
    def __init__(self, func_name, expected_t, actual_arg_type, arg_name, tvar_info=None):
        self._func_name, self._expected_t, self._actual_arg_type, self._arg_name, self._tvar_info = (
            func_name,
            expected_t,
            actual_arg_type,
            arg_name,
            tvar_info,
        )
        actual_type_str = f"ts[{self.pretty_typename(actual_arg_type)}]" if actual_arg_type else "None"

        super().__init__(
            f"In function {func_name}: Expected ts[{self.pretty_typename(expected_t)}] for "
            + ("return value, " if arg_name is None else f"argument '{arg_name}', ")
            + f"got {actual_type_str}{self.get_tvar_info_str(tvar_info)}"
        )

    def __reduce__(self):
        return (
            TSArgTypeMismatchError,
            (self._func_name, self._expected_t, self._actual_arg_type, self._arg_name, self._tvar_info),
        )


class TSDictBasketKeyMismatchError(TypeMismatchError):
    def __init__(self, func_name, expected_t, arg_name):
        self._func_name, self._expected_t, self._arg_name = func_name, expected_t, arg_name
        super().__init__(
            f"In function {func_name}: Expected ts[{self.pretty_typename(expected_t)}] for argument {arg_name} must have str keys "
        )

    def __reduce__(self):
        return (TSDictBasketKeyMismatchError, (self._func_name, self._expected_t, self._arg_name))


class NestedTsTypeError:
    def __init__(self):
        super().__init__("Found nested ts type - this is not allowed")


class _InstanceTypeResolverBase(metaclass=ABCMeta):
    def __init__(
        self,
        function_name: str,
        input_or_output_definitions: typing.Union[typing.Tuple[InputDef], typing.Tuple[OutputDef]],
        values: typing.List[object],
        forced_tvars: typing.Union[typing.Dict[str, typing.Type], None],
        is_input=True,
        allow_none_ts=False,
    ):
        self._function_name = function_name
        self._input_or_output_definitions = input_or_output_definitions
        self._arguments = values
        self._forced_tvars = forced_tvars
        self._def_name = "inputdef" if is_input else "outputdef"
        self._allow_none_ts = allow_none_ts

        self._tvars: typing.Dict[str, type] = {}
        self._conflicting_tvar_types = {}

        if self._forced_tvars:
            self._forced_tvars = {k: ContainerTypeNormalizer.normalize_type(v) for k, v in self._forced_tvars.items()}
            self._tvars.update(**self._forced_tvars)

        self._cur_def = None
        self._cur_arg = None
        self._resolve_types()
        self._cur_def = None
        self._cur_arg = None

    def _resolve_types(self):
        for arg, in_out_def in zip(self._arguments, self._input_or_output_definitions):
            self._cur_def = in_out_def
            self._cur_arg = arg
            # TODO type check here
            if in_out_def.kind.is_single_ts():
                self._add_ts_value(arg, in_out_def)
            elif in_out_def.kind.is_scalar():
                if CspTypingUtils.is_generic_container(in_out_def.typ):
                    self._add_container_scalar_value(arg, in_out_def)
                else:
                    self._add_scalar_value(arg, in_out_def)
            elif in_out_def.kind == ArgKind.ALARM:
                # TODO: Handle alarms better?
                pass
            elif in_out_def.kind.is_non_dynamic_basket():
                self._add_basket_ts_value(arg, in_out_def)
            elif in_out_def.kind.is_dynamic_basket():
                self._add_dynamic_basket(arg, in_out_def)
            else:
                raise RuntimeError(f"Unexpected {self._def_name} kind {in_out_def.kind}")
        self._try_resolve_tvar_conflicts()

    def _is_expected_generic_meta(self, typ, expected_generic_meta):
        is_generic_container = CspTypingUtils.is_generic_container(typ)
        return (
            is_generic_container
            and (
                CspTypingUtils.get_origin(typ) is expected_generic_meta
                or (expected_generic_meta is typing.List and CspTypingUtils.get_origin(typ) is FastList)
            )
        ) or (
            not is_generic_container
            and isinstance(expected_generic_meta, typ)
            and issubclass(expected_generic_meta, typ)
        )

    def _rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
        self, actual_sub_type, expected_sub_type, is_in_ts=False
    ):
        if expected_sub_type is typing.Any:
            return True
        elif isinstance(expected_sub_type, typing.TypeVar):
            # If it's a generic and the type doesn't match we will throw exception on conflict at the end
            self._add_t_var_resolution(expected_sub_type.__name__, actual_sub_type)
            return True
        elif CspTypingUtils.is_forward_ref(expected_sub_type):
            self._add_t_var_resolution(expected_sub_type.__forward_arg__, actual_sub_type)
            return True
        elif isinstance(expected_sub_type, str):
            # If it's a generic and the type doesn't match we will throw exception on conflict at the end
            self._add_t_var_resolution(expected_sub_type, actual_sub_type)
            return True
        elif CspTypingUtils.is_generic_container(expected_sub_type):
            origin_typ = CspTypingUtils.get_origin(expected_sub_type)
            if not self._is_expected_generic_meta(actual_sub_type, origin_typ):
                # For multidimensional arrays we allow to pass one dimensional arrays as well
                if origin_typ is not csp.typing.NumpyNDArray or not self._is_expected_generic_meta(
                    actual_sub_type, csp.typing.Numpy1DArray
                ):
                    return False
            if origin_typ is TsType:
                if is_in_ts:
                    raise NestedTsTypeError()
                return self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                    actual_sub_type.__args__[0], expected_sub_type.__args__[0], is_in_ts=True
                )
            elif origin_typ is typing.List or origin_typ is typing.Set:
                return self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                    actual_sub_type.__args__[0], expected_sub_type.__args__[0], is_in_ts=is_in_ts
                )
            elif origin_typ is typing.Dict:
                if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                    actual_sub_type.__args__[0], expected_sub_type.__args__[0], is_in_ts=is_in_ts
                ):
                    return False
                return self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                    actual_sub_type.__args__[1], expected_sub_type.__args__[1], is_in_ts=is_in_ts
                )
            elif origin_typ in (csp.typing.Numpy1DArray, csp.typing.NumpyNDArray):
                # We want to support also passing ndarray wherever csp.typing.NumpyArray[...] is expected
                if CspTypingUtils.is_generic_container(actual_sub_type):
                    return self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                        actual_sub_type.__args__[0], expected_sub_type.__args__[0], is_in_ts=is_in_ts
                    )
                else:
                    return issubclass(expected_sub_type, actual_sub_type)
            else:
                # TODO: consider not raising here but doing something smarter
                raise NotImplementedError(f"Unsupported generic type decoration {expected_sub_type}")
        elif CspTypingUtils.is_union_type(expected_sub_type):
            for t in expected_sub_type.__args__:
                if self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(actual_sub_type, t):
                    return True
            return False
        else:
            # At this point it must be a scalar value
            res_type = UpcastRegistry.instance().resolve_type(expected_sub_type, actual_sub_type, raise_on_error=False)
            return res_type is expected_sub_type
        return True

    def _add_container_scalar_value(self, arg, in_out_def):
        if not self._rec_validate_container_and_resolve_tvars(arg, in_out_def.typ):
            self._raise_arg_mismatch_error(arg)

    def _raise_arg_mismatch_error(self, arg=None, tvar_info=None):
        if self._cur_def.kind.is_single_ts():
            arg_type = type(arg) if arg is not None else None
            if arg is not None:
                if not isinstance(arg, Edge):
                    raise ArgTypeMismatchError(
                        func_name=self._function_name,
                        expected_t=self._cur_def.typ,
                        actual_arg=arg,
                        arg_name=self._cur_def.name,
                        tvar_info=tvar_info,
                    )
                if isTsDynamicBasket(arg.tstype):
                    arg_type = arg.tstype
                else:
                    arg_type = arg.tstype.typ
            raise TSArgTypeMismatchError(
                func_name=self._function_name,
                expected_t=self._cur_def.typ.typ,
                actual_arg_type=arg_type,
                arg_name=self._cur_def.name,
                tvar_info=tvar_info,
            )
        else:
            expected_type = self._cur_def.typ
            if self._cur_def.kind == ArgKind.BASKET_TS:
                if hasattr(self._cur_def, "shape") and self._cur_def.shape is not None:
                    if isinstance(self._cur_def.shape, list):
                        expected_type = typing.Dict[type(self._cur_def.shape[0]), self._cur_def.typ]
                    else:
                        expected_type = typing.List[self._cur_def.typ]
            raise ArgTypeMismatchError(
                func_name=self._function_name,
                expected_t=expected_type,
                actual_arg=arg,
                arg_name=self._cur_def.name,
                tvar_info=tvar_info,
            )

    def _add_scalar_value(self, arg, in_out_def):
        inp_def_type = in_out_def.typ
        if isinstance(inp_def_type, typing.TypeVar):
            inp_def_type = inp_def_type.__name__

        if isinstance(inp_def_type, str):
            tvar = inp_def_type
            typ = arg
            if tvar[0] == "~":  # Passed by value
                typ = type(arg)
                tvar = tvar[1:]
            else:
                if not CspTypingUtils.is_type_spec(typ):
                    typ = ContainerTypeNormalizer.normalize_type(typ)
                if not CspTypingUtils.is_type_spec(typ):
                    self._raise_arg_mismatch_error(arg=arg)
            self._add_t_var_resolution(tvar, typ, arg if arg is not typ else None)
        else:
            if not self._is_scalar_value_matching_spec(inp_def_type, arg) and arg is not None:
                self._raise_arg_mismatch_error(arg)

    def _is_scalar_value_matching_spec(self, inp_def_type, arg):
        if inp_def_type is typing.Any:
            return True
        if CspTypingUtils.is_callable(inp_def_type):
            return callable(arg)
        resolved_type = UpcastRegistry.instance().resolve_type(inp_def_type, type(arg), raise_on_error=False)
        if resolved_type is inp_def_type:
            return True
        elif (
            CspTypingUtils.is_generic_container(inp_def_type)
            and CspTypingUtils.get_orig_base(inp_def_type) is resolved_type
        ):
            return True
        if CspTypingUtils.is_union_type(inp_def_type):
            types = inp_def_type.__args__
            for t in types:
                if self._is_scalar_value_matching_spec(t, arg):
                    return True
        if isinstance(arg, SnapType):
            return arg.ts_type.typ is inp_def_type
        if isinstance(arg, SnapKeyType):
            return arg.key_tstype.typ is inp_def_type
        return False

    def _rec_validate_container_and_resolve_tvars(self, sub_arg, sub_type_def):
        if isinstance(sub_arg, SnapType):
            return sub_arg.ts_type.typ == sub_type_def
        if sub_type_def is typing.Any:
            return True
        elif isinstance(sub_type_def, typing.TypeVar):
            # If it's a generic and the type doesn't match we will throw exception on conflict at the end
            self._add_t_var_resolution(sub_type_def.__name__, type(sub_arg))
            return True
        elif CspTypingUtils.is_forward_ref(sub_type_def):
            self._add_t_var_resolution(sub_type_def.__forward_arg__, type(sub_arg))
            return True
        elif CspTypingUtils.is_generic_container(sub_type_def):
            # TODO: THESE CHECKS MIGHT BE EXPENSIVE AND WE MIGHT WANT TO HAVE CONFIG TO DISABLE THEM

            if CspTypingUtils.get_origin(sub_type_def) is typing.List:
                if not isinstance(sub_arg, list):
                    return False
                (sub_type,) = sub_type_def.__args__
                for v in sub_arg:
                    if not self._rec_validate_container_and_resolve_tvars(v, sub_type):
                        return False
            elif CspTypingUtils.get_origin(sub_type_def) is typing.Set:
                if not isinstance(sub_arg, set):
                    return False
                (sub_type,) = sub_type_def.__args__
                for v in sub_arg:
                    if not self._rec_validate_container_and_resolve_tvars(v, sub_type):
                        return False
            elif CspTypingUtils.get_origin(sub_type_def) is typing.Dict:
                if not isinstance(sub_arg, dict):
                    return False
                key_type, value_type = sub_type_def.__args__
                for k, v in sub_arg.items():
                    if not self._rec_validate_container_and_resolve_tvars(k, key_type):
                        return False
                    if not self._rec_validate_container_and_resolve_tvars(v, value_type):
                        return False
            elif tstype.isTsType(sub_type_def):
                return self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(sub_type_def.typ, sub_arg)
            else:
                # For any type that we don't support, we can't verify anyway, so better let user
                # specify correct type instead of forcing them to specify object
                return True
        else:
            # At this point it must be a scalar value
            if not self._is_scalar_value_matching_spec(sub_type_def, sub_arg):
                return False
        return True

    def _add_ts_value(self, arg, in_out_def):
        if not isinstance(arg, Edge) or isTsDynamicBasket(arg.tstype):
            if self._allow_none_ts and arg is None:
                return
            self._raise_arg_mismatch_error(arg)

        if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(arg.tstype.typ, in_out_def.typ.typ):
            self._raise_arg_mismatch_error(arg)

    @abstractmethod
    def _get_basket_kind(self, in_out_def):
        raise NotImplementedError()

    def _add_list_basket_ts_value(self, args, in_out_def):
        assert CspTypingUtils.get_origin(in_out_def.typ) is typing.List
        if not isinstance(args, (list, tuple)):
            if self._allow_none_ts and args is None:
                return
            self._raise_arg_mismatch_error(args)
        (ts_arg_type,) = in_out_def.typ.__args__
        for arg in args:
            if not isinstance(arg, Edge) or not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                arg.tstype.typ, ts_arg_type.typ
            ):
                self._raise_arg_mismatch_error(args)

    def _add_dict_basket_ts_value(self, args, in_out_def):
        assert CspTypingUtils.get_origin(in_out_def.typ) is typing.Dict
        if not isinstance(args, dict):
            if self._allow_none_ts and args is None:
                return
            self._raise_arg_mismatch_error(args)
        key_type, ts_arg_type = in_out_def.typ.__args__
        for k, v in args.items():
            if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                type(k), key_type
            ) or not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(v.tstype.typ, ts_arg_type.typ):
                self._raise_arg_mismatch_error(args)

    def _add_basket_ts_value(self, args, in_out_def):
        basket_kind = self._get_basket_kind(in_out_def)
        if basket_kind == BasketKind.LIST:
            self._add_list_basket_ts_value(args, in_out_def)
        elif basket_kind == BasketKind.DICT:
            self._add_dict_basket_ts_value(args, in_out_def)
        else:
            raise NotImplementedError(f"Unsupported basket kind: {basket_kind}")

        if args is not None and len(args) == 0:
            raise ValueError("0-sized timeseries baskets are not supported")

    def _add_dynamic_basket(self, arg, in_out_def):
        if not isinstance(arg, Edge) or not isTsDynamicBasket(arg.tstype):
            self._raise_arg_mismatch_error(arg)

        if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(arg.tstype, in_out_def.typ):
            self._raise_arg_mismatch_error(arg.tstype)

    def _add_t_var_resolution(self, tvar, resolved_type, arg=None):
        old_tvar_type = self._tvars.get(tvar)
        if old_tvar_type is None:
            self._tvars[tvar] = self._resolve_tvar_container_internal_types(tvar, resolved_type, arg)
            return
        elif self._forced_tvars and tvar in self._forced_tvars:
            # We must not change types, it's forced. So we will have to make sure that the new resolution matches the old
            # one
            if arg is not None:
                if not self._rec_validate_container_and_resolve_tvars(arg, old_tvar_type):
                    self._raise_arg_mismatch_error(arg=arg, tvar_info={tvar: old_tvar_type})
            else:
                if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(resolved_type, old_tvar_type):
                    self._raise_arg_mismatch_error(arg=self._cur_arg, tvar_info={tvar: old_tvar_type})
            return

        combined_type = UpcastRegistry.instance().resolve_type(resolved_type, old_tvar_type, raise_on_error=False)
        if combined_type is None:
            conflicting_tvar_types = self._conflicting_tvar_types.get(tvar)
            if conflicting_tvar_types is None:
                conflicting_tvar_types = []
                self._conflicting_tvar_types[tvar] = conflicting_tvar_types

            conflicting_tvar_types.append(resolved_type)

        if combined_type is not None and combined_type != old_tvar_type:
            self._tvars[tvar] = combined_type

    def _resolve_tvar_container_internal_types(self, tvar, container_typ, arg, raise_on_error=True):
        if arg is None:
            return container_typ
        if container_typ not in (set, dict, list, numpy.ndarray):
            return container_typ
        # It's possible that we provided type as scalar argument, that's illegal for containers, it must specify explicitly typed
        # list
        if arg is container_typ:
            if raise_on_error:
                raise ArgContainerMismatchError(
                    func_name=self._function_name, expected_t=tvar, actual_arg=arg, arg_name=self._cur_def.name
                )
            else:
                return False
        if len(arg) == 0:
            return container_typ
        res = None
        if isinstance(arg, set):
            first_val = arg.__iter__().__next__()
            first_val_t = self._resolve_tvar_container_internal_types(tvar, type(first_val), first_val)
            if first_val_t:
                res = typing.Set[first_val_t]
        elif isinstance(arg, list):
            first_val = arg.__iter__().__next__()
            first_val_t = self._resolve_tvar_container_internal_types(tvar, type(first_val), first_val)
            if first_val_t:
                res = typing.List[first_val_t]
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
                res = typing.Dict[first_key_t, first_val_t]
        if not res and raise_on_error:
            raise ContainerTypeVarResolutionError(self._function_name, tvar, arg)
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
                    raise TypeError(
                        f"Conflicting type resolution for {tvar} when calling to {self._function_name} : "
                        + f"{resolved_type, conflicting_type}"
                    )

    @property
    def tvars(self) -> typing.Dict[str, type]:
        return self._tvars


class InputInstanceTypeResolver(_InstanceTypeResolverBase):
    def __init__(
        self,
        function_name: str,
        input_definitions: typing.Tuple[InputDef],
        arguments: typing.List[object],
        forced_tvars: typing.Union[typing.Dict[str, typing.Type], None],
        allow_none_ts: bool = False,
    ):
        self._scalar_inputs: typing.List[object] = []
        self._ts_inputs: typing.List[Edge] = []
        super().__init__(
            function_name=function_name,
            input_or_output_definitions=input_definitions,
            values=arguments,
            forced_tvars=forced_tvars,
            allow_none_ts=allow_none_ts,
        )

    def _get_basket_kind(self, in_out_def):
        return in_out_def.basket_kind

    def _add_container_scalar_value(self, arg, in_out_def):
        super()._add_container_scalar_value(arg, in_out_def)
        self._scalar_inputs.append(arg)

    def _add_scalar_value(self, arg, in_out_def):
        super()._add_scalar_value(arg, in_out_def)
        self._scalar_inputs.append(arg)

    def _add_ts_value(self, arg, in_out_def):
        if isinstance(arg, AttachType):
            # For csp.attach we keep the argument as a scalar, but type check as the value timeseries of the dynamic basket
            if not self._rec_validate_type_spec_vs_type_spec_and_resolve_tvars(
                arg.value_tstype.typ, in_out_def.typ.typ
            ):
                self._raise_arg_mismatch_error(arg)
            self._scalar_inputs.append(arg)
            return

        super()._add_ts_value(arg, in_out_def)
        self._ts_inputs.append(arg)

    def _add_basket_ts_value(self, args, in_out_def):
        super()._add_basket_ts_value(args, in_out_def)
        if isinstance(args, tuple):
            self._ts_inputs.append(list(args))
        else:
            self._ts_inputs.append(args)

    def _add_dynamic_basket(self, arg, in_out_def):
        super()._add_dynamic_basket(arg, in_out_def)
        self._ts_inputs.append(arg)

    @property
    def ts_inputs(self) -> typing.List[Edge]:
        return self._ts_inputs

    @property
    def scalar_inputs(self) -> typing.List[object]:
        return self._scalar_inputs


class GraphOutputTypeResolver(_InstanceTypeResolverBase):
    def __init__(
        self,
        function_name: str,
        output_definitions: typing.Tuple[OutputDef],
        values: typing.List[object],
        forced_tvars: typing.Union[typing.Dict[str, typing.Type], None],
    ):
        super().__init__(
            function_name=function_name,
            input_or_output_definitions=output_definitions,
            values=values,
            forced_tvars=forced_tvars,
            allow_none_ts=False,
        )

    def _add_container_scalar_value(self, arg, in_out_def):
        raise TypeError("Graph is trying to return scalar value, it can only returns ts and basket values")

    def _add_scalar_value(self, arg, in_out_def):
        raise TypeError("Graph is trying to return scalar value, it can only returns ts and basket values")

    def _get_basket_kind(self, in_out_def):
        if in_out_def.shape is not None:
            assert tstype.isTsType(in_out_def.typ)
            if isinstance(in_out_def.shape, int):
                return BasketKind.LIST
            else:
                assert isinstance(in_out_def.shape, (list, tuple))
                return BasketKind.DICT

        if CspTypingUtils.get_origin(in_out_def.typ) is typing.Dict:
            return BasketKind.DICT
        else:
            assert CspTypingUtils.get_origin(in_out_def.typ) is typing.List
            return BasketKind.LIST

    def _add_list_basket_ts_value(self, args, in_out_def):
        # For basket outputs we have 2 possibilities:
        # The shape is resolved and then we have to do special handling, otherwise the base handling should be fine
        if in_out_def.kind == ArgKind.BASKET_TS and in_out_def.shape is not None:
            if len(args) != in_out_def.shape:
                raise RuntimeError(
                    f"In function {self._function_name}: Expected output shape for output {in_out_def.name} is of length {in_out_def.shape}, actual length is {len(args)}"
                )
            expected_ts_type = in_out_def.typ.typ
            for value in args:
                if not self._rec_validate_container_and_resolve_tvars(expected_ts_type, value.tstype):
                    self._raise_arg_mismatch_error(args)
        else:
            super()._add_list_basket_ts_value(args, in_out_def)

    def _add_dict_basket_ts_value(self, args, in_out_def):
        # For basket outputs we have 2 possibilities:
        # The shape is resolved and then we have to do special handling, otherwise the base handling should be fine
        if in_out_def.kind == ArgKind.BASKET_TS and in_out_def.shape is not None:
            if len(args) != len(in_out_def.shape):
                raise RuntimeError(
                    f"In function {self._function_name}: Expected output shape for output {in_out_def.name} is of length {len(in_out_def.shape)}, actual length is {len(args)}"
                )

            for k in in_out_def.shape:
                if k not in args:
                    raise RuntimeError(
                        f"In function {self._function_name}: Expected key {k} for output {in_out_def.name} is missing from the actual returned value"
                    )

            expected_ts_type = in_out_def.typ.typ

            for value in args.values():
                if not self._rec_validate_container_and_resolve_tvars(expected_ts_type, value.tstype):
                    self._raise_arg_mismatch_error(args)
        else:
            super()._add_dict_basket_ts_value(args, in_out_def)
