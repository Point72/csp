import typing

from csp.impl.types.common_definitions import OutputBasketContainer
from csp.impl.types.typing_utils import CspTypingUtils


class UnresolvedTvarError(TypeError):
    def __init__(self, tvar_name):
        super().__init__("missing tvar %s" % tvar_name)


class GenericValuesResolver(object):
    @classmethod
    def resolve_generic_values(cls, typ: type, generic_values_container: typing.Dict[str, type]):
        """Resolve any generics within type specification and return type that doesn't contain any generic variables
        :param typ: The type to be resolved (can be basic type like "int" or generic type like "typing.List[int]" or typing.List['T']
        :param generic_values_container: A
        :return:
        """
        if isinstance(typ, str):
            # We have some inconsistency and here we try to address it.
            # '~T' means instance of T while List['T'] means list of instances of T
            # but it's without '~', we always want to resolve to instance type
            if typ.startswith("~"):
                typ = typ[1:]
            if typ not in generic_values_container:
                raise UnresolvedTvarError(typ)
            return generic_values_container[typ]
        if CspTypingUtils.is_forward_ref(typ):
            return cls.resolve_generic_values(typ.__forward_arg__, generic_values_container)
        elif isinstance(typ, typing.TypeVar):
            return cls.resolve_generic_values(typ.__name__, generic_values_container)
        elif isinstance(typ, OutputBasketContainer):
            return OutputBasketContainer(
                typ=cls.resolve_generic_values(typ.typ, generic_values_container),
                shape=typ.shape,
                eval_type=typ.eval_type,
                shape_func=typ.shape_func,
                **typ.ast_kwargs,
            )
        elif CspTypingUtils.is_generic_container(typ):
            orig_t = CspTypingUtils.get_origin(typ)
            resolved_args = []
            resolved_any = False
            for arg in typ.__args__:
                arg_new = cls.resolve_generic_values(arg, generic_values_container)
                resolved_args.append(arg_new)
                if arg_new is not arg:
                    resolved_any = True
            if resolved_any:
                return orig_t[tuple(resolved_args)]
            else:
                return typ

        return typ
