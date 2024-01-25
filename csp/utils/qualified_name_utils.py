import importlib


class QualifiedNameUtils:
    """A utility class to enable resolving strings to python objects and the other way around

    For example consider a type module1.sub_module.Class.Subclass. get_qualified_object_name will produce this path as string,
    while get_object_from_qualified_name will import it and return the actual class.
    """

    _QUALIFIED_NAMES_CACHE = {}
    NONE = object()

    @classmethod
    def get_qualified_object_name(cls, obj):
        return f"{obj.__module__}.{obj.__qualname__}"

    @classmethod
    def register_type(cls, typ, qualified_type_name=None):
        """Register an alias for a typ. For some types that aren't directly importable (such as
           types defined inside function) this registration is mandatory.
        :param typ:
        :param qualified_type_name:
        :return:
        """
        if qualified_type_name is None:
            qualified_type_name = cls.get_qualified_object_name(typ)
        cls._QUALIFIED_NAMES_CACHE[qualified_type_name] = typ

    @classmethod
    def get_object_from_qualified_name(cls, qualified_name: str):
        res = cls._QUALIFIED_NAMES_CACHE.get(qualified_name, cls.NONE)
        if res is not cls.NONE:
            return res

        idx = len(qualified_name)
        while True:
            module_name = qualified_name[:idx]
            try:
                module = importlib.import_module(module_name)
                item_name = qualified_name[idx + 1 :]
                break
            except ModuleNotFoundError as e:
                idx = qualified_name.rfind(".", 0, idx)
                if idx < 0:
                    raise TypeError(f"Unable to resolve type {qualified_name}") from e
        try:
            cur = module
            for cur_item in item_name.split("."):
                cur = getattr(cur, cur_item)

            cls._QUALIFIED_NAMES_CACHE[qualified_name] = cur
        except AttributeError as e:
            raise TypeError(f"Unable to resolve type {qualified_name}") from e
        return cur
