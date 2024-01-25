import inspect
import threading
from abc import ABCMeta
from contextlib import contextmanager
from functools import wraps

from csp.impl.constants import UNSET

__all__ = [
    "register_injected_object",
    "register_injected_provider",
    "set_new_registry_thread_instance",
    "get_cur_registry",
    "set_existing_registry_thread_instance",
    "Injected",
    "get_injected_value",
    "auto_inject",
    "injected_property",
    "AutoInjectableMeta",
    "AutoInjectable",
]


class ObjectFactoryRegistry:
    """A container of the injected object factories. It stores a hierarchy of injected factories.

    Usually should not be used directly by the user, other convenience functions in this file wrap the functionality of this class.

    Notes about key lookup and contexts:
        - By default initially the global context is used
        - A stack of thread local instances of ObjectFactory is maintained
        - A new thread local is created using call to set_new_thread_instance. By default the new instance has read access to all parent scopes
          but any new registered factory will not be visible in the parent
        - An existing factory can be set using set_existing_thread_instance. The use cases for this are rare, one of the main use cases is to
          replicate in a child thread the scope of the parent thread if needed so they share the factory namespace resolution.
    Thread safety notes:
        This class is in general NOT thread safe. The only valid usage of this class across threads is if the parent thread sets up the keys
        and then all threads access these keys to read.
    Example usage:
        class A:
            pass

        class B:
            def __init__(self, value):
                self.value = value

        ObjectFactoryRegistry.instance()['a_provider'] = A
        ObjectFactoryRegistry.instance()['a_provider2'] = lambda: A()
        ObjectFactoryRegistry.instance()['b_provider'] = B
        a_instance = A()
        b_instance = B(42)
        ObjectFactoryRegistry.instance()['my_singletons.a'] = lambda: a_instance
        ObjectFactoryRegistry.instance()['my_singletons.b'] = lambda: b_instance

        a_singleton = ObjectFactoryRegistry.instance()['my_singletons.a']()
        assert ObjectFactoryRegistry.instance()['my_singletons.a']() is a_singleton
        a1 = ObjectFactoryRegistry.instance()['a_provider']()
        a2 = ObjectFactoryRegistry.instance()['a_provider']()
        a3 = ObjectFactoryRegistry.instance()['a_provider2']()
        a4 = ObjectFactoryRegistry.instance()['a_provider2']()
        # All the "a" objects are of type A
        assert isinstance(a1, A) and isinstance(a2, A) and isinstance(a3, A) and isinstance(a4, A)
        # They are all different objects:
        assert len(set([id(a_singleton), id(a1), id(a2), id(a3), id(a4)])) == 5
        b_singleton = ObjectFactoryRegistry.instance()['my_singletons.b']()
        b1 = ObjectFactoryRegistry.instance()['b_provider'](1)
        b2 = ObjectFactoryRegistry.instance()['b_provider'](2)
        # All the "b" objects are of type B
        assert isinstance(b1, B) and isinstance(b2, B)
        # They are all different objects:
        assert len(set([id(b_singleton), id(b1), id(b2)])) == 3
        with ObjectFactoryRegistry().set_new_thread_instance() as instance:
            assert instance is ObjectFactoryRegistry.instance()
            # The parent keys are visible in the child by default
            assert ObjectFactoryRegistry.instance()['my_singletons.a']() is a_instance
            ObjectFactoryRegistry.instance()['my_singletons.a2'] = lambda: a2
            # We can set new singletons this way in the child:
            # Note that since it already exists in the parent, we must provide the override flag
            ObjectFactoryRegistry.instance().set_object_factory('my_singletons.a2', lambda: a2, allow_override=True)
            assert ObjectFactoryRegistry.instance().get_object_factory('my_singletons.a2')() is a2
            # The parent singletons are still visible:
            assert ObjectFactoryRegistry.instance().get_object_factory('my_singletons.a')() is a_singleton

        # We can also opt out of inheriting keys from parent
        with ObjectFactoryRegistry().set_new_thread_instance(False) as instance:
            assert instance is ObjectFactoryRegistry.instance()
            # now my_singletons.a is inaccessible
            assert ObjectFactoryRegistry.instance().get_object_factory('my_singletons.a', None) is None

        # The child factory doesn't exist here anymore so 'a2' can't be accessed
        assert ObjectFactoryRegistry.instance().get_object_factory('my_singletons.a2', None) is None

        # we can print the ObjectFactoryRegistry instance to see the hierarchy
        str(ObjectFactoryRegistry.instance())
    """

    _GLOBAL_INSTANCE = None
    _THREAD_INSTANCE = threading.local()

    def __init__(self, parent=None):
        super().__setattr__("_items", {})
        super().__setattr__("_parent", parent)

    def set_object_factory(self, key, value, allow_override=False):
        """Register a new object factory
        :param key: A hierarchical key like "a.b.c"
        :param value: The registered factory, must be callable
        :param allow_override: A boolean flag that specifies that override is allowed if the given key is already registered. If the key
        exists and allow_override is False then RuntimeError will be raised
        """
        assert (
            callable(value) or isinstance(value, ObjectFactoryRegistry)
        ), f"Registered injected object must be callable instead of {type(value)} (use register_injected_object) for non callable objects"

        if not allow_override and self._get_object_factory_impl(key, default_value=None) is not None:
            raise RuntimeError(f"Trying to override existing key {key} without allow_override set")
        assert isinstance(key, str)
        path = key.split(".")
        cur = self
        for sub_factory in path[:-1]:
            cur = cur.get_sub_registry(sub_factory, True)
        cur._items[path[-1]] = value

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, key):
        return self.get_object_factory(key)

    def __setitem__(self, key, value):
        self.set_object_factory(key, value)

    def _get_object_factory_impl(self, key, default_value=UNSET):
        try:
            if isinstance(key, str):
                path = key.split(".")
                cur = self
                for sub_factory in path[:-1]:
                    cur = cur._get_object_factory_impl(sub_factory)
                return cur._items[path[-1]]
            else:
                return self._items[key]
        except KeyError:
            if self._parent:
                return self._parent._get_object_factory_impl(key, default_value)
            if default_value is UNSET:
                raise KeyError(f"No registered object provider found for key '{key}'")
            else:
                return default_value

    def get_object_factory(self, key, default_value=UNSET):
        """Get a previously registered object factory
        :param key: A hierarchical key like "a.b.c"
        :param default_value: A default value if the factory key is not found, If the default_value is not provided and the key is not found
        then KeyError will be raised
        :return: The factory object corresponding to the given key
        """
        res = self._get_object_factory_impl(key, default_value)
        if isinstance(res, ObjectFactoryRegistry):
            raise KeyError(f"Trying to access item via partial path specification '{key}'")
        return res

    @property
    def parent(self):
        """
        :return: The parent fallback factory
        """
        return self._parent

    @property
    def keys(self):
        """
        :return: The top level keys of the factory registry
        """
        return sorted(self._items.keys(), key=lambda k: str(k))

    def clear(self):
        """Clear all keys from the factory registry.
        Note: If there is a parent registry, those values still remain and can be accessed for read.
        """
        self._items.clear()

    def _to_string(self, rows=None, level=0):
        if rows is None:
            rows = []
        single_space = " " * 4
        space = single_space * (level + 1)
        for k in self.keys:
            v = self._items[k]
            if isinstance(v, ObjectFactoryRegistry):
                rows.append(f"{space}{k}:")
                v._to_string(rows, level + 1)
            else:
                v_str = str(v)
                if "\n" in v_str:
                    rows.append(f"{space}{k}:")
                    for v_row in v_str.split("\n"):
                        rows.append(f"{space}{single_space}{v_row}")
                else:
                    rows.append(f"{space}{k}:{v_str}")
        if level == 0:
            joined_rows = "\n".join(rows)
            return f"ObjectFactoryRegistry(\n{joined_rows}\n)"

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return str(self)

    @classmethod
    @contextmanager
    def set_existing_thread_instance(cls, instance):
        """Set existing registry instance to be active for the current thread
        :param instance:
        """
        prev = getattr(cls._THREAD_INSTANCE, "instance", None)
        cls._THREAD_INSTANCE.instance = instance
        try:
            yield cls._THREAD_INSTANCE.instance
        finally:
            if prev is None:
                delattr(cls._THREAD_INSTANCE, "instance")
            else:
                cls._THREAD_INSTANCE.instance = prev

    @classmethod
    @contextmanager
    def set_new_thread_instance(cls, inherit_from_parent=True):
        """Create a new thread local registry instance
        :param inherit_from_parent: If True, the parent (current instance) keys are accessbile for read.
                                    If False, then the new instance have no parent.
        """
        prev = getattr(cls._THREAD_INSTANCE, "instance", None)
        parent = prev or cls._GLOBAL_INSTANCE if inherit_from_parent else None
        with cls.set_existing_thread_instance(ObjectFactoryRegistry(parent)) as instance:
            yield instance

    @classmethod
    def global_instance(cls):
        """
        :return: The global, unique non thread local registry instance
        """
        return cls._GLOBAL_INSTANCE

    @classmethod
    def instance(cls):
        """
        :return: The current registry instance (thread local if it's set or global if no thread local instance set)
        """
        res = getattr(cls._THREAD_INSTANCE, "instance", None)
        if res:
            return res

        return cls._GLOBAL_INSTANCE

    def get_sub_registry(self, key, allow_create=False):
        """Get a sub-registry of the registry. If we have 'a.b.c' in the registry then we can access sub registry 'a' from top level, 'b' from
        'a' registry and so on
        :param key: The key of the registry. NOTE: 'a.b' can't be specified, we must specify non hierarchical key here
        :param allow_create: Allow creating of sub registry if it doesn't exist
        :return:
        """
        res = self._items.get(key)
        if res is None:
            if allow_create:
                res = ObjectFactoryRegistry()
                self._items[key] = res
        elif not isinstance(res, ObjectFactoryRegistry):
            raise ValueError(
                f"Trying to access sub factory {key} when object of type {type(res)} already exists with same key"
            )
        return res


ObjectFactoryRegistry._GLOBAL_INSTANCE = ObjectFactoryRegistry()


def register_injected_object(key, value, allow_override=False):
    """Registers an object for a given key. This can be retrieved by get_injected_value for example
    :param key: The hierarchical key of the injected object. Like "a.b.c"
    :param value: The value to be injected (object of any type)
    :param allow_override: A boolean flag that specifies that override is allowed if the given key is already registered. If the key
        exists in current or any of the parent registries and allow_override is False then RuntimeError will be raised.
    :return:
    """
    ObjectFactoryRegistry.instance().set_object_factory(key, lambda: value, allow_override)


def register_injected_provider(key, value, singleton=False, allow_override=False):
    """Similar to register_injected_object but instead of registering an object for the given key, registers a callable that will be called
    to provide values for the given key
    :param key: The hierarchical key of the injected object. Like "a.b.c"
    :param value: A callable that provides values to be injected (object of any type)
    :param singleton: A boolean flag. If True, the "value" function will be called only once and all future accesses to value will return the
    same value, otherwise the function will be called for each value.
    :param allow_override: A boolean flag that specifies that override is allowed if the given key is already registered. If the key
        exists and allow_override is False then RuntimeError will be raised
    :return:
    """
    if singleton:

        class _Singleton:
            def __call__(self):
                if not hasattr(self, "_value"):
                    self._value = value()
                return self._value

        ObjectFactoryRegistry.instance().set_object_factory(key, _Singleton(), allow_override)
    else:
        ObjectFactoryRegistry.instance().set_object_factory(key, value, allow_override)


def get_injected_value(key, *args, default_value=UNSET, **kwargs):
    """Retrieves the value of injected object.
    If the object is expected to be registered using register_injected_object, then the function should not provide any args/kwargs. On
    the other hand if the object is expected to be produced by a factory registered by register_injected_provider then additional args/kwargs
    can be passed that will be forwarded to the provider function and the result will be returned.

    :param key: The hierarchical key of the injected object. Like "a.b.c"
    :param args: Positional arguments that should be passed to the provider function.
    :param default_value: A default value if the factory key is not found, If the default_value is not provided and the key is not found
        then KeyError will be raised
    :param kwargs: Keyword arguments that should be passed to the provider function.
    :return:
    """
    if default_value is UNSET:
        return ObjectFactoryRegistry.instance()[key](*args, **kwargs)
    else:
        provider = ObjectFactoryRegistry.instance().get_object_factory(key, default_value=None)
        if provider is None:
            return default_value
        else:
            return provider(*args, **kwargs)


def set_new_registry_thread_instance(inherit_from_parent=True):
    """Create a new thread local registry instance return a context manager.
    :param inherit_from_parent: If True, the parent (current instance) keys are accessbile for read, if False, then the new instance have no parent.

    Usage:
        with set_new_registry_thread_instance() as registry:
            register_injected_object(...)
            ...
    """
    return ObjectFactoryRegistry.set_new_thread_instance(inherit_from_parent)


def get_cur_registry():
    """Get the current registry instance. Usually should not be used, should mostly be used for copying the registry object to another thread.
    :return: The current registry instance (thread local if it's set or global if no thread local instance set).
    """
    return ObjectFactoryRegistry.instance()


def set_existing_registry_thread_instance(cls, instance):
    """Set existing registry instance to be active for the current thread. The main use is to call on a new thread with an instance retrieved
    by get_cur_registry() in the parent thread.
    :param instance:
    """
    return ObjectFactoryRegistry.set_existing_thread_instance(instance)


class Injected:
    """An injected value that can be used as a default argument for a function.
    Example:
        def power_calc(x, y):
            return x ** y

        @auto_inject
        def f(input_value=Injected('power_calc', 3)):
            return input_value

        @auto_inject
        def f2(input_value=Injected('my_magic_constant')):
            return input_value

        self.assertEqual(f(10), 10)

        with set_new_registry_thread_instance():
            register_injected_provider('power_calc', lambda y: power_calc(2, y))
            register_injected_object('my_magic_constant', 42)

            print(f(10)) # prints 10 since we passed in explicit value
            print(f()) # prints 8 since injected value is evaluated to 2**3
            register_injected_provider('power_calc', lambda y: power_calc(3, y), allow_override=True)
            print(f()) # prints 9 since injected value is evaluated to 9**3
            print(f2()) # prints the magic constant 42
            print(f2(123)) # prints the parameter that was passed in, i.e 123
    """

    def __init__(self, key, *args, **kwargs):
        self._key = key
        self._args = args
        self._kwargs = kwargs

    def __str__(self):
        if self._args:
            args_str = "," + ",".join(str(v) for v in self._args)
        else:
            args_str = ""
        if self._kwargs:
            kwargs_str = "," + ",".join(f"{k}={str(v)}" for k, v in self._kwargs.items())
        else:
            kwargs_str = ""
        return f"Injected({str(self._key)}{args_str}{kwargs_str})"

    def __repr__(self):
        return str(self)

    @property
    def value(self):
        """
        :return: The resolved injected value
        """
        res = ObjectFactoryRegistry.instance()[self._key]
        if not callable(res):
            raise KeyError(f"Unable to resolve injected object {str(self._key)}")
        return res(*self._args, **self._kwargs)


class _InjectedArg:
    def __init__(self, index, name, injected_value):
        self.index = index
        self.name = name
        self.injected_value = injected_value


def auto_inject(f):
    """A decorator that resolves Injected default values for a function
    Example usage:
        @auto_inject
        def f(input_value=Injected('power_calc', 3)):
            return input_value
    For a detailed example please see the documentation of Injected.

    :param f: Decorated function
    :return:
    """
    spec = inspect.getfullargspec(f)
    injected_by_index = []
    injected_by_keyword = []

    if spec.defaults:
        skipped_args = len(spec.args) - len(spec.defaults)
        for i, (arg, default_val) in enumerate(zip(spec.args[skipped_args:], spec.defaults)):
            if isinstance(default_val, Injected):
                injected_by_index.append(_InjectedArg(i + skipped_args, arg, default_val))
    if spec.kwonlydefaults:
        for k, v in spec.kwonlydefaults.items():
            if isinstance(default_val, Injected):
                injected_by_keyword.append(_InjectedArg(-1, k, v))

    @wraps(f)
    def _impl(*args, **kwargs):
        positional_index = 0
        # Skip injecting arguments that were overridden by positional
        while positional_index < len(injected_by_index) and injected_by_index[positional_index].index < len(args):
            positional_index += 1
        while positional_index < len(injected_by_index):
            cur_arg = injected_by_index[positional_index]
            if cur_arg.name not in kwargs:
                kwargs[cur_arg.name] = cur_arg.injected_value.value
            positional_index += 1
        for cur_arg in injected_by_keyword:
            if cur_arg.name not in kwargs:
                kwargs[cur_arg.name] = cur_arg.injected_value.value
        return f(*args, **kwargs)

    return _impl


def injected_property(func=None, *args, inject_key=None, read_only=True, default_value=UNSET, **kwargs):
    """A decorator that creates injected properties or properties that depend on injected values.
    The properties are resolved in a lazy fashion, i.e on first call. The value persists after the first evaluation.

    There are 2 main usages of injected_property:
        Inject value directly from the registry:
            property1 = injected_property(inject_key='property1_value')
        Make calculation based on the values that are injected from the registry:
            @injected_property
            def property2(self):
                return self.property1 + get_injected_value('power_calc', 2, 3)

    Example usage:
        def power_calc(x, y):
            return x ** y


        class C1:
            def __str__(self):
                return f"{self.__class__.__name__}(id={id(self)})"

            def __repr__(self):
                return str(self)


        class C2(C1):
            pass


        class A(AutoInjectable):
            property1 = injected_property(inject_key='property1_value')

            @injected_property(read_only=False)
            def property2(self):
                return self.property1.__class__.__name__ + ' ' + str(get_injected_value('power_calc', 2, 3))

            property3 = injected_property(inject_key='power_calc', x=5, y=2)


        a1 = A()
        a2 = A()
        a3 = A()
        with set_new_registry_thread_instance():
            register_injected_provider('property1_value', C1)
            register_injected_provider('power_calc', power_calc)
            print(a1.property1)  # Print C1(id=139706871169376)
            print(a1.property1)  # Print C1(id=139706871169376) - the id is the same as above since it's the same object
            print(a1.property2)  # Print 'C1 8'
            print(a1.property3)  # Print 25, which equals to 5**2
            print(a2.property1)  # Print C1(id=140279271190992)
            a2.property2 = 50
            print(a2.property2)  # Print 50 - the value was set to 50, so the body of the property is not evaluated
        with set_new_registry_thread_instance():
            register_injected_provider('property1_value', C2)
            register_injected_provider('power_calc', power_calc)
            print(a3.property1)  # C2(id=140279271190712)
        # If we print any of the properties here again, they will be equal to the previously printed values since their values already
        # resolved

    :param func: The decorated function or None
    :param args: Can only be specified if func is None, these are the positional arguments that will be passed to object factory
    :param inject_key: Can only be specified if func is None. This is the key of the injected object factory.
    :param read_only: A boolean flag that specifies whether the property value can be set or read only.
    :param default_value: Can only be specified if func is None. The default value for property if no factory exists with inject_key.
    :param kwargs: Can only be specified if func is None, these are the keyword arguments that will be passed to object factory
    """
    if func is not None and not callable(func):
        assert isinstance(func, str)
        assert inject_key is None
        inject_key = func
        func = None

    if func is not None:
        assert not args, "func and args can't be set at the same time"
        assert inject_key is None, "func and inject_key can't be set at the same time"
        assert default_value is UNSET, "func and default_value can't be set at the same time"
        assert not kwargs, "func and kwargs can't be set at the same time"

    def _impl(func):
        class PropertyValue:
            def get_value(self, obj):
                assert (
                    type(type(obj)) is AutoInjectableMeta
                ), "Trying to access injected_property on object with non AutoInjectableMeta metaclass"
                injected_property_values = obj._get_injected_property_values()

                res = injected_property_values.get(id(self), UNSET)
                if res is UNSET:
                    res = func(obj)
                    injected_property_values[id(self)] = res
                return res

            def set_value(self, obj, value):
                assert (
                    type(type(obj)) is AutoInjectableMeta
                ), "Trying to access injected_property on object with non AutoInjectableMeta metaclass"
                obj._get_injected_property_values()[id(self)] = value

        prop_value_container = PropertyValue()
        res = property(lambda self: prop_value_container.get_value(self))
        if not read_only:
            res = res.setter(lambda self, value: prop_value_container.set_value(self, value))
        return res

    if func is not None:
        assert inject_key is None, "inject_key can't be provided with implementation function"
        assert not args, "Invalid call to injected_property, when positional args are provided, inject_key must be first positional argument"
        return _impl(func)
    elif inject_key is not None:
        return _impl(lambda self: get_injected_value(inject_key, *args, **kwargs))
    else:
        return _impl


class AutoInjectableMeta(ABCMeta):
    """A metaclass that provides utilities for automatic injection

    The class provides the following functionality:
        1. Support for using injected_property
        2. Support for using Inject default value on all functions without the need to use the auto_inject decorator.

    """

    def __new__(cls, name, bases, dct):
        new_dct = dict(dct)

        for k, v in dct.items():
            if inspect.isfunction(v):
                new_dct[k] = auto_inject(v)

        def _get_injected_property_values(self):
            if getattr(self, "_injected_property_values", None) is None:
                self._injected_property_values = {}
            return self._injected_property_values

        new_dct["_get_injected_property_values"] = _get_injected_property_values

        return super().__new__(cls, name, bases, new_dct)


class AutoInjectable(metaclass=AutoInjectableMeta):
    """A helper baseclass that sets metaclass to AutoInjectableMeta"""

    pass
