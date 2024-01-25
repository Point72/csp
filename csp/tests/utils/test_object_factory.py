import itertools
import unittest

from csp.utils.object_factory_registry import (
    AutoInjectable,
    Injected,
    ObjectFactoryRegistry,
    auto_inject,
    get_injected_value,
    injected_property,
    register_injected_object,
    register_injected_provider,
    set_new_registry_thread_instance,
)


class TestObjectFactory(unittest.TestCase):
    def setUp(self):
        if hasattr(ObjectFactoryRegistry._THREAD_INSTANCE, "instance"):
            delattr(ObjectFactoryRegistry._THREAD_INSTANCE, "instance")
        ObjectFactoryRegistry.instance().clear()

    def get_running_index_func(self, start=0):
        count = itertools.count(start)
        return lambda: next(count)

    def test_doc_string_example(self):
        class A:
            pass

        class B:
            def __init__(self, value):
                self.value = value

        ObjectFactoryRegistry.instance()["a_provider"] = A
        ObjectFactoryRegistry.instance()["a_provider2"] = lambda: A()
        ObjectFactoryRegistry.instance()["b_provider"] = B
        a_instance = A()
        b_instance = B(42)
        ObjectFactoryRegistry.instance()["my_singletons.a"] = lambda: a_instance
        ObjectFactoryRegistry.instance()["my_singletons.b"] = lambda: b_instance

        a_singleton = ObjectFactoryRegistry.instance()["my_singletons.a"]()
        assert ObjectFactoryRegistry.instance()["my_singletons.a"]() is a_singleton
        a1 = ObjectFactoryRegistry.instance()["a_provider"]()
        a2 = ObjectFactoryRegistry.instance()["a_provider"]()
        a3 = ObjectFactoryRegistry.instance()["a_provider2"]()
        a4 = ObjectFactoryRegistry.instance()["a_provider2"]()
        # All the "a" objects are of type A
        assert isinstance(a1, A) and isinstance(a2, A) and isinstance(a3, A) and isinstance(a4, A)
        # They are all different objects:
        assert len(set([id(a_singleton), id(a1), id(a2), id(a3), id(a4)])) == 5
        b_singleton = ObjectFactoryRegistry.instance()["my_singletons.b"]()
        b1 = ObjectFactoryRegistry.instance()["b_provider"](1)
        b2 = ObjectFactoryRegistry.instance()["b_provider"](2)
        # All the "b" objects are of type B
        assert isinstance(b1, B) and isinstance(b2, B)
        # They are all different objects:
        assert len(set([id(b_singleton), id(b1), id(b2)])) == 3
        with ObjectFactoryRegistry().set_new_thread_instance() as instance:
            assert instance is ObjectFactoryRegistry.instance()
            # The parent keys are visible in the child by default
            assert ObjectFactoryRegistry.instance()["my_singletons.a"]() is a_instance
            ObjectFactoryRegistry.instance()["my_singletons.a2"] = lambda: a2
            # We can set new singletons this way in the child:
            ObjectFactoryRegistry.instance().set_object_factory("my_singletons.a2", lambda: a2, allow_override=True)
            assert ObjectFactoryRegistry.instance().get_object_factory("my_singletons.a2")() is a2
            # The parent singletons are still visible:
            assert ObjectFactoryRegistry.instance().get_object_factory("my_singletons.a")() is a_singleton

        # We can also opt out of inheriting keys from parent
        with ObjectFactoryRegistry().set_new_thread_instance(False) as instance:
            assert instance is ObjectFactoryRegistry.instance()
            # now my_singletons.a is inaccessible
            assert ObjectFactoryRegistry.instance().get_object_factory("my_singletons.a", None) is None

        # The child factory doesn't exist here anymore so 'a2' can't be accessed
        assert ObjectFactoryRegistry.instance().get_object_factory("my_singletons.a2", None) is None
        # we can print the ObjectFactory instance to see the hierarchy
        # str(ObjectFactory.instance())

    def test_basic(self):
        self.assertEqual(ObjectFactoryRegistry.instance(), ObjectFactoryRegistry.global_instance())
        with self.assertRaises(KeyError):
            ObjectFactoryRegistry.instance()["running_index"]
        ObjectFactoryRegistry.instance()["running_index"] = self.get_running_index_func()
        ObjectFactoryRegistry.instance()["constant_value"] = lambda: 42
        self.assertEqual(ObjectFactoryRegistry.instance()["running_index"](), 0)
        self.assertEqual(ObjectFactoryRegistry.instance()["running_index"](), 1)
        self.assertEqual(ObjectFactoryRegistry.instance()["constant_value"](), 42)
        self.assertEqual(ObjectFactoryRegistry.instance()["constant_value"](), 42)
        with ObjectFactoryRegistry.set_new_thread_instance():
            level_1_thread_instance = ObjectFactoryRegistry.instance()
            self.assertEqual(ObjectFactoryRegistry.instance().parent, ObjectFactoryRegistry.global_instance())
            self.assertEqual(ObjectFactoryRegistry.instance()["running_index"](), 2)
            self.assertEqual(ObjectFactoryRegistry.instance()["constant_value"](), 42)
            ObjectFactoryRegistry.instance()["running_index2"] = self.get_running_index_func(100)
            ObjectFactoryRegistry.instance()["sub_category.running_index3"] = self.get_running_index_func(200)
            self.assertEqual(ObjectFactoryRegistry.instance()["running_index2"](), 100)
            self.assertEqual(ObjectFactoryRegistry.instance()["sub_category.running_index3"](), 200)
            self.assertEqual(ObjectFactoryRegistry.instance().get_sub_registry("sub_category")["running_index3"](), 201)
            with ObjectFactoryRegistry.set_new_thread_instance():
                self.assertEqual(ObjectFactoryRegistry.instance().parent, level_1_thread_instance)
                self.assertEqual(ObjectFactoryRegistry.instance()["running_index2"](), 101)
            with ObjectFactoryRegistry.set_new_thread_instance(False):
                self.assertIsNone(ObjectFactoryRegistry.instance().parent)
                with self.assertRaises(KeyError):
                    ObjectFactoryRegistry.instance()["running_index2"]()

            with self.assertRaises(KeyError):
                ObjectFactoryRegistry.instance()["running_index3"]()
        with self.assertRaises(KeyError):
            ObjectFactoryRegistry.instance()["running_index2"]
        with self.assertRaises(KeyError):
            ObjectFactoryRegistry.instance()["sub_category.running_index3"]
        self.assertEqual(ObjectFactoryRegistry.instance()["running_index"](), 3)
        self.assertEqual(ObjectFactoryRegistry.instance().running_index(), 4)
        with self.assertRaises(KeyError):
            ObjectFactoryRegistry.instance().get_object_factory("running_index3")
        self.assertEqual(ObjectFactoryRegistry.instance().get_object_factory("running_index3", lambda: 43)(), 43)

    def test_injected_function_values(self):
        @auto_inject
        def f(x=Injected("injected_x"), y=Injected("injected_y")):
            return x + y

        def simple_eval():
            self.assertEqual(f(1, 2), 3)
            self.assertEqual(f(x=1, y=2), 3)
            self.assertEqual(f(1, y=2), 3)

        simple_eval()
        with self.assertRaises(KeyError):
            f(1)
        with self.assertRaises(KeyError):
            f(x=1)
        with self.assertRaises(KeyError):
            f(y=2)

        with set_new_registry_thread_instance():
            register_injected_object("injected_x", 42)
            simple_eval()
            # Still no y
            with self.assertRaises(KeyError):
                f(1)
            with self.assertRaises(KeyError):
                f(x=1)
            self.assertEqual(f(y=2), 44)
            register_injected_provider("injected_y", self.get_running_index_func())
            self.assertEqual(f(), 42)
            self.assertEqual(f(), 43)
            with self.assertRaises(RuntimeError):
                register_injected_object("injected_y", 0)
            register_injected_object("injected_y", 0, allow_override=True)
            self.assertEqual(f(), 42)
            self.assertEqual(f(), 42)
            register_injected_provider(
                "injected_y", self.get_running_index_func(3), singleton=True, allow_override=True
            )
            self.assertEqual(f(), 45)
            self.assertEqual(f(), 45)

        with set_new_registry_thread_instance():
            register_injected_object("injected_y", 42)
            simple_eval()
            with self.assertRaises(KeyError):
                f()
            self.assertEqual(f(1), 43)
            self.assertEqual(f(x=1), 43)

        def power_calc(x, y):
            return x**y

        @auto_inject
        def f(input_value=Injected("power_calc", 3)):
            return input_value

        @auto_inject
        def f2(input_value=Injected("my_magic_constant")):
            return input_value

        self.assertEqual(f(10), 10)

        with set_new_registry_thread_instance():
            register_injected_provider("power_calc", lambda y: power_calc(2, y))
            register_injected_object("my_magic_constant", 42)
            self.assertEqual(f(10), 10)
            self.assertEqual(f(), 2**3)
            register_injected_provider("power_calc", lambda y: power_calc(3, y), allow_override=True)
            self.assertEqual(f(), 3**3)
            self.assertEqual(f2(), 42)
            self.assertEqual(f2(123), 123)

    def test_injected_classes(self):
        def power_calc(x, y):
            return x**y

        class A(AutoInjectable):
            property1 = injected_property(inject_key="property1_value")

            @injected_property(read_only=False)
            def property2(self):
                return self.property1 + get_injected_value("power_calc", 2, 3)

            property3 = injected_property(inject_key="power_calc", x=5, y=2)

            def f(self, x=Injected("property1_value")):
                return x

        a1 = A()
        a2 = A()
        a3 = A()
        with set_new_registry_thread_instance():
            register_injected_provider("property1_value", self.get_running_index_func())
            register_injected_provider("power_calc", power_calc)
            self.assertEqual(a1.property1, 0)
            self.assertEqual(a1.property2, 8)
            self.assertEqual(a1.property3, 25)
            self.assertEqual(a2.property1, 1)
            a2.property2 = 50
            self.assertEqual(a2.property2, 50)
            self.assertEqual(a2.property3, 25)
            self.assertEqual(a2.f(), 2)
        with set_new_registry_thread_instance():
            register_injected_provider("property1_value", self.get_running_index_func(100))
            register_injected_provider("power_calc", power_calc)
            self.assertEqual(a3.property1, 100)
            self.assertEqual(a3.property2, 108)
            self.assertEqual(a3.property3, 25)

        self.assertEqual(a1.property1, 0)
        self.assertEqual(a1.property2, 8)
        self.assertEqual(a1.property3, 25)
        self.assertEqual(a2.property1, 1)
        self.assertEqual(a2.property2, 50)
        self.assertEqual(a2.property3, 25)
        self.assertEqual(a3.property1, 100)
        self.assertEqual(a3.property2, 108)
        self.assertEqual(a3.property3, 25)

    def test_override_parent_key(self):
        register_injected_object("my_magic_constant", 0)
        with set_new_registry_thread_instance():
            with self.assertRaises(RuntimeError):
                register_injected_object("my_magic_constant", 42)
            self.assertEqual(get_injected_value("my_magic_constant"), 0)
            register_injected_object("my_magic_constant", 42, allow_override=True)
            self.assertEqual(get_injected_value("my_magic_constant"), 42)
        self.assertEqual(get_injected_value("my_magic_constant"), 0)


if __name__ == "__main__":
    unittest.main()
