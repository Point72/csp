import csp
import pytest

class Base(csp.Struct):
    x: int

class Derived(Base):
    y: str  # only new fields, no need to re-override x

class MoreDerived(Derived):
    z: float

class Base2(csp.Struct):
    data: Base

class Derived2(Base2):
    data: Derived  # override with subclass (allowed)


def test_compatible_override():
    d = Derived(x=1, y="hi")
    d2 = Derived2(data=d)

    assert isinstance(d2.data, Derived)
    assert d2.data.x == 1
    assert d2.data.y == "hi"


def test_from_dict_override():
    obj = Derived2.from_dict({"data": {"x": 5, "y": "abc"}})
    assert isinstance(obj.data, Derived)
    assert obj.data.x == 5
    assert obj.data.y == "abc"



def test_incompatible_override():
    with pytest.raises(TypeError):
        class BadOverride(Base2):
            data: str  # not a subclass → should fail
