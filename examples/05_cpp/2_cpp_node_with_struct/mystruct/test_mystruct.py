from datetime import datetime

import csp
from mystruct.node import use_struct_generic, use_struct_specific
from mystruct.struct import MyStruct


@csp.graph
def simple_graph() -> csp.Outputs(generic=csp.ts[MyStruct], specific=csp.ts[MyStruct]):
    """Test graph that uses both generic and specific struct nodes."""
    st = csp.const(MyStruct(a=1, b="abc"))
    generic = use_struct_generic(st)
    specific = use_struct_specific(generic)
    csp.output(generic=generic, specific=specific)


def test_mystruct():
    start = datetime(2020, 1, 1)
    ret = csp.run(simple_graph, starttime=start)

    # Graph should return generic and specific outputs
    assert "generic" in ret
    assert "specific" in ret
    assert len(ret["generic"]) == 1
    assert len(ret["specific"]) == 1

    # Verify the struct values by accessing fields directly
    generic_value = ret["generic"][0][1]
    specific_value = ret["specific"][0][1]

    assert isinstance(generic_value, MyStruct)
    assert isinstance(specific_value, MyStruct)

    # use_struct_generic passes through unchanged
    assert generic_value.a == 1
    assert generic_value.b == "abc"

    # use_struct_specific uppercases the 'b' field
    assert specific_value.a == 1
    assert specific_value.b == "ABC"


def test_struct_creation():
    """Basic test that struct creation and field access works."""
    s = MyStruct(a=42, b="hello")
    assert s.a == 42
    assert s.b == "hello"


def test_node_import():
    """Test that the C++ node can be imported."""
    from mystruct.node import use_struct_generic, use_struct_specific

    assert use_struct_generic is not None
    assert use_struct_specific is not None
