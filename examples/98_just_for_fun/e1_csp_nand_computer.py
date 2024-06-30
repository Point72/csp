"""
The purpose of this example is to help demonstrate the difference between csp.node and csp.graph concepts (and because its awesome).
We define a single node, which is a simple NAND gate. We then wire up the NAND gate nodes in more and more complex constructs under csp.graph definitions
to build up our "computer".

csp.graph calls only define the wiring of the NAND gate nodes. At runtime, the only things that exist and execute are the NAND nodes.

The wiring done here has been taken from the awesome book  https://www.amazon.com/Elements-Computing-Systems-Building-Principles/dp/0262640686
"""

from datetime import datetime
from typing import List

import csp
import csp.showgraph
from csp import ts


@csp.node
def nand(a: ts[bool], b: ts[bool]) -> ts[bool]:
    """the one and only node in this application!"""
    if csp.ticked(a, b) and csp.valid(a, b):
        return not (a and b)


@csp.graph
def not_(a: ts[bool]) -> ts[bool]:
    return nand(a, a)


@csp.graph
def and_(a: ts[bool], b: ts[bool]) -> ts[bool]:
    return not_(nand(a, b))


@csp.graph
def or_(a: ts[bool], b: ts[bool]) -> ts[bool]:
    return nand(not_(a), not_(b))


@csp.graph
def xor(a: ts[bool], b: ts[bool]) -> ts[bool]:
    return or_(and_(a, not_(b)), and_(not_(a), b))


@csp.graph
def half_adder(a: ts[bool], b: ts[bool]) -> csp.Outputs(sum=ts[bool], carry=ts[bool]):
    return csp.output(sum=xor(a, b), carry=and_(a, b))


@csp.graph
def full_adder(a: ts[bool], b: ts[bool], c: ts[bool]) -> csp.Outputs(sum=ts[bool], carry=ts[bool]):
    bc = half_adder(b, c)
    abc = half_adder(a, bc.sum)
    return csp.output(sum=abc.sum, carry=or_(abc.carry, bc.carry))


# Here we take in 16 "bits" as a basket of bools
@csp.graph
def addInt(a: [ts[bool]], b: [ts[bool]]) -> csp.OutputBasket(List[ts[bool]]):
    carry = csp.const(False)
    out = []
    for idx in range(len(a)):
        fa = full_adder(a[idx], b[idx], carry)
        out.append(fa.sum)
        carry = fa.carry

    return out


def number_to_basket(n: int, bits: int):
    s = "{0:b}".format(n).rjust(bits, "0")
    # if len(s) > bits:
    #    raise OverflowError()

    # We generate baskets where LSB is the [0] entry, MSB is the last
    return [csp.const(b == "1") for b in reversed(s)][:bits]


# These nodes are only used for conversion to int and string for display purposes
@csp.node
def basket_to_number(x: [ts[bool]]) -> ts[int]:
    if csp.ticked(x):
        return int("".join(str(int(v)) for v in reversed(list(x.validvalues()))), base=2)


@csp.node
def basket_to_bitstring(x: [ts[bool]]) -> ts[str]:
    if csp.ticked(x):
        return "".join(str(int(v)) for v in reversed(list(x.validvalues())))


@csp.graph
def my_graph(bits: int = 16):
    x = number_to_basket(42001, bits)
    y = number_to_basket(136, bits)

    csp.print("x", basket_to_number(x))
    csp.print("y", basket_to_number(y))
    csp.print("x_bits", basket_to_bitstring(x))
    csp.print("y_bits", basket_to_bitstring(y))

    add = addInt(x, y)

    csp.print("x+y", basket_to_number(add))
    csp.print("x+y_bits", basket_to_bitstring(add))


def main():
    # Show graph with 4-bit ints to limit size
    csp.showgraph.show_graph(my_graph, 4)

    csp.run(my_graph, starttime=datetime(2022, 6, 24))


if __name__ == "__main__":
    main()
