The following boolean and mathematical operations are available which should be self explanatory.

Also note that there is syntactic sugar in place when wiring a graph.
Edges have most operators overloaded includes `+`, `-`, `*`, `/`, `**`, `>`, `>=`, `<`, `<=`, `==`, `!=`, so you can have code like `csp.const(1) + csp.const(2)` work properly.
Right hand side values will also automatically be upgraded to `csp.const(<value>)` if its detected that its not an edge, so something like `x = csp.const(1) + 2` will work as well.

## Table of Contents

1. Binary logical operators

- **`csp.not_(ts[bool]) → ts[bool]`**
- **`csp.and_(x: [ts[bool]]) → ts[bool]`**
- **`csp.or_(x: [ts[bool]]) → ts[bool]`**

2. Binary mathematical operators

- **`csp.add(x: ts['T'], y: ts['T']) → ts['T']`**
- **`csp.sub(x: ts['T'], y: ts['T']) → ts['T']`**
- **`csp.multiply(x: ts['T'], y: ts['T']) → ts['T']`**
- **`csp.divide(x: ts['T'], y: ts['T']) → ts[float]`**
- **`csp.pow(x: ts['T'], y: ts['T']) → ts['T']`**
- **`csp.min/max(x: ts['T'], y: ts['T']) → ts['T']`**
- **`gt/ge/lt/le/eq/ne(x: ts['T'], y: ts['T']) → ts[bool]`**

3. Unary mathematical operators

- **`ln/log2/log10(x: ts[float]) → ts[float]`**
- **`exp/exp2(x: ts[float]) → ts[float]`**
- **`sqrt(x: ts[float]) → ts[float]`**
- **`abs(x: ts[float]) → ts[float]`**
- **`sin/cos/tan/arcsin/arccos/arctan/sinh/cosh/tanh/arcsinh/arccosh/arctanh(x: ts[float]) → ts[float]`**
- **`erf(x: ts[float]) → ts[float]`**

Many of these are also exposed as dunder operators:

a. Operators

- **`__add__`**
- **`__sub__`**
- **`__mul__`**
- **`__truediv__`**
- **`__floordiv__`**
- **`__pow__`**
- **`__invert__`**: bitwise not
- **`__mod__`**
- **`__abs__`**
- **`__pos__`**
- **`__neg__`**

a. Comparators

- **`__gt__`**
- **`__ge__`**
- **`__lt__`**
- **`__le__`**
- **`__eq__`**
- **`__ne__`**
