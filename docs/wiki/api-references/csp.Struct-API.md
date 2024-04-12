`csp.Struct` is a native, first-class CSP type that should be used for struct-like data ( key, values ). `csp.Struct` is implemented as a high performant C++ object and CSP C++ adapters are able to create / access them efficiently from C++.

In general its recommended to use `csp.Struct` for any object-like data required on your timeseries rather than plain old python object unless there is good reason not to.

## `csp.Struct` definition

`csp.Struct` types need to declare all their fields as annotated fields, for example:

```python
class MyData(csp.Struct):
    a: int
    b: str = 'default'
    c: list

    def some_method(self):
        return a * 2

>>> MyData()
MyData( a='<unset>', b='default', c='<unset>' )
```

The variables `a`, `b`, `c` here define the struct members (similar to `__slots__` on python objects, structs can not have any other attributes set on them other than the ones listed here).

**Defaults**: Note that members can be defined with default values, as `b` is here.

**Unset fields**:  Note that struct fields can be "unset". Fields can be checked for existence with `hasattr(o, field)` and can be removed with `del o.field`.

**Methods**: Note that you can define methods on structs just like any other python object.

## Special handling at graph time

While building your graph, if you have an edge that represents a `csp.Struct` type you can access a member of that struct at graph time. What this means is that when you do `edge.field` in your graph code, you will get a new edge that will tick with the value of that field. CSP will implicitly inject a `csp.node` that will extract that field whenever the struct timeseries ticks. Note that if the struct ticks and the field is unset, the field's edge will not tick. Here's an example of this in practice:

```python
import csp
from datetime import datetime

class Trade(csp.Struct):
    price: float
    size: int

@csp.graph
def my_graph():
    trades = csp.curve(Trade,
                       [(datetime(2020, 1, 1), Trade(price=100.01, size=200)),
                       (datetime(2020, 1, 1, 0, 0, 1), Trade(price=100.01, size=300))]
             )

    sizes = trades.size
    cumqty = csp.accum(sizes)

    csp.print('trades', trades)
    csp.print('cumqty', cumqty)


csp.run( my_graph, starttime = datetime( 2020, 1, 1 ))


>>> 2020-01-01 00:00:00 trades:Trade( price=100.01, size=200 )
>>> 2020-01-01 00:00:00 cumqty:200
>>> 2020-01-01 00:00:01 trades:Trade( price=100.01, size=300 )
>>> 2020-01-01 00:00:01 cumqty:500
```

`trades` is defined as a timeseries of `Trade` objects. On line 13 we access the `size` field of the `trades` timeseries, then accumulate the sizes to get `cumqty` edge.

## Available methods

- **`clear(self)`** clear all fields on the struct
- **`collectts(self, **kwargs)`**: `kwargs` expects key/values of struct fields and time series to populate the struct. This will return an Edge representing a ticking struct created from all the ticking inputs provided. Structs will only be generated from inputs that actively ticked in the given engine cycle (see `fromts` to create struct from all valid inputs)
- **`copy(self)`**: return a shallow copy of the struct
- **`copy_from(self, rhs)`**: copy data from `rhs` into this instance. `rhs` must be of the same type or a derived type. Copy will include unsetting fields unset in the `rhs`.
- **`update_from(self, rhs)`**: copy only the set fields from the `rhs` into this instance
- **`update(self, **kwargs)`**: in this instance, set the provided fields with the provided values
- **`fromts(self, trigger=None, /, **kwargs)`**: similar to `collectts` above, `fromts` will create a ticking Struct timeseries from the valid values of all the provided inputs whenever any of them tick. `trigger` is an optional position-only argument which is used as a trigger timeseries for when to convert inputs into a struct tick. By default any input tick will generate a new struct of valid inputs
- **`from_dict(self, dict)`**: convert a regular python dict to an instance of the struct
- **`metadata(self)`**: returns the struct's metadata as a dictionary of key : type pairs
- **`to_dict(self)`**: convert struct instance to a python dictionary
- **`all_fields_set(self)`**: returns `True` if all the fields on the struct are set. Note that this will not recursively check sub-struct fields

# Note on inheritance

`csp.Struct` types may inherit from each other, but **multiple inheritance is not supported**. Composition is usually a good choice in absence of multiple inheritance.
