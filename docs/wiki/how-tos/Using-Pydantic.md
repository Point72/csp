 Using Pydantic with CSP

CSP integrates deeply with Pydantic (v2+) to provide seamless validation, serialization, and type coercion for CSP Structs, Enums, and NumPy arrays. This integration enables powerful features like fast input validation and cross-platform data interchange.

## Overview

CSP implements the `__get_pydantic_core_schema__` protocol for its custom types, allowing Pydantic to validate and serialize them natively. When you wire nodes or graphs, CSP dynamically creates Pydantic models for input/output validation, resulting in:

- **Fast type validation** — Pydantic's optimized validator is used at graph entry points
- **Type coercion** — Automatic conversion (e.g., string → int) when validating inputs
- **JSON serialization** — Built-in support for `dump_json()` and `dump_python()`
- **Optional validation** — Pass `use_pydantic=True` to functions like `Struct.from_dict()` for explicit validation

### Enable/Disable

CSP uses Pydantic by default when available. Disable it with:
```bash
export CSP_PYDANTIC=false
```

---

## Structs

CSP `Struct` classes expose Pydantic-compatible core schemas. You can:

1. **Construct from a dict** with validation
2. **Use Pydantic TypeAdapter** for faster serialization
3. **Handle nested Structs** with automatic validation
4. **Get type coercion** (e.g., `"42"` → `42`)

### Basic Example

```python
import csp
from pydantic import TypeAdapter

class Trade(csp.Struct):
    ticker: str
    price: float
    quantity: int = 1

# Option 1: from_dict with pydantic validation
trade = Trade.from_dict({
    "ticker": "AAPL",
    "price": "123.45",      # coerced to float
    "quantity": "10"         # coerced to int
}, use_pydantic=True)

print(trade.price)      # 123.45 (float)
print(trade.quantity)   # 10 (int)

# Option 2: Use TypeAdapter for direct serialization
adapter = TypeAdapter(Trade)
json_str = adapter.dump_json(trade)
trade_from_json = adapter.validate_python({"ticker": "AAPL", "price": 123.45})
```

### Nested Structs

Pydantic validates nested structures automatically:

```python
class Order(csp.Struct):
    trade: Trade           # nested Struct
    order_id: int
    status: str = "pending"

data = {
    "trade": {"ticker": "AAPL", "price": "123.45"},  # nested dict
    "order_id": 101
}

order = Order.from_dict(data, use_pydantic=True)
assert isinstance(order.trade, Trade)
assert order.trade.price == 123.45
```

### Underscore Attributes (Private Fields)

CSP allows underscore-prefixed fields (CSP convention), but Pydantic skips them during validation:

```python
class StructWithPrivate(csp.Struct):
    public_field: int
    _internal_id: int = 0  # Pydantic ignores this

# Pydantic won't validate _internal_id, but CSP will accept it
s = StructWithPrivate.from_dict(
    {"public_field": 42, "_internal_id": 99},
    use_pydantic=True
)
assert s._internal_id == 99
```

---

## Enums

CSP Enums implement Pydantic's core schema protocol, allowing validation by name or value. Enums serialize to strings in JSON.

### Example

```python
import csp

class Status(csp.Enum):
    PENDING = csp.Enum.auto()
    ACTIVE = csp.Enum.auto()
    DONE = csp.Enum.auto()

class Task(csp.Struct):
    name: str
    status: Status = Status.PENDING

# Pydantic coerces string to enum
task = Task.from_dict({
    "name": "review",
    "status": "ACTIVE"  # coerced to Status.ACTIVE
}, use_pydantic=True)

assert task.status == Status.ACTIVE

# When converting to JSON, enum is serialized as string
from pydantic import TypeAdapter
adapter = TypeAdapter(Task)
json_str = adapter.dump_json(task)
# → '{"name":"review","status":"ACTIVE"}'
```

### Key Behaviors

- **Input validation**: Accepts enum name (string) or enum value, coerces to the enum type
- **JSON serialization**: Enums serialize to their string name
- **Dict serialization** (non-pydantic): Pydantic route preserves the Enum class; CSP's native `to_dict()` serializes to string

---

## NumPy Arrays

CSP provides Pydantic validators for NumPy arrays via `NumpyNDArray` and `Numpy1DArray` helper types. These ensure:

- Arrays validate with the correct dtype
- Arrays serialize to lists in JSON
- Type checking happens at validation time

### Example

```python
import csp
from csp.typing import NumpyNDArray, Numpy1DArray
import numpy as np
from typing import List
from pydantic import TypeAdapter

class Signal(csp.Struct):
    name: str
    data: Numpy1DArray[np.float32]
    metadata: NumpyNDArray[np.int32]  # any shape

# Pydantic validates and converts
signal = Signal.from_dict({
    "name": "signal_1",
    "data": [1.0, 2.5, 3.7],           # converted to 1D array
    "metadata": [[1, 2], [3, 4]]       # converted to 2D array
}, use_pydantic=True)

assert signal.data.dtype == np.float32
assert signal.data.shape == (3,)
assert signal.metadata.shape == (2, 2)

# Serialization to JSON converts arrays to lists
adapter = TypeAdapter(Signal)
json_str = adapter.dump_json(signal)
# → '{"name":"signal_1","data":[1.0,2.5,3.7],"metadata":[[1,2],[3,4]]}'
```

---

## Using Pydantic Models with CSP Nodes

You can pass Pydantic `BaseModel` instances to CSP nodes, and CSP will validate them via its dynamic input models:

```python
from pydantic import BaseModel
import csp

class PriceUpdate(BaseModel):
    ticker: str
    price: float
    timestamp: int

@csp.node
def process_update(update: csp.ts[PriceUpdate]) -> csp.ts[float]:
    return update.price * 1.1  # 10% markup

@csp.graph
def my_graph():
    updates = csp.const(PriceUpdate(ticker="MSFT", price=100.0, timestamp=1234567890))
    result = process_update(updates)
    csp.print("Result", result)

csp.run(my_graph)
```

---

## Validation and Type Coercion

Pydantic's validation automatically coerces compatible types:

```python
import csp

class Data(csp.Struct):
    value: int
    items: list
    count: float

# All get coerced to the correct types
d = Data.from_dict({
    "value": "42",                     # string → int
    "items": (1, 2, 3),                # tuple → list
    "count": "3.14"                    # string → float
}, use_pydantic=True)

print(d.value, d.items, d.count)  # 42 [1, 2, 3] 3.14
```

---

## Caveats and Notes

1. **Shallow copy behavior** — Pydantic performs shallow copies during validation, which may differ from CSP's in-place mutation semantics. If you rely on references to mutable objects, test carefully.

2. **Datetime serialization** — Pydantic and CSP serialize datetimes differently by default. Use Pydantic field validators or custom serializers to control the format:
   ```python
   from pydantic import field_serializer
   from datetime import datetime

   class Event(csp.Struct):
       ts: datetime
       
       @field_serializer('ts')
       def serialize_ts(self, value: datetime) -> str:
           return value.isoformat()
   ```

3. **Underscore fields** — Fields starting with `_` are not validated by Pydantic but do pass through to CSP. Serialization via Pydantic excludes them.

4. **Disabling Pydantic** — Set `CSP_PYDANTIC=false` or call `.from_dict(..., use_pydantic=False)` to use CSP's native (non-Pydantic) validation path.

5. **Dynamic model errors** — If CSP cannot create a Pydantic model for a node's inputs/outputs, it raises a `TypeError` with details. Check `csp/impl/wiring/signature.py` for implementation details.

---

## References and Implementation

**Implementation files:**
- `csp/impl/struct.py` — Struct pydantic schema generation and `from_dict()`
- `csp/impl/enum.py` — Enum pydantic core schema
- `csp/typing.py` — NumPy array validators
- `csp/impl/types/pydantic_types.py` — CSP type validators for Pydantic
- `csp/impl/wiring/signature.py` — Dynamic input/output model creation
- `csp/impl/types/pydantic_type_resolver.py` — Type variable resolution

**Test coverage:**
- `csp/tests/impl/test_struct.py` — Comprehensive Struct + Pydantic tests
- `csp/tests/test_type_checking.py` — Type coercion and validation tests (filter for `CSP_PYDANTIC`)