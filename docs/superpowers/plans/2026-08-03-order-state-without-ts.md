# Native Order State Without Time-Series Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic, graph-local way to inject one shared native C++ state object into C++ `cppimpl` nodes and Python nodes, while retaining ordinary CSP edges as the sole source of scheduling, causality, and triggering.

**Architecture:** Introduce a polymorphic `NativeResource` owned by `std::shared_ptr`, carry resources in a dedicated `CppNode::NodeDef::resources` map, and expose the same allocation to Python through a shared-owning C-extension wrapper. C++ nodes copy a typed `shared_ptr` during construction and dereference it directly thereafter; Python only participates at graph construction and when a Python node explicitly calls the wrapper. State is never placed on a CSP time series.

**Tech Stack:** C++17, CPython C API, CSP C++ engine and `cppimpl` node macros, CMake, GoogleTest, pytest, Markdown.

## Global Constraints

- The order store and its mutation/read methods are implemented in C++, not Python.
- A C++ node must not hold a `PyObject`, `DialectGenericType`, capsule, integerized pointer, or registry key to reach the store at runtime.
- The resource reference provides access and lifetime only. It must not create an implicit graph dependency, trigger a node, or replace a CSP event edge.
- Exactly one engine-thread node mutates a given order store. Readers may perform synchronous lookups while their node executes.
- A resource is fresh per graph/engine run. No process-global store or registry is introduced.
- Access from threads outside the CSP engine is unsupported unless the application resource supplies its own synchronization.
- This proposal does not add arbitrary C++ objects to CSP time-series types, `csp.Struct`, `Dictionary`, history, serialization, replay, or checkpointing.
- Existing scalar conversion remains unchanged for every non-resource scalar, including `DialectGenericType` fallback.
- Missing resources and resource type mismatches fail during native node construction, before the engine starts.

---

## Proposal

### Problem statement

A trading graph needs a Nautilus-style current-order view:

- one native node applies order events to one mutable `OrderStore`;
- C++ strategy, execution, and risk nodes synchronously call `orders->get(order_id)`;
- Python nodes can inspect the same current state through a Python extension object;
- consumers that react to a committed order event must observe the state after that event has been applied; and
- the full map is not copied, serialized, or emitted on every update.

The current `cppimpl` construction path cannot express this cleanly. `CppNode::NodeDef` contains only input/output definitions and a `Dictionary` of scalars. `PyCppNode.cpp` converts the Python scalar dictionary wholesale. Values outside the closed `Dictionary::Value` variant fall back to `DialectGenericType`, so a native node must currently retain or unwrap a Python object.

That fallback is useful for Python-oriented generic scalars, but it is the wrong ownership and type boundary for native shared state.

### Proposed ownership model

Add a minimal engine-level marker and pointer alias:

```cpp
class NativeResource
{
public:
    virtual ~NativeResource() = default;
};

using NativeResourcePtr = std::shared_ptr<NativeResource>;
```

Application state derives from the marker:

```cpp
class OrderStore final : public csp::NativeResource
{
public:
    const OrderState * get( OrderId id ) const;
    void apply( const OrderEvent & event );

private:
    std::unordered_map<OrderId, OrderState> m_orders;
};
```

Each graph run constructs one `std::shared_ptr<OrderStore>`. The graph passes that capability to every node that needs it. `CppNode::NodeDef` keeps native resources separate from ordinary scalar values:

```cpp
struct NodeDef
{
    InOutDefs inputs;
    InOutDefs outputs;
    Dictionary scalars;
    std::unordered_map<std::string, NativeResourcePtr> resources;
};
```

Keeping resources out of `Dictionary::Value` is intentional. A resource has identity and graph-run lifetime; it does not have useful value equality, hashing, recursive conversion, or serialization semantics.

### Direct C++ access

`CppNode` gains a typed constructor-time lookup and wrapper:

```cpp
template<typename T>
std::shared_ptr<T> resourceValue( const char * name ) const;

template<typename T>
class Resource
{
public:
    Resource( const char * name, const CppNode & node ) : m_value( node.resourceValue<T>( name ) ) {}

    T * operator->() const { return m_value.get(); }
    T & operator*() const { return *m_value; }
    const std::shared_ptr<T> & value() const { return m_value; }

private:
    std::shared_ptr<T> m_value;
};

#define RESOURCE_INPUT( Type, Name ) csp::CppNode::Resource<Type> Name{#Name,*this};
```

The writer is then ordinary C++:

```cpp
DECLARE_CPPNODE( apply_order_event )
{
    INIT_CPPNODE( apply_order_event ) {}

    TS_INPUT( OrderEvent, event );
    RESOURCE_INPUT( OrderStore, orders );
    TS_OUTPUT( OrderEvent );

    INVOKE()
    {
        if( csp.ticked( event ) )
        {
            orders->apply( event );
            RETURN( event ); // committed notification, emitted after mutation
        }
    }
};
```

A read-only C++ consumer can request `const OrderStore`:

```cpp
RESOURCE_INPUT( const OrderStore, orders );
```

After construction, these nodes only hold and dereference a native `std::shared_ptr`. They include no Python headers and perform no Python operations in `START`, `INVOKE`, or `STOP`.

A fully native graph builder can populate `NodeDef::resources` directly. When a graph is hosted by Python, the binding layer necessarily participates once while constructing the node, but the C++ node itself still receives the native pointer directly.

### Python access to the same allocation

Add an internal C-extension base type, `PyNativeResource`, which owns a `NativeResourcePtr`. Application extensions derive a typed Python view from it. For the order store, the extension exposes read methods such as `get`, `contains`, and `open_orders`; it does not reimplement the map in Python.

Conceptually, the factory is:

```cpp
PyObject * createOrderStore()
{
    return PyNativeResource::create(
        &PyOrderStore::PyType,
        std::make_shared<OrderStore>() );
}
```

and a Python node receives that wrapper as an ordinary scalar:

```python
@csp.node
def inspect_committed_order(
    committed: ts[OrderEvent],
    orders: PyOrderStore,
) -> ts[OrderState]:
    if csp.ticked(committed):
        return orders.get(committed.order_id)
```

The Python wrapper and every C++ node own `shared_ptr`s to the same allocation. Destroying the graph-side Python variable does not invalidate a constructed C++ node, and destroying the engine does not invalidate a wrapper that the application still holds. The `OrderStore` is destroyed when the last owner releases it.

`PyCppNode.cpp` recognizes `PyNativeResource` values before generic scalar conversion:

```cpp
while( PyDict_Next( pyscalars, &pos, &key, &value ) )
{
    auto name = fromPython<std::string>( key );
    if( PyObject_TypeCheck( value, &PyNativeResource::PyType ) )
        nodedef.resources.emplace( name, PyNativeResource::resource( value ) );
    else
        nodedef.scalars.insert( name, fromPython<Dictionary::Value>( value ) );
}
```

This is a graph-construction bridge, not a time-series or tick-time bridge.

### Scheduling and ordering contract

The resource and event edge have different jobs:

| Mechanism | Provides | Does not provide |
|---|---|---|
| `shared_ptr<OrderStore>` | identity, lifetime, synchronous lookup | rank, triggering, event association, history |
| CSP edge | dependency rank, triggering, event identity, cycle boundaries | shared mutable object ownership |

The writer must mutate first and emit its committed output second. Any downstream consumer that needs the newly applied state consumes that output:

```text
OrderEvent --> apply_order_event(writer) -- committed --> risk(reader)
                         |                                 |
                         +---------- shared store --------+
```

When `risk` executes because `committed` ticked, the writer invocation has completed and `risk` sees the applied state. The edge may be active when the committed event should trigger the consumer, or made passive when another input triggers the consumer but same-cycle writer-before-reader rank is still required.

Use these rules:

| Consumer requirement | Required wiring |
|---|---|
| Process this commit and read its applied state | Active writer-to-reader committed edge |
| Trigger elsewhere, but read after a same-cycle write | Passive writer-to-reader dependency edge |
| Read whatever state was last committed whenever the node happens to run | Resource only; ordering is deliberately unspecified relative to unrelated writers |
| Match an event to historical state after later writes occurred | Immutable snapshot or explicit versioned history; outside this proposal |

A transitive path such as `writer -> normalizer -> reader` is sufficient. A common ancestor with sibling paths, `source -> writer` and `source -> reader`, is not sufficient because nodes at the same rank have undefined relative order.

### Strategy-to-execution feedback phase

For the trading topology discussed here:

```text
market/other events --> strategy(reader) --> order execution(writer)
                            ^                        |
                            +------ feedback --------+

external execution events -----------------------> writer
```

the forward strategy-to-execution edge deliberately places strategy before the writer in the current cycle. Therefore:

1. On a market-triggered strategy invocation, strategy reads the state committed by prior writer invocations.
2. Strategy emits a command.
3. Order execution applies the command or external execution event to the store.
4. Order execution emits a committed event after `apply` returns.
5. `csp.feedback` delivers that event to strategy on the next engine cycle, possibly at the same engine timestamp.
6. On that feedback-triggered invocation, strategy reads the newly committed state.

This is a two-phase loop, not a same-cycle cyclic dependency. Feedback does solve the post-commit notification path. It cannot make the upstream strategy see a downstream mutation earlier in the cycle that caused that mutation; asking for that would create a causal cycle.

If independent external inputs tick strategy and execution at the same engine time, the shared reference does not invent a total ordering between those inputs. The explicit graph phase still governs: a strategy invocation before the writer sees prior committed state; an invocation caused by the writer's committed feedback sees the new state. If the business rule requires a different precedence, normalize those inputs into an explicit sequenced event stream or change the graph topology.

### Multiple events and collapsing

The store is stable for the duration of a node invocation because CSP executes nodes serially. Exact event/state matching still requires the writer's event stream to preserve the intended granularity:

- If a writer invocation applies one event and emits one committed tick, the downstream reader observes that event's state.
- If one invocation intentionally applies a batch and emits once, readers observe the final state of the batch.
- If several independent inputs can tick in one invocation, define their application order and emit a batch commit, or normalize them upstream into one ordered stream.
- If multiple same-timestamp events must remain distinct, use a non-collapsing source/path and verify one writer invocation and one committed notification per event.

No revision counter is required for synchronous downstream or feedback-phase consumers under this contract. Revisions become necessary only for asynchronous readers, arbitrary delayed work, or requests for exact historical state after subsequent commits.

### Threading and mutation policy

The default contract is single-writer, engine-thread access:

- only the designated writer calls mutating methods;
- graph nodes perform reads synchronously while executing;
- no background thread mutates the resource; and
- Python code outside the running graph does not inspect the resource concurrently.

Under those conditions no lock is necessary. A resource intentionally shared with external threads must implement its own synchronization and document its snapshot semantics; that is an application concern, not behavior supplied by `NativeResource`.

### Alternatives not selected

- **`DialectGenericType` plus a one-time unwrap:** workable without CSP changes, but native node construction remains Python-aware and each application invents its own unwrap convention.
- **A `csp.Struct` containing a pointer/reference:** structs are values with type, copying, conversion, and potential serialization semantics. A process-local native capability does not belong in them.
- **An arbitrary C++ object on a time series:** this adds type-system, cache, history, conversion, and Python-dialect problems while still not making mutation ordering implicit.
- **A raw pointer, capsule, or integer address:** cannot express ownership and can become dangling or type-confused.
- **A process-global registry keyed by integer:** removes Python from node construction but introduces hidden global state, cleanup hazards, ID reuse, and cross-engine contamination.
- **Copying the full order map on every event:** gives immutable event-time snapshots but defeats the desired synchronous current-state cache. It remains the correct option when consumers truly require historical snapshots.

### Acceptance criteria

- A native `OrderStore` allocation is created once and shared with both C++ and Python nodes.
- A C++ node declares `RESOURCE_INPUT(OrderStore, orders)` and never touches Python to use it.
- A Python node calls an extension method on the same allocation.
- Non-resource scalar behavior is unchanged.
- Missing and incorrectly typed resources fail with the node name, resource name, expected type, and actual type in the exception.
- A committed writer-to-reader edge proves apply-before-read behavior.
- The strategy-to-writer feedback test proves prior-state visibility in the forward phase and new-state visibility in the feedback phase.
- Same-timestamp non-collapsing events remain paired with their committed state.
- Resource lifetime is correct when either the Python wrapper or the graph releases its owner first.
- Documentation states that resources provide no scheduling, history, thread safety, serialization, or cross-process transport.

---

## Implementation Tasks

### Task 1: Add the engine-level native resource carrier

**Files:**

- Create: `cpp/csp/engine/NativeResource.h`
- Modify: `cpp/csp/engine/CppNode.h`
- Modify: `cpp/csp/engine/CMakeLists.txt`
- Create: `cpp/tests/engine/test_native_resource.cpp`
- Modify: `cpp/tests/engine/CMakeLists.txt`

- [ ] **Step 1: Write the failing C++ tests**

  Add a `TestStore : NativeResource`, an unrelated `OtherResource`, and a minimal `CppNode` test subclass. Cover:

  - exact typed lookup returns the original shared allocation;
  - `Resource<const TestStore>` provides read-only access;
  - a missing name throws `ValueError` containing the node and resource names;
  - an `OtherResource` under the requested name throws `TypeError` containing expected and actual C++ type names; and
  - the node wrapper retains ownership after the original `NodeDef` resource map and caller `shared_ptr` are cleared.

- [ ] **Step 2: Register the test executable**

  Add `test_native_resource` to `cpp/tests/engine/CMakeLists.txt`, link it to `csp_engine`, `GTest::gtest`, and `GTest::gtest_main`, and add it to the existing test install target list.

- [ ] **Step 3: Confirm the test does not compile yet**

  Run: `make build`

  Expected: compilation fails because `csp/engine/NativeResource.h`, `NodeDef::resources`, and `CppNode::resourceValue` do not exist.

- [ ] **Step 4: Implement `NativeResource` and typed lookup**

  In `NativeResource.h`, define the virtual base, `NativeResourcePtr`, and `NativeResourceMap`. In `CppNode.h`:

  - add `NativeResourceMap resources` to `NodeDef`;
  - implement `resourceValue<T>()` with `std::remove_const_t<T>`, `std::is_base_of_v`, and `std::dynamic_pointer_cast`;
  - implement `Resource<T>` so it copies the `shared_ptr` while `m_nodedef` is valid; and
  - add `RESOURCE_INPUT` and `RESOURCE_INPUT_RENAMED` macros mirroring scalar input naming.

  Do not add resource alternatives to `Dictionary::Value`.

- [ ] **Step 5: Build and run the focused C++ test**

  Run:

  ```bash
  make build
  ./csp/tests/bin/test_native_resource --gtest_color=yes
  ```

  Expected: the build succeeds and all `NativeResource` tests pass.

- [ ] **Step 6: Commit the core API**

  ```bash
  git add cpp/csp/engine/NativeResource.h cpp/csp/engine/CppNode.h cpp/csp/engine/CMakeLists.txt \
    cpp/tests/engine/test_native_resource.cpp cpp/tests/engine/CMakeLists.txt
  git commit -s -m "engine: add typed native node resources"
  ```

### Task 2: Add the CPython wrapper for native resources

**Files:**

- Create: `cpp/csp/python/PyNativeResource.h`
- Create: `cpp/csp/python/PyNativeResource.cpp`
- Modify: `cpp/csp/python/CMakeLists.txt`
- Modify: `cpp/csp/python/csptestlibimpl.cpp`
- Create: `csp/tests/impl/test_native_resource.py`

- [ ] **Step 1: Write the failing wrapper and lifetime tests**

  In `csptestlibimpl.cpp`, declare `TestStore : NativeResource` and an application-style `PyTestStore` type derived from `PyNativeResource`. Add a `_test_native_resource()` factory, read-only Python methods `get()` and `allocation_id()`, and a module-level live-allocation counter for lifetime assertions.

  In `test_native_resource.py`, assert that:

  - the factory returns a `PyTestStore` whose initial value is readable;
  - its Python methods dispatch to the C++ `TestStore`; and
  - repeated wrapper references report one allocation ID;
  - the native live-allocation counter returns to its baseline after the final owner disappears.

- [ ] **Step 2: Confirm the test fails**

  Run: `python -m pytest -v csp/tests/impl/test_native_resource.py -k wrapper`

  Expected: import or compilation fails because `PyNativeResource` has not been implemented.

- [ ] **Step 3: Implement the internal base extension type**

  Implement `PyNativeResource` as a `CSPIMPL_EXPORT` C-extension base with:

  - a heap-allocated `NativeResourcePtr` field initialized only by `create()`;
  - `tp_new = nullptr` so Python cannot instantiate an empty resource;
  - `tp_flags` including `Py_TPFLAGS_BASETYPE` so typed extension views can derive from it;
  - deallocation that deletes the `shared_ptr` holder and then frees the Python object;
  - `create(PyTypeObject *, NativeResourcePtr)` for typed wrappers;
  - `isNativeResource(PyObject *)` and `resource(PyObject *)`; and
  - `resourceAs<T>(PyObject *)` with the same checked dynamic cast used by `CppNode`.

  Register the base in `_cspimpl` as the internal name `_NativeResource`. Add the source and public header to `cpp/csp/python/CMakeLists.txt` so external C++ extension modules can derive typed views.

- [ ] **Step 4: Build and run the wrapper tests**

  Run:

  ```bash
  make build
  python -m pytest -v csp/tests/impl/test_native_resource.py -k wrapper
  ```

  Expected: all selected tests pass without exposing a Python-side state implementation.

- [ ] **Step 5: Commit the Python view boundary**

  ```bash
  git add cpp/csp/python/PyNativeResource.h cpp/csp/python/PyNativeResource.cpp \
    cpp/csp/python/CMakeLists.txt cpp/csp/python/csptestlibimpl.cpp \
    csp/tests/impl/test_native_resource.py
  git commit -s -m "python: wrap native node resources"
  ```

### Task 3: Route resource scalars directly into C++ nodes

**Files:**

- Modify: `cpp/csp/python/PyCppNode.cpp`
- Modify: `cpp/csp/python/csptestlibimpl.cpp`
- Modify: `csp/tests/impl/test_native_resource.py`

- [ ] **Step 1: Add failing mixed-language integration tests**

  Add test `cppimpl` nodes to `csptestlibimpl.cpp`:

  - `set_native_resource`: takes `ts[int] value`, declares `RESOURCE_INPUT(TestStore, store)`, writes the value, and emits a committed `ts[int]` after the write;
  - `read_native_resource`: takes the committed tick, declares `RESOURCE_INPUT(const TestStore, store)`, and emits the synchronously read value; and
  - `expect_other_resource`: requests the wrong resource type for error-path coverage.

  In Python, define matching `@csp.node(cppimpl=...)` declarations. Assert that:

  - the C++ writer and C++ reader observe values `1, 2, 3` through one native allocation;
  - a Python reader triggered by the writer's committed edge observes `1, 2, 3` through `PyTestStore.get()`;
  - a missing resource and a wrong resource type fail while constructing the native node; and
  - an ordinary arbitrary Python scalar still reaches an existing `DialectGenericType` test node unchanged.

- [ ] **Step 2: Confirm the resource is still treated as `DialectGenericType`**

  Run: `python -m pytest -v csp/tests/impl/test_native_resource.py -k cppimpl`

  Expected: the native resource tests fail because `PyCppNode.cpp` has not populated `NodeDef::resources`.

- [ ] **Step 3: Split resource extraction from ordinary scalar conversion**

  Replace the wholesale `fromPython<Dictionary>(pyscalars)` call with one `PyDict_Next` loop. For each string key:

  - if the value is a `PyNativeResource`, insert its `NativeResourcePtr` into `nodedef.resources`;
  - otherwise convert exactly once with `fromPython<Dictionary::Value>` and insert into `nodedef.scalars`; and
  - preserve the current exception behavior for non-string keys and failed scalar conversions.

  Do not add Python conversion logic to `CppNode.h` or `NativeResource.h`.

- [ ] **Step 4: Build and run the integration tests**

  Run:

  ```bash
  make build
  python -m pytest -v csp/tests/impl/test_native_resource.py -k "cppimpl or dialect_generic"
  ```

  Expected: C++/C++, C++/Python, error-path, and scalar-regression tests pass.

- [ ] **Step 5: Commit direct native injection**

  ```bash
  git add cpp/csp/python/PyCppNode.cpp cpp/csp/python/csptestlibimpl.cpp \
    csp/tests/impl/test_native_resource.py
  git commit -s -m "cppimpl: inject native resources directly"
  ```

### Task 4: Lock down CSP ordering and feedback behavior

**Files:**

- Modify: `cpp/csp/python/csptestlibimpl.cpp`
- Modify: `csp/tests/impl/test_native_resource.py`

- [ ] **Step 1: Add apply-before-read tests**

  Construct a graph where a C++ writer applies one integer update, emits that integer as its committed output, and both a C++ reader and Python reader consume the committed edge. Feed three events at the same timestamp through a non-collapsing path.

  Record `(committed_event, synchronous_store_value)` at each reader and assert exactly:

  ```python
  [(1, 1), (2, 2), (3, 3)]
  ```

  This test must fail if the writer emits before mutation, if events collapse, or if either reader reaches a different allocation.

- [ ] **Step 2: Add the strategy-to-writer feedback test**

  Build this graph:

  - a Python test strategy reads `PyTestStore` and emits an increment command only on its market input;
  - the C++ writer consumes the command, updates the resource, and emits the committed value;
  - a `csp.feedback(int)` binds the committed output back to the strategy; and
  - the strategy records store values separately for market-triggered and feedback-triggered invocations.

  For one market event starting from zero, assert:

  ```python
  [("market", 0), ("feedback", 1)]
  ```

  For three market events at distinct timestamps, assert the alternating phase sequence:

  ```python
  [
      ("market", 0), ("feedback", 1),
      ("market", 1), ("feedback", 2),
      ("market", 2), ("feedback", 3),
  ]
  ```

  Keep same-timestamp non-collapsing coverage in Step 1. Do not require feedback and the next independent source event at one timestamp to arrive in separate strategy invocations; that would test scheduler tie-breaking rather than the feedback phase contract.

- [ ] **Step 3: Add a passive dependency test**

  Create a consumer triggered by a separate clock input, pass the writer's committed output as an input, call `csp.make_passive` on it during start, and assert that when writer and clock participate in the same cycle the consumer reads the applied value. Also assert that a committed tick alone does not invoke the passive consumer.

- [ ] **Step 4: Run the focused ordering tests**

  Run: `python -m pytest -v csp/tests/impl/test_native_resource.py -k "ordering or feedback or passive"`

  Expected: all selected tests pass and no revision/version field is used.

- [ ] **Step 5: Commit the scheduling contract tests**

  ```bash
  git add cpp/csp/python/csptestlibimpl.cpp csp/tests/impl/test_native_resource.py
  git commit -s -m "test: cover native resource ordering"
  ```

### Task 5: Document native resources as an advanced concept

**Files:**

- Create: `docs/wiki/concepts/Native-Resources.md`
- Modify: `docs/wiki/_Sidebar.md`

- [ ] **Step 1: Write the public documentation**

  Adapt the proposal section of this plan into user documentation. Include:

  - the resource-versus-edge responsibility table;
  - direct C++ and Python-view examples;
  - active, passive, and no-edge selection rules;
  - the strategy-to-execution feedback phase;
  - batching and non-collapsing requirements;
  - the single-writer/engine-thread contract;
  - lifetime guarantees; and
  - explicit non-guarantees for history, serialization, cross-process use, and external-thread safety.

- [ ] **Step 2: Link the concept from the sidebar**

  Add `Native Resources` under `Concepts` in `docs/wiki/_Sidebar.md`.

- [ ] **Step 3: Validate the documentation**

  Run:

  ```bash
  python -m mdformat --check docs/wiki/concepts/Native-Resources.md docs/wiki/_Sidebar.md
  python -m codespell_lib docs/wiki/concepts/Native-Resources.md docs/wiki/_Sidebar.md
  ```

  Expected: both commands exit zero.

- [ ] **Step 4: Commit the concept documentation**

  ```bash
  git add docs/wiki/concepts/Native-Resources.md docs/wiki/_Sidebar.md
  git commit -s -m "docs: explain native shared resources"
  ```

### Task 6: Run full verification and review the public boundary

**Files:**

- Review all files changed in Tasks 1-5.

- [ ] **Step 1: Run formatting and static checks**

  Run:

  ```bash
  git diff --check main...HEAD
  make lint
  ```

  Expected: both commands exit zero.

- [ ] **Step 2: Run native and focused Python tests**

  Run:

  ```bash
  ./csp/tests/bin/test_native_resource --gtest_color=yes
  python -m pytest -v csp/tests/impl/test_native_resource.py
  ```

  Expected: all focused tests pass.

- [ ] **Step 3: Run the full repository test suites**

  Run:

  ```bash
  make test-cpp
  make test-py
  ```

  Expected: all C++ and Python tests pass.

- [ ] **Step 4: Audit the boundary**

  Run:

  ```bash
  rg -n "DialectGenericType|PyObject|Python.h" cpp/csp/engine/NativeResource.h cpp/csp/engine/CppNode.h
  rg -n "NativeResource" cpp/csp/engine cpp/csp/python cpp/csp/python/csptestlibimpl.cpp \
    csp/tests/impl/test_native_resource.py docs/wiki/concepts/Native-Resources.md
  ```

  Expected: the first command finds no Python dependency in the new engine resource API; the second lists only the planned engine, binding, fixture, test, and documentation uses.

- [ ] **Step 5: Review ownership and ordering invariants**

  Confirm from code and tests that:

  - each wrapper copies a `shared_ptr` during node construction;
  - no node retains `NodeDef`, `PyObject`, or a borrowed pointer;
  - only the writer fixture mutates the store;
  - committed output occurs after mutation;
  - feedback is described and tested as a next-cycle phase; and
  - no claim implies that the resource itself creates rank or historical consistency.

- [ ] **Step 6: Record final verification in the pull request**

  Add the exact commands and pass counts from Steps 1-3 to the pull-request description. If any platform-specific suite cannot run, state the missing prerequisite and retain the focused cross-language tests as mandatory evidence.
