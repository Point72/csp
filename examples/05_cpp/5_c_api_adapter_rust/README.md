# Custom Rust Adapter via C API

This example demonstrates how to create CSP adapters in **Rust** using the stable C API
and **PyO3** for Python bindings. It is built using **hatch-rs** for seamless integration
with Python's build system.

## Overview

The example implements:

- **RustAdapterManager** - Manages adapter lifecycle
- **RustInputAdapter** - Push input adapter with lifecycle callbacks
- **RustOutputAdapter** - Output adapter that logs received values

## Project Structure

```
5_c_api_adapter_rust/
├── Cargo.toml           # Rust package manifest
├── pyproject.toml       # Python package config (uses hatch-rs)
├── build.rs             # Build script for linking
├── src/
│   ├── lib.rs           # PyO3 module with Python bindings
│   ├── bindings.rs      # FFI bindings for C API
│   ├── adapter_manager.rs   # Adapter manager implementation
│   ├── input_adapter.rs     # Push input adapter
│   └── output_adapter.rs    # Output adapter
└── exampleadapter/
    ├── __init__.py      # Package init
    └── __main__.py      # Python wrapper and demo
```

## Building

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.9+
- CSP installed (`pip install csp`)
- hatch-rs installed (`pip install hatch-rs`)

### Build Commands

```bash
cd examples/05_cpp/5_c_api_adapter_rust

# Build the wheel (recommended)
hatch-build --hooks-only -t wheel

# Run tests
cargo test
```

## Current Status

This example demonstrates the **structure and patterns** for implementing CSP adapters in
Rust. The module can be imported and the VTable capsules are properly created:

```python
>>> import exampleadapter
>>> exampleadapter._example_input_adapter(100)
<capsule object "csp.c.InputAdapterCapsule" at 0x...>
```

### Integration with CSP

CSP exports C API symbols from its shared libraries, making them available at runtime.
To fully integrate Rust adapters with CSP:

1. **Create VTable capsules** in Rust (already done in this example)
1. **Use Python bridge functions** to connect capsules to CSP's wiring layer

The Python wiring follows the same pattern as the C example:

```python
from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring import input_adapter_def

from . import _exampleadapterimpl  # Rust extension

def _create_rust_input_adapter(mgr, engine, pytype, push_mode, scalars):
    interval_ms = scalars[1] if len(scalars) > 1 else 100

    # Create capsule from Rust
    capsule = _exampleadapterimpl._example_input_adapter(interval_ms)

    # Pass to CSP bridge
    return _cspimpl._c_api_push_input_adapter(
        mgr, engine, pytype, push_mode, (capsule, None)
    )

rust_input = input_adapter_def(
    "rust_input",
    _create_rust_input_adapter,
    ts["T"],
    typ="T",
    interval_ms=int,
)
```

### Limitations

The `start` callback in push input adapters receives an adapter handle, which is needed
to push data into the graph via functions like `ccsp_push_input_adapter_push_int64`.
To call these functions from Rust:

- Link against CSP's libraries at build time, or
- Use `dlopen`/`dlsym` to load symbols at runtime
- Or build as part of CSP's cmake build system

See the [C API Adapter example](../4_c_api_adapter/) for a complete working implementation.

## Key Concepts

### PyO3 Integration

The `lib.rs` module exposes Rust functions to Python using PyO3:

```rust
#[pyfunction]
fn _example_input_adapter(py: Python<'_>, interval_ms: i32) -> PyResult<PyObject> {
    let adapter = Box::new(RustInputAdapter::new(interval_ms as u64));
    let user_data = Box::into_raw(adapter) as *mut c_void;

    let vtable = CCspPushInputAdapterVTable {
        user_data,
        start: Some(rust_input_adapter_start),
        stop: Some(rust_input_adapter_stop),
        destroy: Some(rust_input_adapter_destroy),
    };

    create_input_adapter_capsule(py, vtable)
}
```

### C API VTables

The adapters implement the CSP C ABI by creating VTable structures with callbacks:

```rust
#[repr(C)]
pub struct CCspPushInputAdapterVTable {
    pub user_data: *mut c_void,
    pub start: Option<unsafe extern "C" fn(...)>,
    pub stop: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}
```

### Python Capsules

VTables are passed to Python as capsules, which CSP's adapter system understands.
Static strings are used for capsule names to ensure they remain valid:

```rust
static INPUT_ADAPTER_CAPSULE_NAME: &[u8] = b"csp.c.InputAdapterCapsule\0";

pub fn create_input_adapter_capsule(
    py: Python<'_>,
    vtable: CCspPushInputAdapterVTable,
) -> PyResult<PyObject> {
    let vtable_ptr = Box::into_raw(Box::new(vtable)) as *mut c_void;

    unsafe {
        let capsule = ffi::PyCapsule_New(
            vtable_ptr,
            INPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(destructor),
        );
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}
```

### Memory Management

Rust adapters use `Box` for heap allocation with proper cleanup:

```rust
// Create adapter (transfers ownership to C via capsule)
let adapter = Box::new(RustInputAdapter::new(interval_ms));
let user_data = Box::into_raw(adapter) as *mut c_void;

// Destroy callback (reclaims ownership and drops)
pub unsafe extern "C" fn rust_input_adapter_destroy(user_data: *mut c_void) {
    if !user_data.is_null() {
        let _ = Box::from_raw(user_data as *mut RustInputAdapter);
    }
}
```

## hatch-rs Configuration

The `pyproject.toml` configures hatch-rs to build the Rust extension:

```toml
[build-system]
requires = ["hatchling", "hatch-rs", "csp"]
build-backend = "hatchling.build"

[tool.hatch.build.hooks.hatch-rs]
module = "exampleadapter"
```

## See Also

- [C API Adapter](../4_c_api_adapter/README.md) - C implementation example
- [CSP Documentation](https://github.com/Point72/csp)
- [PyO3 Documentation](https://pyo3.rs/)
- [hatch-rs](https://github.com/python-project-templates/hatch-rs)
