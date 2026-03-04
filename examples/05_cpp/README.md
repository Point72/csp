# C++ Nodes and Adapters

This directory contains examples demonstrating how to extend CSP with custom
C++ components, including nodes and adapters.

## Examples

| Directory                                           | Description                                        |
| --------------------------------------------------- | -------------------------------------------------- |
| [1_cpp_node](./1_cpp_node/)                         | Basic C++ node implementation                      |
| [2_cpp_node_with_struct](./2_cpp_node_with_struct/) | C++ node that works with `csp.Struct` types        |
| [3_cpp_adapter](./3_cpp_adapter/)                   | C++ adapter and adapter manager (internal API)     |
| [4_c_api_adapter](./4_c_api_adapter/)               | C adapter using the **stable C API** (recommended) |
| [5_c_api_adapter_rust](./5_c_api_adapter_rust/)     | Rust adapter using the stable C API (scaffold)     |

## Choosing an Approach

### For Custom Nodes

Use **C++ Nodes** (`1_cpp_node`, `2_cpp_node_with_struct`) when you need:

- High-performance computation in the graph
- Access to C/C++ libraries
- Complex state management in native code

### For Custom Adapters

**Recommended: C API** (`4_c_api_adapter`, `5_c_api_adapter_rust`)

- Stable API with backward compatibility
- Works with C, C++, Rust, or any language with C FFI
- Well-defined VTable interface for callbacks
- See [C API Documentation](../../docs/wiki/c-api/README.md)

**Advanced: C++ API** (`3_cpp_adapter`)

- Full access to CSP internals
- More powerful but **API is not stable**
- Use only if you need features not exposed via C API

## See Also

- [CSP Documentation](../../docs/wiki/)
- [Writing Adapters Guide](../../docs/wiki/how-tos/Write-Adapters.md)
- [C API Reference](../../docs/wiki/c-api/README.md)
