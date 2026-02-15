# Custom C++ Node w/ Struct

This is a small example to create a custom C++ node, interfacing with `csp.Struct` instances from C++.

Compile:

```bash
python setup.py build build_ext --inplace
```

Run:

```bash
python -m mystruct
```

> [!WARNING]
> This example is for demonstration, and is a pattern CSP uses internally for fast nodes.
> It is not recommended to use as the C++ API is not stable and may change without notice. Use at your own risk.
> For adapters, we have a stable C API that is recommended to use instead. See [C API Adapter](../4_c_api_adapter/README.md) example for more details.
