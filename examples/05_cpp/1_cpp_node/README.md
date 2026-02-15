# Custom C++ Node

This is a small example to create a custom C++ node.

Compile:

```bash
python setup.py build build_ext --inplace
```

Run:

```bash
python -m piglatin
```

Output:

```raw
2020-01-01 00:00:00.500000 input:pig
2020-01-01 00:00:00.500000 output:IGPAY
2020-01-01 00:00:01.500000 input:latin
2020-01-01 00:00:01.500000 output:ATINLAY
2020-01-01 00:00:05 input:fun
2020-01-01 00:00:05 output:UNFAY
```

> [!WARNING]
> This example is for demonstration, and is a pattern CSP uses internally for fast nodes.
> It is not recommended to use as the C++ API is not stable and may change without notice. Use at your own risk.
> For adapters, we have a stable C API that is recommended to use instead. See [C API Adapter](../4_c_api_adapter/README.md) example for more details.
