To get various data sources into and out of the graph, various Input and Output Adapters are available, such as CSV, Parquet, and database adapters (amongst others).
Users can also write their own input and output adapters, as explained below.

There are two types of Input Adapters: **Historical** (aka Simulated) adapters and **Realtime** Adapters.

Historical adapters are used to feed in historical timeseries data into the graph from some data source which has timeseries data.
Realtime Adapters are used to feed in live event based data in realtime, generally events created from external sources on separate threads.

There is not distinction of Historical vs Realtime output adapters since outputs need not care if the generated timeseries data which are wired into them are generated from realtime or historical inputs.

In CSP terminology, a single adapter corresponds to a single timeseries edge in the graph.
There are common cases where a single data source may be used to provide data to multiple adapter (timeseries) instances, for example a single CSV file with price data for many stocks can be read once but used to provide data to many individual, one per stock.
In such cases an AdapterManager is used to coordinate management of the single source (CSV file, database, Kafka connection, etc) and provided data to individual adapters.

Note that adapters can be quickly written and prototyped in Python, and if needed can be moved to a C++ implementation for more efficiency.
For maximum portability and ABI stability, adapters can also be written in C (or any language with C FFI such as Rust or Go) using the [C API](../api-references/C-APIs.md).
