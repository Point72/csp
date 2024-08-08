## Table of Contents

- [Table of Contents](#table-of-contents)
- [Kafka](#kafka)
  - [API](#api)
  - [MessageMapper](#messagemapper)
  - [Subscribing and Publishing](#subscribing-and-publishing)
  - [Known Issues](#known-issues)
- [Parquet](#parquet)
  - [ParquetReader](#parquetreader)
    - [API](#api-1)
    - [Subscription](#subscription)
  - [ParquetWriter](#parquetwriter)
    - [Construction](#construction)
    - [Publishing](#publishing)
- [DBReader](#dbreader)
  - [TimeAccessor](#timeaccessor)

## Kafka

The Kafka adapter is a user adapter to stream data from a Kafka bus as a reactive time series. It leverages the [librdkafka](https://github.com/confluentinc/librdkafka) C/C++ library internally.

The `KafkaAdapterManager` instance represents a single connection to a broker.
A single connection can subscribe and/or publish to multiple topics.

### API

```python
KafkaAdapterManager(
    broker,
    start_offset: typing.Union[KafkaStartOffset,timedelta,datetime] = None,
    group_id: str = None,
    group_id_prefix: str = '',
    max_threads=100,
    max_queue_size=1000000,
    auth=False,
    security_protocol='SASL_SSL',
    sasl_kerberos_keytab='',
    sasl_kerberos_principal='',
    ssl_ca_location='',
    sasl_kerberos_service_name='kafka',
    rd_kafka_conf_options=None,
    debug: bool = False,
    poll_timeout: timedelta = timedelta(seconds=1)
):
```

- **`broker`**: name of the Kafka broker, such as `protocol://host:port`

- **`start_offset`**: signify where to start the stream playback from (defaults to `KafkaStartOffset.LATEST`).
  Can be one of the`KafkaStartOffset` enum types or:

  - `datetime`: to replay from the given absolute time
  - `timedelta`: this will be taken as an absolute offset from starttime to playback from

- **`group_id`**: if set, this adapter will behave as a consume-once consumer.
  `start_offset` may not be set in this case since adapter will always replay from the last consumed offset.

- **\`group_id_prefix**: when not passing an explicit group_id, a prefix can be supplied that will be use to prefix the UUID generated for the group_id

- **`max_threads`**: maximum number of threads to create for consumers.
  The topics are round-robin'd onto threads to balance the load.
  The adapter won't create more threads than topics.

- **`max_queue_size`**: maximum size of the (internal to Kafka) message queue.
  If the queue is full, messages can be dropped, so the default is very large.

### MessageMapper

In order to publish or subscribe, you need to define a MsgMapper.
These are the supported message types:

- **`JSONTextMessageMapper(datetime_type = DateTimeType.UNKNOWN)`**
- **`ProtoMessageMapper(datetime_type = DateTimeType.UNKNOWN)`**

You should choose the `DateTimeType` based on how you want (when publishing) or expect (when subscribing) your datetimes to be represented on the wire.
The supported options are:

- `UINT64_NANOS`
- `UINT64_MICROS`
- `UINT64_MILLIS`
- `UINT64_SECONDS`

The enum is defined in [csp/adapters/utils.py](https://github.com/Point72/csp/blob/main/csp/adapters/utils.py#L5).

Note the `JSONTextMessageMapper` currently does not have support for lists.
To subscribe to json data with lists, simply subscribe using the `RawTextMessageMapper` and process the text into json (e.g. via json.loads).

### Subscribing and Publishing

Once you have an `KafkaAdapterManager` object and a `MsgMapper` object, you can subscribe to topics using the following method:

```python
KafkaAdapterManager.subscribe(
  ts_type: type,
  msg_mapper: MsgMapper,
  topic: str,
  key=None,
  field_map: typing.Union[dict,str] = None,
  meta_field_map: dict = None,
  push_mode: csp.PushMode = csp.PushMode.LAST_VALUE,
  adjust_out_of_order_time: bool = False
):
```

- **`ts_type`**: the timeseries type you want to get the data on. This can be a `csp.Struct` or basic timeseries type
- **`msg_mapper`**: the `MsgMapper` object discussed above
- **`topic`**: the topic to subscribe to
- **`key`**: The key to subscribe to. If `None`, then this will subscribe to all messages on the topic. Note that in this "wildcard" mode, all messages will tick as "live" as replay in engine time cannot be supported
- **`field_map`**: dictionary of `{message_field: struct_field}` to define how the subscribed message gets mapped onto the struct
- **`meta_field_map`**: to extract meta information from the kafka message, provide a meta_field_map dictionary of meta field info → struct field name to place it into.
  The following meta fields are currently supported:
  - **`"partition"`**: which partition the message came from
  - **`"offset"`**: the kafka offset of the given message
  - **`"live"`**: whether this message is "live" and not being replayed
  - **`"timestamp"`**: timestamp of the kafka message
  - **`"key"`**: key of the message
- **`push_mode`**: `csp.PushMode` (LAST_VALUE, NON_COLLAPSING, BURST)
- **`adjust_out_of_order_time`**: in some cases it has been seen that kafka can produce out of order messages, even for the same key.
  This allows the adapter to be more laz and allow it through by forcing time to max(time, prev time)

Similarly, you can publish on topics using the following method:

```python
KafkaAdapterManager.publish(
  msg_mapper: MsgMapper,
  topic: str,
  key: str,
  x: ts['T'],
  field_map: typing.Union[dict,str] = None
):
```

- **`msg_mapper`**: same as above
- **`topic`**: same as above
- **`key`**: key to publish to
- **`x`**: the timeseries to publish
- **`field_map`**: dictionary of {struct_field: message_field} to define how the struct gets mapped onto the published message.
  Note this dictionary is the opposite of the field_map in subscribe()

### Known Issues

If you are having issues, such as not getting any output or the application simply locking up, start by ensuring that you are logging the adapter's `status()` with a `csp.print`/`log` call and set `debug=True`.
Then follow the known issues below.

- Reason: `GSSAPI Error: Unspecified GSS failure.  Minor code may provide more information (No Kerberos credentials available)`

  - **Resolution**: Kafka uses kerberos tickets for authentication. Need to set-up kerberos token first

- `Message received on unknown topic: errcode: Broker: Group authorization failed error: FindCoordinator response error: Group authorization failed.`

  - **Resolution**: Kafka broker running on windows are case sensitive to kerberos token. When creating Kerberos token with kinit, make sure to use principal name with case sensitive user id.

- `authentication: SASL handshake failed (start (-4)): SASL(-4): no mechanism available: No worthy mechs found (after 0ms in state AUTH_REQ)`

  - **Resolution**: cyrus-sasl-gssapi needs to be installed on the box for Kafka kerberos authentication

- `Message error on topic "an-example-topic". errcode: Broker: Topic authorization failed error: Subscribed topic not available: an-example-topic: Broker: Topic authorization failed)`

  - **Resolution**: The user account does not have access to the topic

## Parquet

### ParquetReader

The `ParquetReader` adapter is a generic user adapter to stream data from [Apache Parquet](https://parquet.apache.org/) files as a CSP time series.
`ParquetReader` adapter supports only flat (non hierarchical) parquet files with all the primitive types that are supported by the CSP framework.

#### API

```python
ParquetReader(
  self,
  filename_or_list,
  symbol_column=None,
  time_column=None,
  tz=None
):
    """
    :param filename_or_list: The specifier of the file/files to be read. Can be either:
       - Instance of str, in which case it's interpreted os a path of single file to be read
       - A callable, in which case it's interpreted as a generator function that will be called like f(starttime, endtime) where starttime and endtime
         are the start and end times of the current engine run. It's expected to generate a sequence of filenames to read.
       - Iterable container, for example a list of files to read
    :param symbol_column: An optional parameter that specifies the name of the symbol column if the file if there is any
    :param time_column: A mandatory specification of the time column name in the parquet files. This column will be used to inject the row values
      from parquet at the given timestamps.
    :param tz: The pytz timezone of the timestamp column, should only be provided if the time_column in parquet file doesn't have tz info.
"""
```

#### Subscription

```python
def subscribe(
    self,
    symbol,
    typ,
    field_map=None,
    push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING
):
    """Subscribe to the rows corresponding to a given symbol
    This form of subscription can be used only if non empty symbol_column was supplied during ParquetReader construction.
    :param symbol: The symbol to subscribe to, for example 'AAPL'
    :param typ: The type of the CSP time series subscription. Can either be a primitive type like int or alternatively a type
    that inherits from csp.Struct, in which case each instance of the struct will be constructed from the matching file columns.
    :param field_map: A map of the fields from parquet columns for the CSP time series. If typ is a primitive, then field_map should be
    a string specifying the column name, if typ is a csp.Struct then field_map should be a str->str dictionary of the form
    {column_name:struct_field_name}. For structs field_map can be omitted in which case we expect a one to one match between the given Struct
    fields and the parquet files columns.
    :param push_mode: A push mode for the output adapter
    """

def subscribe_all(
    self,
    typ,
    field_map=None,
    push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING
):
    """Subscribe to all rows of the input files.
    :param typ: The type of the CSP time series subscription. Can either be a primitive type like int or alternatively a type
    that inherits from csp.Struct, in which case each instance of the struct will be constructed from the matching file columns.
    :param field_map: A map of the fields from parquet columns for the CSP time series. If typ is a primitive, then field_map should be
    a string specifying the column name, if typ is a csp.Struct then field_map should be a str->str dictionary of the form
    {column_name:struct_field_name}. For structs field_map can be omitted in which case we expect a one to one match between the given Struct
    fields and the parquet files columns.
    :param push_mode: A push mode for the output adapter
    """
```

Parquet reader provides two subscription methods.
**`subscribe`** produces a time series only of the rows that correspond to the given symbol,
\*\*`subscribe_all`\*\*produces a time series of all rows in the parquet files.

### ParquetWriter

The ParquetWriter adapter is a generic user adapter to stream data from CSP time series to [Apache Parquet](https://parquet.apache.org/) files.
`ParquetWriter` adapter supports only flat (non hierarchical) parquet files with all the primitive types that are supported by the CSP framework.
Any time series of Struct objects will be flattened to multiple columns.

#### Construction

```python
ParquetWriter(
    self,
    file_name: Optional[str],
    timestamp_column_name,
    config: Optional[ParquetOutputConfig] = None,
    filename_provider: Optional[csp.ts[str]] = None
):
    """
    :param file_name: The path of the output parquet file name. Must be provided if no filename_provider specified. If both file_name and filename_provider are specified then file_name will be used as the initial output file name until filename_provider provides a new file name.
    :param timestamp_column_name: Required field, if None is provided then no timestamp will be written.
    :param config: Optional configuration of how the file should be written (such as compression, block size,...).
    :param filename_provider: An optional time series that provides a times series of file paths. When a filename_provider time series provides a new file path, the previous open file name will be closed and all subsequent data will be written to the new file provided by the path. This enable partitioning and splitting the data based on time.
    """
```

#### Publishing

```python
def publish_struct(
    self,
    value: ts[csp.Struct],
    field_map: Dict[str, str] = None
):
    """Publish a time series of csp.Struct objects to file

    :param value: The time series of Struct objects that should be published.
    :param field_map: An optional dict str->str of the form {struct_field_name:column_name} that maps the names of the
    structure fields to the column names to which the values should be written. If the field_map is non None, then only
    the fields that are specified in the field_map will be written to file. If field_map is not provided then all fields
    of a structure will be written to columns that match exactly the field_name.
    """

def publish(
    self,
    column_name,
    value: ts[object]
):
    """Publish a time series of primitive type to file
    :param column_name: The name of the parquet file column to which the data should be written to
    :param value: The time series that should be published
    """
```

Parquet writer provides two publishing methods.
**`publish_struct`** is used to publish time series of **`csp.Struct`** objects while **`publish`** is used to publish primitive time series.
The columns in the written parquet file is a union of all columns that were published (the order is preserved).
A new row is written to parquet file whenever any of the inputs ticks.
For the given row, any column that corresponds to a time series that didn't tick, will have null values.

## DBReader

The DBReader adapter is a generic user adapter to stream data from a database as a reactive time series.
It leverages sqlalchemy internally in order to be able to access various DB backends.

Please refer to the [SQLAlchemy Docs](https://docs.sqlalchemy.org/en/13/core/tutorial.html) for information on how to create sqlalchemy connections.

The DBReader instance represents a single connection to a database.
From a single reader you can subscribe to various streams, either the entire stream of data (which would basically represent the result of a single join) or if a symbol column is declared, subscribe by symbol which will then demultiplex rows to the right adapter.

```python
DBReader(self, connection, time_accessor, table_name=None, schema_name=None, query=None, symbol_column=None, constraint=None):
        """
        :param connection: sqlalchemy engine or (already connected) connection object.
        :param time_accessor: TimeAccessor object
        :param table_name: name of table in database as a string
        :param query: either string query or sqlalchemy query object. Ex: "select * from users"
        :param symbol_column: name of symbol column in table as a string
        :param constraint: additional sqlalchemy constraints for query. Ex: constraint = db.text('PRICE>:price').bindparams(price = 100.0)
        """
```

- **connection**: seqlalchemy engine or existing connection object.
- **time_accessor**: see below
- **table_name**: either table or query is required.
  If passing a table_name then this table will be queried against for subscribe calls
- **query**: (optional) if table isn't supplied user can provide a direct query string or sqlalchemy query object.
  This is useful if you want to run a join call.
  For basic single-table queries passing table_name is preferred
- **symbol_column**: (optional) in order to be able to demux rows bysome column, pass `symbol_column`.
  Example case for this is if database has data stored for many symbols in a single table, and you want to have a timeseries tick per symbol.
- **constraint**: (optional) additional sqlalchemy constraints for query. Ex: `constraint = db.text('PRICE>:price').bindparams(price= 100.0)`

### TimeAccessor

All data fed into CSP must be time based.
`TimeAccessor` is a helper class that defines how to extract timestamp information from the results of the data.
Users can define their own `TimeAccessor` implementation or use pre-canned ones:

- `TimestampAccessor( self, time_column, tz=None)`: use this if there exists a single datetime column already.
  Provide the column name and optionally the timezone of the column (if its timezone-less in the db)
- `DateTimeAccessor(self, date_column, time_column, tz=None)`: use this if there are two separate columns for date and time, this accessor will combine the two columns to create a single datetime.
  Optionally pass tz if time column is timezone-less in the db

User implementations would have to extend `TimeAccessor` interface.
In addition to defining how to convert db columns to timestamps, accessors are also used to augment the query to limit the data for the graph's start and end times.

Once you have a DBReader object created, you can subscribe to time_series from it using the following methods:

- `subscribe(self, symbol, typ, field_map=None)`
- `subscribe_all(self, typ, field_map=None)`

Both of these calls expect `typ` to be a `csp.Struct` type.
`field_map` is a dictionary of `{ db_column : struct_column }` mappings that define how to map the database column names to the fields on the struct.

`subscribe` is used to subscribe to a stream for the given symbol (symbol_column is required when creating DBReader)

`subscribe_all` is used to retrieve all the data resulting from the request as a single timeseries.
