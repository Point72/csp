# `csp` examples

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Example</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <!-- Basic -->
        <tr>
            <td rowspan=4><a href="./01_basics/">Basics</a></td>
            <td><a href="./01_basics/e1_basic.py">Basic Graph</a></td>
            <td>Basic sum of constant integers</td>
        </tr>
        <tr>
            <td><a href="./01_basics/e2_ticking.py">Ticking Graphs</a></td>
            <td>Cumulative sum of streaming integers</td>
        </tr>
        <tr>
            <td><a href="./01_basics/e3_show_graph.py">Visualizing a Graph</a></td>
            <td>Bid-ask spread of a trade with graph visualization</td>
        </tr>
        <tr>
            <td><a href="./01_basics/e4_trade_pnl.py">Complete Example (Trading)</a></td>
            <td>Volume weighted average price (VWAP) and profit and loss (PnL)</td>
        </tr>
        <!-- Intermediate -->
        <tr>
            <td rowspan=4><a href="./02_intermediate/">Intermediate</a></td>
            <td><a href="./02_intermediate/e1_feedback.py">Feedback Connections</a></td>
            <td>Add a feedback edge between nodes in a graph</td>
        </tr>
        <tr>
            <td><a href="./e02_intermediate/2_stats.py">Statistics Nodes</a></td>
            <td>Use the CSP statistics library on simulated trading data</td>
        </tr>
        <tr>
            <td><a href="./02_intermediate/e3_numpy_stats.py">Statistics Nodes with Numpy</a></td>
            <td>Rolling window statistics on a set of three symbols using NumPy arrays</td>
        </tr>
        <tr>
            <td><a href="./02_intermediate/e4_exprtk.py">Expression Nodes with <code>exprtk</code></a></td>
            <td>Calculate mathematical expressions with streaming data</td>
        </tr>
        <!-- Using Adapters -->
        <tr>
            <td rowspan=4><a href="./03_using_adapters/">Using Adapters</a></td>
            <td><a href="./03_using_adapters/kafka/e1_kafka.py">Kafka Adapter Example</a></td>
            <td>
                Stream data from a Kafka bus using KafkaAdapterManager and MessageMapper
            </td>
        </tr>
        <tr>
            <td><a href="./03_using_adapters/parquet/e1_parquet_write_read.py">Parquet Adapter Example</a></td>
            <td>
                Read from and write CSP data to Parquet files
            </td>
        </tr>
        <tr>
            <td><a href="./03_using_adapters/websocket/e1_websocket_client.py">Websocket Client</a></td>
            <td>
                Send and receive messages over a websocket connection
            </td>
        </tr>
        <tr>
            <td><a href="./03_using_adapters/websocket/e2_websocket_output.py">Websocket Output</a></td>
            <td>
                Send data over a websocket connection and view HTML output
            </td>
        </tr>
        <!-- Writing Adapters -->
        <tr>
            <td rowspan=8><a href="./04_writing_adapters/">Writing Adapters</a></td>
            <td><a href="./04_writing_adapters/e1_generic_push_adapter.py">Generic Push Adapter</a></td>
            <td>
                Push real-time data into a CSP graph
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e2_pullinput.py">Pull Input Adapter</a></td>
            <td>
                Replay historical data using a pull adapter
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e3_adaptermanager_pullinput.py">Pull Input Adapter with Adapter
                    Manager</a></td>
            <td>
                Provide data to multiple input adapters from a single source
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e4_pushinput.py">Push Input Adapter</a></td>
            <td>
                Write your own push adapter for real-time data
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e5_adaptermanager_pushinput.py">Push Input Adapter with Adapter
                    Manager</a></td>
            <td>
                Use an adapter manager with real-time data sources
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e6_outputadapter.py">Output Adapter</a></td>
            <td>
                Create a custom writer for CSP output data
            </td>
        </tr>
        <tr>
            <td><a href="./04_writing_adapters/e7_adaptermanager_inputoutput.py">Complete Input/Output Adapter with
                    Adapter Manager</a></td>
            <td>
                Manage input and output adapters with a single adapter manager
            </td>
        </tr>
        <tr>
            <td><a href="./07_end_to_end/earthquake.ipynb">Push-Pull Input Adapter for Earthquake Data</a></td>
            <td>
                Create a push-pull adapter which transitions from replay to live execution
            </td>
        </tr>
        <!-- Writing C++ Nodes and Adapters -->
        <tr>
            <td rowspan=2><a href="./05_cpp/">Writing C++ Nodes and Adapters</a></td>
            <td><a href="./05_cpp/1_cpp_node/">C++ Node</a></td>
            <td>
                Extend CSP with a pig latin C++ node 
            </td>
        </tr>
        <tr>
            <td><a href="./05_cpp/2_cpp_node_with_struct/">C++ Node with <code>csp.Struct</code></a></td>
            <td>
                Write a C++ node with a csp.Struct input
            </td>
        </tr>
        <!-- Advanced -->
        <tr>
            <td rowspan=2><a href="./06_advanced/">Advanced</a></td>
            <td><a href="./06_advanced/e1_dynamic.py">Dynamic Graphs</a></td>
            <td>
                Update the shape of a graph at runtime
            </td>
        </tr>
        <tr>
            <td><a href="./06_advanced/e2_pandas_extension.py">Pandas Extension</a></td>
            <td>
                Use CSP within a pandas DataFrame
            </td>
        </tr>
        <!-- End-to-end examples -->
        <tr>
            <td rowspan=4><a href="./07_end_to_end/">End-to-end examples</a></td>
            <td><a href="./07_end_to_end/mta.ipynb">MTA Subway Data</a></td>
            <td>
                Access real-time New York City transit data
            </td>
        </tr>
        <tr>
            <td><a href="./07_end_to_end/seismic_waveform.ipynb">Seismic Data with obspy</a></td>
            <td>
                Analyze seismic waveforms and compare with batch processing methods
            </td>
        </tr>
        <tr>
            <td><a href="./07_end_to_end/wikimedia.ipynb">Wikipedia Updates and Edits</a></td>
            <td>
                Monitor live updates to all Wikimedia sites
            </td>
        </tr>
        <tr>
            <td><a href="./07_end_to_end/earthquake.ipynb">World Earthquake Dashboard</a></td>
            <td>
                Display recent earthquakes on a live-updating world map
            </td>
        </tr>
        <!-- Others -->
        <tr>
            <td><a href="./98_just_for_fun/">Just for fun!</a></td>
            <td><a href="./98_just_for_fun/e1_csp_nand_computer.py">NAND Computer</a></td>
            <td>
                Understand <code>csp.node</code> & <code>csp.graph</code> by connecting NAND logic gates
            </td>
        </tr>
        <tr>
            <td><a href="./99_developer_tools/">Developer Tools</a></td>
            <td><a href="./99_developer_tools/e1_profiling.py">Profiling <code>csp</code> code</a></td>
            <td>
                Profile a CSP Graph, view static attributes and runtime performance
            </td>
        </tr>
    </tbody>
</table>
