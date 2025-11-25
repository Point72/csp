# Using `is_configured_realtime()` for Adapter Selection

The `csp.is_configured_realtime()` function allows you to check whether your graph is running in **realtime mode** or **historical (backtest) mode** at graph-building time. This is useful for selecting different adapters or behavior based on the execution mode.

## What It Does

Returns `True` if the graph will run in realtime mode, `False` if running in historical/backtest mode.

## When to Use

You typically use `is_configured_realtime()` in your `@csp.graph` function to conditionally wire different adapters:

```python
import csp

@csp.graph
def my_graph():
    if csp.is_configured_realtime():
        # Use live/realtime adapter
        data = FetchLiveData(url="wss://stream.example.com")
    else:
        # Use historical adapter for backtesting
        data = FetchHistoricalData(start_date="2023-01-01")
    
    # Rest of graph uses 'data' regardless of source
    result = process_data(data)
    csp.add_graph_output("output", result)

# Run in historical mode (backtest)
csp.run(my_graph, starttime=datetime(2023, 1, 1), endtime=datetime(2023, 1, 31))

# Run in realtime mode (live)
csp.run(my_graph, starttime=datetime.utcnow(), endtime=datetime.utcnow() + timedelta(hours=1), realtime=True)
```

## Practical Example: Wikimedia Stream

The wikimedia example demonstrates this pattern perfectly. It uses the same graph to read either live or historical data:

```python
from datetime import datetime, timedelta

@csp.graph
def wiki_graph():
    if csp.is_configured_realtime():
        # Live realtime feed
        URL = "https://stream.wikimedia.org/v2/stream/recentchange"
        events = FetchWikiData(url=URL)
    else:
        # Historical data with time range
        URL = f"https://stream.wikimedia.org/v2/stream/recentchange?since={start_time}"
        events = HistoricalWikiData(url=URL)

    # Same processing regardless of adapter
    en_wiki = csp.filter(events.servername == "en.wikipedia.org", events)
    csp.print("Wiki event:", en_wiki)

# Backtest: read historical data
start_time = datetime.utcnow() - timedelta(days=2)
end_time = start_time + timedelta(hours=23)
csp.run(wiki_graph, starttime=start_time, endtime=end_time)

# Live: read realtime events
csp.run(
    wiki_graph,
    starttime=datetime.utcnow(),
    endtime=datetime.utcnow() + timedelta(seconds=30),
    realtime=True
)
```

## Key Points

1. **Graph-building time only** — Use `is_configured_realtime()` during graph construction (`@csp.graph` functions), not inside `@csp.node` functions during execution.

2. **One graph, two modes** — By conditionally selecting adapters, you can use the same business logic for both backtesting and production.

3. **Common pattern** — Most adapters come in two flavors:
   - **Historical adapter** (PullInputAdapter) — reads from a data file or historical API
   - **Realtime adapter** (PushInputAdapter) — streams live data from a live source

4. **Read-only** — This function only reads the configuration; it doesn't modify behavior.

## Related Functions

- `csp.run(..., realtime=True)` — Set the realtime flag when running the graph
- `csp.starttime, csp.endtime` — Parameters passed to adapters; realtime adapters typically ignore `endtime`

## See Also

- [Write Realtime Input Adapters](Write-Realtime-Input-Adapters.md)
- [Write Historical Input Adapters](Write-Historical-Input-Adapters.md)
- [Adapters](../concepts/Adapters.md)
