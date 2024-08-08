The CSP engine can be run in two flavors, realtime and simulation.

In simulation mode, the engine is always run at full speed pulling in time-based data from its input adapters and running them through the graph.
All inputs in simulation are driven off the provided timestamped data of its inputs.

In realtime mode, the engine runs in wallclock time as of "now".
Realtime engines can get data from realtime adapters which source data on separate threads and pass them through to the engine (ie think of activeMQ events happening on an activeMQ thread and being passed along to the engine in "realtime").

Since engines can run in both simulated and realtime mode, users should **always** use **`csp.now()`** to get the current time in a `csp.node`.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Simulation Mode](#simulation-mode)
- [Realtime Mode](#realtime-mode)
- [csp.PushMode](#csppushmode)
- [Handling Duplicate Timestamps](#handling-duplicate-timestamps)
- [Realtime Group Event Synchronization](#realtime-group-event-synchronization)

## Simulation Mode

Simulation mode is the default mode of the engine.
As stated above, simulation mode is used when you want your engine to crunch through historical data as fast as possible.
In simulation mode, the engine runs on some historical data that is fed in through various adapters.
The adapters provide events by time, and they are streamed into the engine via the adapter timeseries in time order.
`csp.timer` and `csp.node` alarms are scheduled and executed in "historical time" as well.
Note that there is no strict requirement for simulated runs to run on historical dates.
As long as the engine is not in realtime mode, it remains in simulation mode until the provided endtime, even if endtime is in the future.

## Realtime Mode

Realtime mode is opted into by passing `realtime=True` to `csp.run(...)`.
When run in realtime mode, the engine will run in simulation mode from the provided starttime → wallclock "now" as of the time of calling run.
Once the simulation run is done, the engine switches into realtime mode.
Under realtime mode, external realtime adapters will be able to send data into the engine thread.
All time based inputs such as `csp.timer` and alarms will switch to executing in wallclock time as well.

As always, `csp.now()` should still be used in `csp.node` code, even when running in realtime mode.
`csp.now()` will be the time assigned to the current engine cycle.

## csp.PushMode

When consuming data from input adapters there are three choices on how one can consume the data:

| PushMode           | EngineMode | Description                                                                                                                                                       |
| :----------------- | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LAST_VALUE**     | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once with the last value on a given timestamp                            |
|                    | Realtime   | all ticks that occurred since previous engine cycle will collapse / conflate to the latest value                                                                  |
| **NON_COLLAPSING** | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once per engine cycle. subsequent cycles will execute with the same time |
|                    | Realtime   | all ticks that occurred since previous engine cycle will be ticked across subsequent engine cycles as fast as possible                                            |
| **BURST**          | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once with a list of all values                                           |
|                    | Realtime   | all ticks that occurred since previous engine cycle will tick once with a list of all the values                                                                  |

## Handling duplicate timestamps

In `csp`, there can be multiple engine cycles that occur at the same engine time. This is often the case when using nodes with internal alarms (e.g. [`csp.unroll`](Base-Nodes-API#cspunroll)) or using feedback edges ([`csp.feedback`](Feedback-and-Delayed-Edge#cspfeedback)).
If multiple events are scheduled at the same timestamp on a single time-series edge, they will be executed on separate cycles *in the order* they were scheduled. For example, consider the code snippet below:

```python
import csp
from csp import ts
from datetime import datetime, timedelta

@csp.node
def ticks_n_times(x: ts[int], n: int) -> ts[int]:
    # Ticks out a value n times, incrementing it each time
    with csp.alarms():
        alarm = csp.alarm(int)

    if csp.ticked(x):
        for i in range(n):
            csp.schedule_alarm(alarm, timedelta(), x+i)

    if csp.ticked(alarm):
        return alarm

@csp.graph
def duplicate_timestamps():
    v = csp.const(1)
    csp.print('ticks_once', ticks_n_times(v, 1))
    csp.print('ticks_twice', ticks_n_times(v, 2))
    csp.print('ticks_thrice', ticks_n_times(v, 3))

csp.run(duplicate_timestamps, starttime=datetime(2020,1,1))
```

When we run this graph, the output is:

```raw
2020-01-01 00:00:00 ticks_once:1
2020-01-01 00:00:00 ticks_twice:1
2020-01-01 00:00:00 ticks_thrice:1
2020-01-01 00:00:00 ticks_twice:2
2020-01-01 00:00:00 ticks_thrice:2
2020-01-01 00:00:00 ticks_thrice:3
```

A real life example is when using `csp.unroll` to tick out a list of values on separate engine cycles. If we were to use `csp.sample` on the output, we would get the *first* value that is unrolled at each timestamp. Why?
The event that is scheduled on the sampling timer is its first (and only) event at that time; thus, it is executed on the first engine cycle, and samples the first unrolled value.

```python
def sampling_unroll():
    u = csp.unroll(csp.const.using(T=[int])([1, 2, 3]))
    s = csp.sample(csp.const(True), u)
    csp.print('unrolled', u)
    csp.print('sampled', s)
    
csp.run(sampling_unroll, starttime=datetime(2020,1,1))
```

Output:

```raw
2020-01-01 00:00:00 unrolled:1
2020-01-01 00:00:00 sampled:1
2020-01-01 00:00:00 unrolled:2
2020-01-01 00:00:00 unrolled:3
```

## Realtime Group Event Synchronization

The CSP framework supports properly synchronizing events across multiple timeseries that are sourced from the same realtime adapter.
A classical example of this is a market data feed.
Say you consume bid, ask and trade as 3 separate time series for the same product / exchange.
Since the data flows in asynchronously from a separate thread, bid, ask and trade events could end up executing in the engine at arbitrary slices of time, leading to crossed books and trades that are out of range of the bid/ask.
The engine can properly provide a correct synchronous view of all the inputs, regardless of their PushModes.
Its up to adapter implementations to determine which inputs are part of a synchronous "PushGroup".

Here's a classical example.
An Application wants to consume conflating bid/ask as LAST_VALUE but it doesn't want to conflate trades, so its consumed as NON_COLLAPSING.

Lets say we have this sequence of events on the actual market data feed's thread, coming in one the wire in this order.
The columns denote the time the callbacks come in off the market data thread.

<table>
<tbody>
<tr>
<th>Event</th>
<th>T</th>
<th>T+1</th>
<th>T+2</th>
<th>T+3</th>
<th>T+4</th>
<th>T+5</th>
<th>T+6</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.00</td>
<td>100.01</td>
<td><br />
</td>
<td>99.97</td>
<td>99.98</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.02</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
<td><br />
</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td><br />
</td>
<td><br />
</td>
<td>100.02</td>
<td><br />
</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>

Without any synchronization you can end up with nonsensical views based on random timing.
Here's one such possibility (bid/ask are still LAST_VALUE, trade is NON_COLLAPSING).

Over here ET is engine time.
Lets assume engine had a huge delay and hasn't processed any data submitted above yet.
Without any synchronization, bid/ask would completely conflate, and trade would unroll over multiple engine cycles

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET</th>
<th>ET+1</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.00</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td>100.02</td>
<td>100.03</td>
</tr>
</tbody>
</table>

However, since market data adapters will group bid/ask/trade inputs together, the engine won't let bid/ask events advance ahead of trade events since trade is NON_COLLAPSING.
NON_COLLAPSING inputs will essentially act as a barrier, not allowing events ahead of the barrier tick before the barrier is complete.
Lets assume again that the engine had a huge delay and hasn't processed any data submitted above.
With proper barrier synchronizations the engine cycles would look like this under the same conditions:

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET</th>
<th>ET+1</th>
<th>ET+2</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.01</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.03</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td>100.02</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>

Note how the last ask tick of 100.00 got held up to a separate cycle (ET+2) so that trade could tick with the correct view of bid/ask at the time of the second trade (ET+1)

As another example, lets say the engine got delayed briefly at wire time T, so it was able to process T+1 data.
Similarly it got briefly delayed at time T+4 until after T+6.  The engine would be able to process all data at time T+1, T+2, T+3 and T+6, leading to this sequence of engine cycles.
The equivalent "wire time" is denoted in parenthesis

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET (T+1)</th>
<th>ET+1 (T+2)</th>
<th>ET+2 (T+3)</th>
<th>ET+3 (T+5)</th>
<th>ET+4 (T+6)</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.01</td>
<td><br />
</td>
<td>99.97</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.02</td>
<td>100.03</td>
<td><br />
</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td><br />
</td>
<td>100.02</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>
