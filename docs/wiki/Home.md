`csp` ("Composable Stream Processing") is a functional-like reactive
language that makes time-series stream processing simple to do.  The
main reactive engine is a C++ based engine which has been exposed to
python ( other languages may optionally be extended in future versions
). `csp` applications define a connected graph of components using a
declarative language (which is essentially python).  Once a graph is
constructed it can be run using the C++ engine. Graphs are composed of
some number of "input" adapters, a set of connected calculation "nodes"
and at the end sent off to "output" adapters. Inputs as well as the
engine can be seamlessly run in simulation mode using historical input
adapters or in realtime mode using realtime input adapters.

# Contents

- [0. Introduction](https://github.com/Point72/csp/wiki/0.-Introduction)
- [1. Generic Nodes (csp.baselib)](<https://github.com/Point72/csp/wiki/1.-Generic-Nodes-(csp.baselib)>)
- [2. Math Nodes (csp.math)](<https://github.com/Point72/csp/wiki/2.-Math-Nodes-(csp.math)>)
- [3. Statistics Nodes (csp.stats)](<https://github.com/Point72/csp/wiki/3.-Statistics-Nodes-(csp.stats)>)
- [4. Random Time Series Generation](<https://github.com/Point72/csp/wiki/4.-Random-Time-Series-Generation-(csp.random)>)
- [5. Adapters](https://github.com/Point72/csp/wiki/5.-Adapters)
- [6. Dynamic Graphs](https://github.com/Point72/csp/wiki/6.-Dynamic-Graphs)
- [7. csp.Struct](https://github.com/Point72/csp/wiki/7.-csp.Struct)
- [8. Profiler](https://github.com/Point72/csp/wiki/8.-Profiler)
- [9. Caching](https://github.com/Point72/csp/wiki/9.-Caching)

# Installation

We ship binary wheels to install `csp`  on MacOS and Linux via `pip`:

```bash
pip install csp
```

Other platforms will need to see the instructions to [build `csp` from
source](https://github.com/Point72/csp/wiki/98.-Building-From-Source).

We plan to create conda packages on conda-forge and ship binaries for Windows in
the near future.

# Contributing

Contributions are welcome on this project. We distribute under the terms of the [Apache 2.0 license](https://github.com/Point72/csp/blob/main/LICENSE).

For **bug reports** or **small feature requests**, please open an issue on our [issues page](https://github.com/Point72/csp/issues).

For **questions** or to discuss **larger changes or features**, please use our [discussions page](https://github.com/Point72/csp/discussions).

For **contributions**, please see our [developer documentation](https://github.com/Point72/csp/wiki/99.-Developer). We have `help wanted` and `good first issue` tags on our issues page, so these are a great place to start.

For **documentation updates**, make PRs that update the pages in `/docs/wiki`. The documentation is pushed to the GitHub wiki automatically through a GitHub workflow. Note that direct updates to this wiki will be overwritten.

# Roadmap

We do not have a formal roadmap, but we're happy to discuss features, improvements, new adapters, etc, in our [discussions area](https://github.com/Point72/csp/discussions). Here are some high level items we hope to accomplish in the next few months:

- Support `clang` compiler ([#33](https://github.com/Point72/csp/issues/33))
- Redis Pub/Sub Adapter with [Redis-plus-plus](https://github.com/sewenew/redis-plus-plus) ([#61](https://github.com/Point72/csp/issues/61))
- C++-based websocket adapter
- Slack Adapter with [python-slack-sdk](https://github.com/slackapi/python-slack-sdk) ([#17](https://github.com/Point72/csp/issues/17))
- `csp-gateway`: Application development framework, built with [FastAPI](https://fastapi.tiangolo.com) and [Perspective](https://github.com/finos/perspective). This is a library we have built internally at Point72 on top of `csp` that we hope to open source later in 2024. It allows for easier construction of modular `csp` applications, along with a pluggable REST/WebSocket API and interactive UI.
- Interactive graph viewer, both standalone and viewable in Jupyter (with something like [ipydagred3](https://github.com/timkpaine/ipydagred3))
- Sympy integration via [`lambdify`](https://docs.sympy.org/latest/modules/utilities/lambdify.html) ([#59](https://github.com/Point72/csp/issues/59))
