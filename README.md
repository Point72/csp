<br />
<a href="https://github.com/point72/csp#gh-light-mode-only">
  <img src="https://github.com/point72/csp/raw/main/docs/img/csp-light.png?raw=true#gh-light-mode-only" alt="csp" width="400"></a>
</a>
<a href="https://github.com/point72/csp#gh-dark-mode-only">
  <img src="https://github.com/point72/csp/raw/main/docs/img/csp-dark.png?raw=true#gh-dark-mode-only" alt="csp" width="400"></a>
</a>
<br/>

[![PyPI](https://img.shields.io/pypi/v/csp.svg?style=flat)](https://pypi.python.org/pypi/csp)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/Point72/csp/LICENSE)
[![Build Status](https://github.com/Point72/csp/actions/workflows/build.yml/badge.svg)](https://github.com/Point72/csp/actions/workflows/build.yml)
[![Python Versions](https://img.shields.io/badge/python-3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue)](https://github.com/Point72-OSPO/csp/blob/pyproject.toml)

<br/>

`csp` is a high performance reactive stream processing library. The main engine is a C++ complex event graph processor, with bindings exposed into Python. Its key features include switchable simulation/realtime timesteps for both offline and online processing, custom input and output adapters for integration with static and streaming data sources and sinks, and extensible acceleration via customizeable C++ nodes for calculations.

The high level goal of `csp` is to make writing realtime code simple and performant. Write event driven code once, test it in simulation, then deploy as realtime without any code changes.


Here is a very simple example of a small `csp` program to calculate a [bid-ask spread](https://www.investopedia.com/terms/b/bid-askspread.asp). In this example, we use a constant bid and ask, but in the real world you might pipe these directly into your live streaming data source, or into your historical data source, without modifications to your core logic.


```python
import csp
from csp import ts
from datetime import datetime


@csp.node
def spread(bid: ts[float], ask: ts[float]) -> ts[float]:
    if csp.valid(bid, ask):
        return ask - bid


@csp.graph
def my_graph():
    bid = csp.const(1.0)
    ask = csp.const(2.0)
    bid = csp.multiply( bid, csp.const(4) )
    ask = csp.multiply( ask, csp.const(3) )
    s = spread(bid, ask)

    csp.print('spread', s)
    csp.print('bid', bid)
    csp.print('ask', ask)


if __name__ == '__main__':
    csp.run(my_graph, starttime=datetime.utcnow())
```


## Getting Started
See [our wiki!](https://github.com/Point72/csp/wiki)

## Development
Check out the [Developer Documentation](https://github.com/Point72/csp/wiki/99.-Developer)

## License
This software is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.
