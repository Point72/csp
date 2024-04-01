We do not have a formal roadmap, but we're happy to discuss features, improvements, new adapters, etc, in our [discussions area](https://github.com/Point72/csp/discussions). Here are some high level items we hope to accomplish in the next few months:

- Support `clang` compiler ([#33](https://github.com/Point72/csp/issues/33))
- Redis Pub/Sub Adapter with [Redis-plus-plus](https://github.com/sewenew/redis-plus-plus) ([#61](https://github.com/Point72/csp/issues/61))
- C++-based websocket adapter
- Slack Adapter with [python-slack-sdk](https://github.com/slackapi/python-slack-sdk) ([#17](https://github.com/Point72/csp/issues/17))
- `csp-gateway`: Application development framework, built with [FastAPI](https://fastapi.tiangolo.com) and [Perspective](https://github.com/finos/perspective). This is a library we have built internally at Point72 on top of CSP that we hope to open source later in 2024. It allows for easier construction of modular CSP applications, along with a pluggable REST/WebSocket API and interactive UI.
- Interactive graph viewer, both standalone and viewable in Jupyter (with something like [ipydagred3](https://github.com/timkpaine/ipydagred3))
- Sympy integration via [`lambdify`](https://docs.sympy.org/latest/modules/utilities/lambdify.html) ([#59](https://github.com/Point72/csp/issues/59))
