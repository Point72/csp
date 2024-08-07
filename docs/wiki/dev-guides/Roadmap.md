We do not have a formal roadmap, but we're happy to discuss features, improvements, new adapters, etc, in our [discussions area](https://github.com/Point72/csp/discussions).

Here are some high level items we hope to accomplish in the next few months:

- Establish a better pattern for adapters ([#165](https://github.com/Point72/csp/discussions/165))
- Parallelization to improve runtime, for historical/offline distributions
- Support for cross-process communication in realtime distributions

## Adapters and Extensions

- C++-based HTTP/SSE adapter
- C++-based Redis adapter
- Add support for other graph viewers, including interactive / standalone / Jupyter

## Other Open Source Projects

- `csp-gateway`: Application development framework, built with [FastAPI](https://fastapi.tiangolo.com) and [Perspective](https://github.com/finos/perspective). This is a library we have built internally at Point72 on top of `csp` that we hope to open source later in 2024. It allows for easier construction of modular `csp` applications, along with a pluggable REST/WebSocket API and interactive UI.
