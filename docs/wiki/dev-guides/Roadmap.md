We do not have a formal roadmap, but we're happy to discuss features, improvements, new adapters, etc, in our [discussions area](https://github.com/Point72/csp/discussions). Here are some high level items we hope to accomplish in the next few months:

- Support `clang` compiler and full MacOS support ([#33](https://github.com/Point72/csp/issues/33) / [#132](https://github.com/Point72/csp/pull/132))
- Support `msvc` compiler and full Windows support ([#109](https://github.com/Point72/csp/issues/109))
- Establish a better pattern for adapters ([#165](https://github.com/Point72/csp/discussions/165))

## Adapters and Extensions

- Redis Pub/Sub Adapter with [Redis-plus-plus](https://github.com/sewenew/redis-plus-plus) ([#61](https://github.com/Point72/csp/issues/61))
- C++-based websocket adapter
  - Client adapter in [#152](https://github.com/Point72/csp/pull/152)
- C++-based HTTP/SSE adapter
- Add support for other graph viewers, including interactive / standalone / Jupyter

## Other Open Source Projects

- `csp-gateway`: Application development framework, built with [FastAPI](https://fastapi.tiangolo.com) and [Perspective](https://github.com/finos/perspective). This is a library we have built internally at Point72 on top of `csp` that we hope to open source later in 2024. It allows for easier construction of modular `csp` applications, along with a pluggable REST/WebSocket API and interactive UI.
