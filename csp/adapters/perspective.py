try:
    from csp_adapter_perspective import *  # noqa: F403
except ImportError:
    raise ModuleNotFoundError("Install `csp-adapter-perspective` to use csp's perspective adapter")
