try:
    from csp_adapter_symphony import *  # noqa: F403
except ImportError:
    raise ModuleNotFoundError("Install `csp-adapter-symphony` to use csp's Symphony adapter")
