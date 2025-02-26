try:
    from csp_adapter_slack import *  # noqa: F403
except ImportError:
    raise ModuleNotFoundError("Install `csp-adapter-slack` to use csp's Slack adapter")
