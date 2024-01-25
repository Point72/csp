try:
    # Import available at the top level when running csp_autogen
    import _csptypesimpl
except ModuleNotFoundError:
    from csp.lib import _csptypesimpl
