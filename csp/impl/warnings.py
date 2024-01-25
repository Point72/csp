class PythonicDeprecationWarning:
    def __init__(self, warn):
        self.warn = warn

    def __bool__(self):
        return self.warn


global WARN_PYTHONIC
WARN_PYTHONIC = PythonicDeprecationWarning(False)  # for now


def set_deprecation_warning(new_value: bool):
    """
    :param new_value: A boolean that specifies whether deprecation warnings should be printed for outdated csp syntax
    :return: The previous value that was set
    """
    old_value = WARN_PYTHONIC.warn
    WARN_PYTHONIC.warn = new_value
    return old_value
