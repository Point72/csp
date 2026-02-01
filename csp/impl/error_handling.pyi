"""Type stubs for csp error handling."""

def set_print_full_exception_stack(new_value: bool) -> None:
    """
    Set whether to print the full exception stack on errors.

    When enabled, csp will print more detailed stack traces
    including internal frames.

    Args:
        value: True to enable full stack traces
    """
    ...

class ExceptionContext:
    """Context manager for exception handling in csp."""

    def __enter__(self) -> "ExceptionContext": ...
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> bool: ...
