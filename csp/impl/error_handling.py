import ast
import os

from pydantic import ValidationError
from pydantic.version import version_short
from pydantic_core import ErrorDetails

import csp.impl
from csp.impl.__cspimpl import _cspimpl


class ExceptionContext:
    PRINT_EXCEPTION_FULL_STACK = False

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.PRINT_EXCEPTION_FULL_STACK:
            from csp.impl.wiring.base_parser import CspParseError

            if exc_val is not None:
                ast_folder = os.path.dirname(ast.__file__)
                impl_folder = os.path.dirname(csp.impl.__file__)
                ignore_prefixes = [impl_folder, ast_folder]
                filtered_tb = []
                while exc_tb is not None:
                    if not any(exc_tb.tb_frame.f_code.co_filename.startswith(prefix) for prefix in ignore_prefixes):
                        if not filtered_tb or filtered_tb[-1] != exc_tb:
                            filtered_tb.append(exc_tb)
                    exc_tb = exc_tb.tb_next

                new_tb = None
                if filtered_tb:
                    new_tb = _cspimpl.create_traceback(
                        None, filtered_tb[-1].tb_frame, filtered_tb[-1].tb_lasti, filtered_tb[-1].tb_lineno
                    )
                    for old_tb in reversed(filtered_tb[:-1]):
                        new_tb = _cspimpl.create_traceback(new_tb, old_tb.tb_frame, old_tb.tb_lasti, old_tb.tb_lineno)

                if exc_type is CspParseError and exc_val.filename is not None and exc_val.lineno is not None:
                    new_tb = _cspimpl.create_traceback(new_tb, exc_val.frame, exc_val.lineno, exc_val.lineno)

                raise exc_val.with_traceback(new_tb)


def set_print_full_exception_stack(new_value: bool):
    """
    :param new_value: A boolean that specifies whether a full exception stack should be printed for debugging purposes
    :return: The previous value that was set
    """
    res = ExceptionContext.PRINT_EXCEPTION_FULL_STACK
    ExceptionContext.PRINT_EXCEPTION_FULL_STACK = new_value
    return res


# ValidationError formatting helpers

INPUT_VALUE_TRUNCATE_LENGTH = int(os.getenv("CSP_INPUT_VALUE_TRUNCATE_LENGTH", "300"))


def fmt_loc(loc: tuple[int | str, ...], prefix: str) -> str:
    def fmt_loc_item(loc_item: int | str) -> str:
        """
        See https://github.com/pydantic/pydantic-core/blob/15b9c7b/src/errors/location.rs#L27-L33
        """
        match loc_item:
            case str() if "." in loc_item:
                return f"`{loc_item}`"
            case _:
                return str(loc_item)

    return ".".join(fmt_loc_item(loc_item) for loc_item in loc).replace(prefix, "")


def get_error_url(error_type: str) -> str:
    return f"https://errors.pydantic.dev/{version_short()}/v/{error_type}"


def truncate_input_value(input_value: str) -> str:
    if (input_len := len(input_value)) > INPUT_VALUE_TRUNCATE_LENGTH:
        mid_point = (INPUT_VALUE_TRUNCATE_LENGTH + 1) // 2
        left_end = max(mid_point - 3, 0)
        right_start = min(input_len - mid_point + 5, input_len)
        return f"{input_value[:left_end]}...{input_value[right_start:]}"
    return input_value


def fmt_line_error(error_details: ErrorDetails, prefix: str) -> str:
    """
    See https://github.com/pydantic/pydantic-core/blob/15b9c7b/src/errors/validation_exception.rs#L527-L572
    """
    error_type = error_details["type"]
    output = [
        fmt_loc(error_details["loc"], prefix),
        f"\n  {error_details['msg']} [type={error_type}",
    ]
    if error_type != "default_factory_not_called":
        input_value = error_details["input"]
        input_type = type(input_value)
        input_type_str = (
            f"{input_type.__module__}." if input_type.__module__ != "builtins" else ""
        ) + input_type.__qualname__
        output.append(f", input_value={truncate_input_value(repr(input_value))}, input_type={input_type_str}")
    output.append(f"]\n    For further information visit {get_error_url(error_type)}")
    return "".join(output)


def fmt_errors(e: ValidationError, prefix: str) -> str:
    """
    See https://github.com/pydantic/pydantic-core/blob/15b9c7b/src/errors/validation_exception.rs#L96-L107
    """
    errors = e.errors()
    count = len(errors)
    line_errors = "\n".join(fmt_line_error(error, prefix) for error in errors)
    plural = "" if count == 1 else "s"
    title = e.title.replace(prefix, "")
    return f"{count} validation error{plural} for {title}\n{line_errors}"
