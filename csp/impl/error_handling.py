import ast
import os

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
