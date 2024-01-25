# For managing AST changes across python versions
import ast
import sys


class ASTUtils38:
    @staticmethod
    def get_subscript_index(subscript: ast.Subscript):
        return subscript.slice.value

    @staticmethod
    def create_subscript_index(value):
        return ast.Index(value=value)


class ASTUtils39:
    @staticmethod
    def get_subscript_index(subscript: ast.Subscript):
        return subscript.slice

    @staticmethod
    def create_subscript_index(value):
        return value


if sys.version_info.major > 3 or sys.version_info.minor >= 9:
    ASTUtils = ASTUtils39
else:
    ASTUtils = ASTUtils38
