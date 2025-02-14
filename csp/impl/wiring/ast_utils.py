# For managing AST changes across python versions
import ast


class ASTUtils39:
    @staticmethod
    def get_subscript_index(subscript: ast.Subscript):
        return subscript.slice

    @staticmethod
    def create_subscript_index(value):
        return value


ASTUtils = ASTUtils39
