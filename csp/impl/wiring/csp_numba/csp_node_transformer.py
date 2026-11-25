import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from csp.impl.wiring.node_parser import ParsedNodeDefinition


@dataclass
class TransformedNode:
    transformed_ast: ast.FunctionDef
    start_body: List[ast.AST]
    stop_body: List[ast.AST]


class CspNodeTransformer(ast.NodeTransformer):
    """
    Transforms CSP node AST for Numba compilation.

    Transformations:
    - csp.ticked(a, b, ...) -> a.ticked() or b.ticked() or ...
    - csp.valid(a, b, ...) -> a.valid() and b.valid() and ...
    - with csp.state(): variables -> State[type] annotations
    - csp.output(a=x, b=y, ...) -> set_output('a', x), set_output('b', y), ...

    NOT transformed (kept as CSP native):
    - ts[type] annotations (inputs and outputs)
    """

    def __init__(self):
        self.csp_call_transformers = {
            "ticked": lambda node: self._transform_signal_check(
                node=node,
                attr_name="ticked",
                op=ast.Or(),
            ),
            "valid": lambda node: self._transform_signal_check(
                node=node,
                attr_name="valid",
                op=ast.And(),
            ),
        }

    def _infer_type(self, name: str, value: ast.AST) -> ast.AST:
        if isinstance(value, ast.Constant):
            val = value.value
            if isinstance(val, bool):
                return ast.Name(id="bool", ctx=ast.Load())
            elif isinstance(val, int):
                return ast.Name(id="int", ctx=ast.Load())
            elif isinstance(val, float):
                return ast.Name(id="float", ctx=ast.Load())
            elif isinstance(val, str):
                return ast.Name(id="str", ctx=ast.Load())
        elif isinstance(value, ast.List):
            return ast.Name(id="list", ctx=ast.Load())
        elif isinstance(value, ast.Dict):
            return ast.Name(id="dict", ctx=ast.Load())
        elif isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                return ast.Name(id=value.func.id, ctx=ast.Load())

        raise ValueError(
            f"Unable to infer type for state variable '{name}' from expression "
            f"'{ast.unparse(value)}'; add an explicit annotation"
        )

    def _is_csp_call(self, node: ast.Call, method_name: str) -> bool:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "csp":
                return node.func.attr == method_name
        return False

    def _transform_signal_check(self, *, node: ast.Call, attr_name: str, op: ast.boolop) -> ast.AST:
        """
        1. csp.ticked(a, b, ...) -> a.ticked() or b.ticked() or ...
        2. csp.valid(a, b, ...) -> a.valid() and b.valid() and ...
        """
        if not node.args:
            raise ValueError(f"csp.{attr_name} requires at least one argument")

        signal_checks = []
        for arg in node.args:
            signal_check = ast.Call(
                func=ast.Attribute(value=self.visit(arg), attr=attr_name, ctx=ast.Load()), args=[], keywords=[]
            )
            signal_checks.append(signal_check)

        if len(signal_checks) == 1:
            return signal_checks[0]

        return ast.BoolOp(op=op, values=signal_checks)

    def _transform_output(self, node: ast.Call) -> List[ast.AST]:
        """Transform csp.output(...) calls into set_output(...) helper calls."""
        if node.args:
            if node.keywords or len(node.args) > 2:
                raise ValueError("csp.output accepts one value, (name, value), or named output values")
            if len(node.args) == 2:
                if not isinstance(node.args[0], ast.Name):
                    raise TypeError("csp.output(name, value) requires a declared output name")
                outputs = ((node.args[0].id, node.args[1]),)
            else:
                outputs = (("output_0", node.args[0]),)
        else:
            outputs = ((kw.arg, kw.value) for kw in node.keywords)

        statements = []
        for name, value in outputs:
            set_output_call = ast.Call(
                func=ast.Name(id="set_output", ctx=ast.Load()),
                args=[ast.Constant(value=name), self.visit(value)],
                keywords=[],
            )
            statements.append(ast.Expr(value=set_output_call))

        return statements

    def visit_Expr(self, node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
        """Lower standalone ``csp.output(...)`` calls into output statements."""
        if isinstance(node.value, ast.Call) and self._is_csp_call(node.value, "output"):
            return self._transform_output(node.value)
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Union[ast.AST, List[ast.AST]]:
        """Lower ``return csp.output(...)`` while preserving the early return."""
        if isinstance(node.value, ast.Call) and self._is_csp_call(node.value, "output"):
            return [*self._transform_output(node.value), ast.Return(value=None)]
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        for method_name, transformer in self.csp_call_transformers.items():
            if self._is_csp_call(node, method_name):
                return transformer(node)

        if self._is_csp_call(node, "output"):
            raise ValueError("csp.output(...) must be used as a standalone statement or returned")

        # Recursively transform arguments
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]

        return node

    @staticmethod
    def _create_state_assignment(
        *,
        target: ast.expr,
        type_annotation: ast.AST,
        value: ast.AST,
        simple: int,
    ) -> ast.AnnAssign:
        """Create a state assignment with State[type] annotation."""
        state_type = ast.Subscript(value=ast.Name(id="State", ctx=ast.Load()), slice=type_annotation, ctx=ast.Load())
        return ast.AnnAssign(target=target, annotation=state_type, value=value, simple=simple)

    def _transform_state_statements(self, statements) -> List[ast.AST]:
        transformed = []

        for stmt in statements:
            if isinstance(stmt, ast.AnnAssign):
                # Already annotated: y: float = 1.0
                if not isinstance(stmt.target, ast.Name):
                    raise TypeError("numba_node state declarations require a simple variable name")
                transformed.append(
                    self._create_state_assignment(
                        target=stmt.target,
                        type_annotation=stmt.annotation,
                        value=stmt.value,
                        simple=stmt.simple,
                    )
                )

            elif isinstance(stmt, ast.Assign):
                # Unannotated: x = 0, need to infer type
                for target in stmt.targets:
                    if not isinstance(target, ast.Name):
                        raise TypeError("numba_node state assignments require a simple variable name")
                    inferred_type = self._infer_type(target.id, stmt.value)
                    transformed.append(
                        self._create_state_assignment(
                            target=target,
                            type_annotation=inferred_type,
                            value=stmt.value,
                            simple=1,
                        )
                    )

            elif isinstance(stmt, ast.Expr):
                transformed.extend(self._flatten_transformed_statements((stmt,)))

            elif not isinstance(stmt, ast.Pass):
                raise TypeError(f"numba_node state blocks do not support {type(stmt).__name__} statements")

        return transformed

    def _flatten_transformed_statements(self, statements: List[ast.AST]) -> List[ast.AST]:
        flattened = []
        for stmt in statements:
            transformed_stmt = self.visit(stmt)
            if isinstance(transformed_stmt, list):
                flattened.extend(transformed_stmt)
            else:
                flattened.append(transformed_stmt)
        return flattened

    def transform_csp_node(self, definition: "ParsedNodeDefinition") -> TransformedNode:
        start_body = self._flatten_transformed_statements(definition.blocks.start)
        stop_body = self._flatten_transformed_statements(definition.blocks.stop)
        transformed_body = self._transform_state_statements(definition.blocks.state)
        transformed_body.extend(self._flatten_transformed_statements(definition.blocks.body))
        func_def = definition.funcdef
        new_func_def = ast.FunctionDef(
            name=func_def.name,
            args=func_def.args,  # Keep original args with ts[type] annotations
            body=transformed_body if transformed_body else [ast.Pass()],
            decorator_list=[],  # Remove decorators
            returns=func_def.returns,  # Keep original return annotation
        )

        ast.fix_missing_locations(new_func_def)
        return TransformedNode(new_func_def, start_body, stop_body)
