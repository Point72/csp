import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from csp.impl.wiring.node_parser import ParsedNodeDefinition


@dataclass
class StateVariable:
    """Represents a state variable with its name and inferred/declared type."""

    name: str
    type_annotation: Optional[ast.AST] = None
    initial_value: Optional[ast.AST] = None


@dataclass
class TransformedNode:
    name: str
    state_variables: List[StateVariable]
    transformed_body: List[ast.AST]
    original_ast: ast.FunctionDef
    transformed_source: str = ""
    start_body: List[ast.AST] = field(default_factory=list)  # Code from with csp.start():
    stop_body: List[ast.AST] = field(default_factory=list)  # Code from with csp.stop():
    transformed_ast: Optional[ast.FunctionDef] = None


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
        self.state_variables: List[StateVariable] = []
        self.start_body: List[ast.AST] = []
        self.stop_body: List[ast.AST] = []
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
        statements = []

        # Handle keyword arguments: csp.output(a=x, b=y)
        for kw in node.keywords:
            set_output_call = ast.Call(
                func=ast.Name(id="set_output", ctx=ast.Load()),
                args=[ast.Constant(value=kw.arg), self.visit(kw.value)],
                keywords=[],
            )
            statements.append(ast.Expr(value=set_output_call))

        # Handle positional arguments for unnamed outputs: csp.output(x)
        for i, arg in enumerate(node.args):
            set_output_call = ast.Call(
                func=ast.Name(id="set_output", ctx=ast.Load()),
                args=[ast.Constant(value=f"output_{i}"), self.visit(arg)],
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

    def _create_state_assignment(
        self,
        *,
        name: str,
        target: ast.expr,
        type_annotation: ast.AST,
        value: ast.AST,
        simple: int,
    ) -> ast.AnnAssign:
        """Create a state assignment with State[type] annotation."""
        state_type = ast.Subscript(value=ast.Name(id="State", ctx=ast.Load()), slice=type_annotation, ctx=ast.Load())
        self.state_variables.append(StateVariable(name=name, type_annotation=type_annotation, initial_value=value))
        return ast.AnnAssign(target=target, annotation=state_type, value=value, simple=simple)

    def _transform_state_statements(self, statements) -> List[ast.AST]:
        transformed = []

        for stmt in statements:
            if isinstance(stmt, ast.AnnAssign):
                # Already annotated: y: float = 1.0
                name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                if name:
                    transformed.append(
                        self._create_state_assignment(
                            name=name,
                            target=stmt.target,
                            type_annotation=stmt.annotation,
                            value=stmt.value,
                            simple=stmt.simple,
                        )
                    )

            elif isinstance(stmt, ast.Assign):
                # Unannotated: x = 0, need to infer type
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        inferred_type = self._infer_type(name, stmt.value)
                        transformed.append(
                            self._create_state_assignment(
                                name=name,
                                target=target,
                                type_annotation=inferred_type,
                                value=stmt.value,
                                simple=1,
                            )
                        )

            elif isinstance(stmt, ast.Expr):
                transformed.append(self.visit(stmt))

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

    def _reset(self):
        self.state_variables = []
        self.start_body = []
        self.stop_body = []

    def _build_transformed_node(self, func_def: ast.FunctionDef, transformed_body: List[ast.AST]) -> TransformedNode:
        new_func_def = ast.FunctionDef(
            name=func_def.name,
            args=func_def.args,  # Keep original args with ts[type] annotations
            body=transformed_body if transformed_body else [ast.Pass()],
            decorator_list=[],  # Remove decorators
            returns=func_def.returns,  # Keep original return annotation
        )

        ast.fix_missing_locations(new_func_def)
        transformed_source = ast.unparse(new_func_def)

        return TransformedNode(
            name=func_def.name,
            state_variables=self.state_variables,
            transformed_body=transformed_body,
            original_ast=func_def,
            transformed_ast=new_func_def,
            transformed_source=transformed_source,
            start_body=self.start_body,
            stop_body=self.stop_body,
        )

    def transform_csp_node(self, definition: "ParsedNodeDefinition") -> TransformedNode:
        self._reset()
        self.start_body = self._flatten_transformed_statements(definition.blocks.start)
        self.stop_body = self._flatten_transformed_statements(definition.blocks.stop)
        transformed_body = self._transform_state_statements(definition.blocks.state)
        transformed_body.extend(self._flatten_transformed_statements(definition.blocks.body))
        return self._build_transformed_node(definition.funcdef, transformed_body)
