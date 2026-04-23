import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union


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
        self.input_names: Set[str] = set()
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
            "output": self._transform_output,
        }
        self.special_block_transformers = {
            "state": self._transform_state_block,
            "start": lambda node: self.start_body.extend(self._flatten_transformed_statements(node.body)) or [],
            "stop": lambda node: self.stop_body.extend(self._flatten_transformed_statements(node.body)) or [],
            "alarms": lambda node: [],
        }

    def _is_special_block(self, node: ast.AST, block_name: str) -> bool:
        """Returns True if this is a block we need to handle"""
        if not isinstance(node, ast.With):
            return False

        if len(node.items) != 1:
            return False

        context_expr = node.items[0].context_expr

        if isinstance(context_expr, ast.Call):
            func = context_expr.func
            if isinstance(func, ast.Name) and func.id == block_name:
                return True
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "csp" and func.attr == block_name:
                    return True

        return False

    def _get_special_block_name(self, node: ast.AST) -> Optional[str]:
        for block_name in self.special_block_transformers:
            if self._is_special_block(node, block_name):
                return block_name
        return None

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

    def _transform_output(self, node: ast.Call) -> Union[ast.AST, List[ast.AST]]:
        """Transform csp.output(a=x, b=y, ...) -> set_output('a', x), set_output('b', y), ..."""
        statements = []

        # Handle keyword arguments: csp.output(a=x, b=y)
        for kw in node.keywords:
            set_output_call = ast.Call(
                func=ast.Name(id="set_output", ctx=ast.Load()),
                args=[ast.Constant(value=kw.arg), self.visit(kw.value)],
                keywords=[],
            )
            statements.append(ast.Expr(value=set_output_call))

        # Handle positional arguments: csp.output(x)
        for i, arg in enumerate(node.args):
            set_output_call = ast.Call(
                func=ast.Name(id="set_output", ctx=ast.Load()),
                args=[ast.Constant(value=i), self.visit(arg)],
                keywords=[],
            )
            statements.append(ast.Expr(value=set_output_call))

        if len(statements) == 1:
            return statements[0]
        return statements

    def visit_Call(self, node: ast.Call) -> ast.AST:
        for method_name, transformer in self.csp_call_transformers.items():
            if self._is_csp_call(node, method_name):
                return transformer(node)

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

    def _transform_state_block(self, node: ast.With) -> List[ast.AST]:
        """
        Transform state block variables to have State[type] annotations.

        with csp.state():
            x = 0
            y: float = 1.0

        becomes:
            x: State[int] = 0
            y: State[float] = 1.0
        """
        transformed = []

        for stmt in node.body:
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

    def _transform_body(self, body: List[ast.AST]) -> List[ast.AST]:
        transformed = []

        for stmt in body:
            block_name = self._get_special_block_name(stmt)
            if block_name is not None:
                transformed.extend(self.special_block_transformers[block_name](stmt))
                continue

            # Transform the statement
            transformed.extend(self._flatten_transformed_statements([stmt]))

        return transformed

    def _transform_func_def(self, func_def: ast.FunctionDef) -> TransformedNode:
        self.state_variables = []
        self.input_names = set()
        self.start_body = []
        self.stop_body = []

        # Collect input names (don't transform annotations)
        for arg in func_def.args.args:
            self.input_names.add(arg.arg)

        transformed_body = self._transform_body(func_def.body)
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
            transformed_source=transformed_source,
            start_body=self.start_body,
            stop_body=self.stop_body,
        )

    def transform_csp_node(self, func) -> TransformedNode:
        if callable(func):
            source = textwrap.dedent(inspect.getsource(func))
        else:
            source = textwrap.dedent(func)

        tree = ast.parse(source)
        func_def = tree.body[0]

        if not isinstance(func_def, ast.FunctionDef):
            raise ValueError("Expected a function definition")

        return self._transform_func_def(func_def)
