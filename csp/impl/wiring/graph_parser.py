import ast

from csp.impl.wiring import Signature
from csp.impl.wiring.base_parser import BaseParser, CspParseError, _pythonic_depr_warning
from csp.impl.wiring.special_output_names import UNNAMED_OUTPUT_NAME


class GraphParser(BaseParser):
    _DEBUG_PARSE = False

    def __init__(self, name, raw_func, func_frame, debug_print=False):
        super().__init__(
            name=name,
            raw_func=raw_func,
            func_frame=func_frame,
            debug_print=debug_print,
        )

    def visit_FunctionDef(self, node):
        # We don't want to modify internal functions/nodes that are defined within a graph
        return node

    def _add_special_outputs_to_return(self, res, special_outputs):
        for k, v in special_outputs.items():
            res.value.keywords.append(ast.keyword(arg=k, value=ast.Name(id=v, ctx=ast.Load())))

    def _wrap_returned_value_and_add_special_outputs(self, returned_value):
        res = ast.Return(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id="csp", ctx=ast.Load()), attr="impl", ctx=ast.Load()),
                        attr="wiring",
                        ctx=ast.Load(),
                    ),
                    attr="OutputsContainer",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg=UNNAMED_OUTPUT_NAME, value=returned_value)],
            )
        )
        self._add_special_outputs_to_return(res, self._get_special_output_name_mapping())
        return res

    def visit_Return(self, node):
        if len(self._outputs) == 0 and node.value is not None:
            raise CspParseError("returning from graph without any outputs defined", node.lineno)

        if len(self._outputs) and node.value is None:
            raise CspParseError("return does not return values with non empty outputs")

        if isinstance(node.value, ast.Call):
            if len(self._outputs) > 1:
                self._validate_output(node)

            parsed_return = self.visit_Call(node.value)
            if isinstance(parsed_return, ast.Call):
                return ast.Return(value=parsed_return, lineno=node.lineno, end_lineno=node.end_lineno)

            return parsed_return

        return node

    def _parse_single_output_definition(self, name, arg_type_node, ts_idx, typ=None):
        return self._parse_single_output_definition_with_shapes(
            name, arg_type_node, ts_idx, typ, enforce_shape_for_baskets=False
        )

    @_pythonic_depr_warning
    def _parse_return(self, node, special_outputs=None):
        if node.args or node.keywords:
            if not self._signature._outputs:
                raise CspParseError("csp return trying to return values for graph with no outputs", node.lineno)
        else:
            if self._signature._outputs:
                raise CspParseError("csp return does not return values with non empty outputs", node.lineno)
            else:
                return ast.Return()
        if len(node.args) > 1 or (len(node.args) and len(node.keywords)):
            raise CspParseError(
                "csp return expects to be called with (value) or (output = value, output2 = value2)", node.lineno
            )
        if len(node.args) == 1:
            if len(self._signature._outputs) > 1:
                raise CspParseError(
                    f"cannot return single unnamed arg in graph returning {len(self._signature._outputs)} outputs",
                    node.lineno,
                )
            elif len(self._outputs) == 0:
                raise CspParseError("returning from graph without any outputs defined", node.lineno)
            elif (
                len(self._signature._outputs) == 1 and self._signature._outputs[0].name is None
            ):  # graph only has one unnamed output:
                return ast.Return(value=node.args[0], lineno=node.lineno, end_lineno=node.end_lineno)
            else:
                node.keywords = [ast.keyword(arg=self._signature._outputs[0].name, value=node.args[0])]
                node.args.clear()

        if len(self._signature._outputs) == 1 and self._signature._outputs[0].name is None:
            raise CspParseError("Invalid return of single unnamed argument.", node.lineno)

        # Allow for passing through other node / graph args with **f()
        if len(node.keywords) == 1 and node.keywords[0].arg is None:
            if not isinstance(node.keywords[0].value, ast.Call):
                raise CspParseError("only unpacking of other csp.node or csp.graph calls are allowed", node.lineno)
            f_name = node.keywords[0].value.func.id
            f = self._func_frame.f_locals.get(f_name) or self._func_frame.f_globals.get(f_name)
            f_sig = getattr(f, "_signature", None)
            if f_sig is None:
                raise CspParseError("only unpacking of other csp.node or csp.graph calls are allowed", node.lineno)
            if set(f_sig._output_map.keys()) != set(self._signature._output_map.keys()):
                raise CspParseError(f"{f_name} outputs dont align with graph outputs", node.lineno)

            # ** unpacking expects a mapping type, so we need to return the ._dict guts of the OutputsContainer type
            node.keywords[0].value = ast.Attribute(
                value=node.keywords[0].value, attr="_dict", lineno=node.lineno, ctx=ast.Load()
            )
        elif len(node.keywords) != len(self._signature._outputs):
            raise CspParseError(
                f"returning {len(node.keywords)} values from graph when expected {len(self._signature._outputs)}",
                node.lineno,
            )

        # Multiple named arguments
        node.func = ast.Attribute(
            value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="csp", ctx=ast.Load()), attr="impl", lineno=node.lineno, ctx=ast.Load()
                ),
                attr="wiring",
                lineno=node.lineno,
                ctx=ast.Load(),
            ),
            attr="OutputsContainer",
            lineno=node.lineno,
            ctx=ast.Load(),
        )
        res = ast.Return(value=node)
        if special_outputs:
            self._add_special_outputs_to_return(res, special_outputs)
        return res

    def visit_Expr(self, node):
        res = self.generic_visit(node)
        if isinstance(res.value, ast.Return) or isinstance(res.value, ast.Assign):
            return res.value
        return res

    def visit_Call(self, node: ast.Call):
        if (isinstance(node.func, ast.Name) and node.func.id == "__return__") or BaseParser._is_csp_output_call(node):
            special_outputs = {}
            return self._parse_return(node, special_outputs)
        return self.generic_visit(node)

    def _is_ts_args_removed_from_signature(self):
        return False

    def _parse_impl(self):
        self._inputs, input_defaults, self._outputs = self.parse_func_signature(self._funcdef)
        # Should have inputs and outputs at this point
        self._signature = Signature(
            self._name, self._inputs, self._outputs, input_defaults, special_outputs=self._special_outputs
        )
        self.generic_visit(self._funcdef)

        newfuncdef = ast.FunctionDef(name=self._funcdef.name, body=self._funcdef.body, returns=None)
        newfuncdef.args = self._funcdef.args
        newfuncdef.decorator_list = []

        ast.fix_missing_locations(newfuncdef)
        ast.increment_lineno(newfuncdef, self._raw_func.__code__.co_firstlineno - 1)

        if self._DEBUG_PARSE or self._debug_print:
            import astor

            print(astor.to_source(newfuncdef))

        self._impl = self._compile_function(newfuncdef)
        self._postprocess_basket_outputs(newfuncdef, enforce_shape_for_baskets=False)
