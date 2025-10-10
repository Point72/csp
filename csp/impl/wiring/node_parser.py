import ast
import copy
import inspect
import textwrap
from warnings import warn

from csp.impl import builtin_functions
from csp.impl.__cspimpl import _cspimpl
from csp.impl.types import tstype
from csp.impl.types.common_definitions import ArgKind, InputDef, OutputBasketContainer
from csp.impl.warnings import WARN_PYTHONIC
from csp.impl.wiring import Signature
from csp.impl.wiring.ast_utils import ASTUtils
from csp.impl.wiring.base_parser import BaseParser, CspParseError, _pythonic_depr_warning


class _SingleProxyFuncArgResolver(object):
    class INVALID_VALUE:
        pass

    def __init__(self, func):
        source = textwrap.dedent(inspect.getsource(func))
        self._func_ast = ast.parse(source).body[0]
        self._arg_name_to_index = {arg.arg: i for (i, arg) in enumerate(self._func_ast.args.args[1:])}
        self._index_to_arg_name = {i: arg_name for (arg_name, i) in self._arg_name_to_index.items()}
        self._defaults = self._func_ast.args.defaults
        self._num_non_defaults = len(self._arg_name_to_index) - len(self._defaults)
        assert self._num_non_defaults >= 0, "At least the first argument - proxy must be non default"
        self._all_defaults = [self.INVALID_VALUE] * self._num_non_defaults + self._defaults
        self._func_filename = func.__code__.co_filename

    def __call__(self, proxy, node):
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise CspParseError(f"Passing arguments with * is unsupported for {self._func_ast.name}", node.lineno)

        for arg in node.keywords:
            if arg.arg is None:
                raise CspParseError(f"Passing arguments with ** is unsupported for {self._func_ast.name}", node.lineno)

        assert len(node.args) <= len(self._arg_name_to_index) + 1, (
            f"Invalid number of arguments provided for {self._func_ast.name}"
        )

        resolved_args = [self.INVALID_VALUE] * len(self._all_defaults)
        resolved_args[: len(node.args) - 1] = node.args[1:]

        for arg in node.keywords:
            arg_index = self._arg_name_to_index[arg.arg]
            if resolved_args[arg_index] is not self.INVALID_VALUE:
                raise CspParseError(
                    f"{self._func_ast.name}() got multiple values for argument '{arg.arg}'", node.lineno
                )
            resolved_args[arg_index] = arg.value

        for arg_index in range(len(resolved_args)):
            if resolved_args[arg_index] is self.INVALID_VALUE:
                if self._all_defaults[arg_index] is self.INVALID_VALUE:
                    arg_name = self._index_to_arg_name[arg_index]
                    raise CspParseError(
                        f"{self._func_ast.name}() missing 1 required positional argument '{arg_name}'", node.lineno
                    )
                else:
                    resolved_args[arg_index] = self._all_defaults[arg_index]

        return ast.Call(
            func=ast.Attribute(value=proxy, attr=self._func_ast.name, ctx=ast.Load()), args=resolved_args, keywords=[]
        )


class NodeParser(BaseParser):
    _CSP_NOW_FUNC = "_csp_now"
    _CSP_ENGINE_START_TIME_FUNC = "_engine_start_time"
    _CSP_ENGINE_END_TIME_FUNC = "_engine_end_time"
    _CSP_ENGINE_STATS_FUNC = "_csp_engine_stats"

    _CSP_STOP_ENGINE_FUNC = "_csp_stop_engine"
    _LOCAL_METHODS = {
        _CSP_NOW_FUNC: _cspimpl._csp_now,
        _CSP_ENGINE_START_TIME_FUNC: _cspimpl._csp_engine_start_time,
        _CSP_ENGINE_END_TIME_FUNC: _cspimpl._csp_engine_end_time,
        _CSP_STOP_ENGINE_FUNC: _cspimpl._csp_stop_engine,
        _CSP_ENGINE_STATS_FUNC: _cspimpl._csp_engine_stats,
    }

    _SPECIAL_BLOCKS_METH = {"alarms", "state", "start", "stop", "outputs"}
    _SPECIAL_BLOCKS_DEPR = {f"__{method}__" for method in _SPECIAL_BLOCKS_METH}
    _SPECIAL_BLOCKS_PYTHONIC = {f"{method}" for method in _SPECIAL_BLOCKS_METH if method != "outputs"}
    _SPECIAL_BLOCKS = _SPECIAL_BLOCKS_DEPR.union(_SPECIAL_BLOCKS_PYTHONIC)

    # These match up with const names defined in PyNode.cpp
    _NODE_P_VARNAME = "node_p"
    _INPUT_VAR_VARNAME = "input_var"
    _INPUT_PROXY_VARNAME = "input_proxy"
    _OUTPUT_PROXY_VARNAME = "output_proxy"

    def __init__(self, name, raw_func, func_frame, debug_print=False):
        super().__init__(
            name=name,
            raw_func=raw_func,
            func_frame=func_frame,
            debug_print=debug_print,
        )
        self._stateblock = []
        self._startblock = []
        self._stopblock = []
        self._func_globals_modified.update(builtin_functions.CSP_BUILTIN_CONTEXT_DICT)
        self._func_globals_modified.update(self._LOCAL_METHODS)
        self._gen = None

        # To catch returning from within a for or while loop, which wouldnt work as it seems
        self._inner_loop_count = 0
        self._returned_outputs = set()

    @_pythonic_depr_warning
    def _parse_alarms(self, node):
        if isinstance(node, ast.With):
            assert len(node.items) == 1
            call = node.items[0].context_expr
            body = node.body
        else:
            call = node
            body = None

        if call.args:
            raise CspParseError("__alarms__ does not accept positional arguments", call.lineno)

        num_alarms = 0

        # First, parse any keywords in the alarm call for backwards compat
        if WARN_PYTHONIC and len(call.keywords):
            warn(
                "Variable declarations within alarms() are deprecated. Instead, declare the variable in a with alarms() context.",
                DeprecationWarning,
            )
        for kwd in call.keywords:
            typ = self._eval_expr(kwd.value)
            if not tstype.isTsType(typ):
                raise CspParseError("__alarms__ alarms must be ts types", node.lineno)

            # We always place alarms as the first set of inputs
            self._inputs.insert(num_alarms, InputDef(kwd.arg, typ, ArgKind.ALARM, None, num_alarms, -1))
            num_alarms += 1

        # Next, parse any variables created in the body that are alarms
        # Anything annotated with a `ts` type is assumed to be an alarm, and the rhs must be
        # a call to `csp.alarm`.
        if body:
            # can't use enumerate as there might be pass / ellipsis / other cruft
            for node in body:
                if BaseParser._is_pass_or_ellipsis(node):
                    # ignore `pass` and `...`
                    continue

                if not (isinstance(node, ast.AnnAssign) or isinstance(node, ast.Assign)):
                    raise CspParseError("Only alarm assignments are allowed in csp.alarms block")

                name = node.target.id if isinstance(node, ast.AnnAssign) else None
                if isinstance(node, ast.Assign):
                    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                        raise CspParseError("Exactly one alarm can be assigned per line", node.lineno)
                    name = node.targets[0].id

                if not (isinstance(node.value, ast.Call) and BaseParser._is_csp_special_func_call(node.value, "alarm")):
                    raise CspParseError("Alarms must be initialized with csp.alarm in __alarms__ block", node.lineno)
                call_node = node.value

                # handle the initial scheduling via `csp.alarm`
                if len(call_node.keywords):
                    raise TypeError("function `csp.alarm` does not take keyword arguments")
                if len(call_node.args) != 1:
                    raise TypeError(
                        f"function `csp.alarm` requires a single type argument: {len(call_node.args)} arguments given"
                    )

                typ = self._eval_expr(call_node.args[0])
                ts_type_arg = tstype.TsType[typ]

                self._inputs.insert(num_alarms, InputDef(name, ts_type_arg, ArgKind.ALARM, None, num_alarms, -1))
                num_alarms += 1

        # Re-assign tsidx on inputs that have been pushed further out
        for idx in range(num_alarms, len(self._inputs)):
            if self._inputs[idx].ts_idx != -1:
                self._inputs[idx] = self._inputs[idx]._replace(ts_idx=self._inputs[idx].ts_idx + num_alarms)

    @_pythonic_depr_warning
    def _parse_state(self, node):
        if isinstance(node, ast.With):
            assert len(node.items) == 1
            call = node.items[0].context_expr
            body = node.body
        else:
            call = node
            body = None

        if call.args:
            raise CspParseError("__state__ does not accept positional arguments")

        # First, parse any keywords in the state block for backwards compat
        if WARN_PYTHONIC and len(call.keywords):
            warn(
                "Variable declarations within state() are deprecated. Instead, declare the variable in a with state() context.",
                DeprecationWarning,
            )
        for kwd in call.keywords:
            self._stateblock.append(ast.Assign(targets=[ast.Name(id=kwd.arg, ctx=ast.Store())], value=kwd.value))

        # Then just attach the state block's body to the _stateblock list, which will be
        if body:
            for syntax in body:
                if not isinstance(syntax, (ast.Pass, ast.Expr, ast.Assign, ast.AnnAssign)):
                    warn(
                        "Only variable assignments and declarations should be present in a csp.state block. Any logic should be moved to csp.start",
                        DeprecationWarning,
                    )
            self._stateblock.extend(body)

    @_pythonic_depr_warning
    def _parse_start(self, node):
        self._startblock = node.body

    @_pythonic_depr_warning
    def _parse_stop(self, node):
        assert isinstance(node, ast.With)
        self._stopblock = node.body

    def _parse_single_output_definition(self, name, arg_type_node, ts_idx, typ=None):
        return self._parse_single_output_definition_with_shapes(name, arg_type_node, ts_idx, typ)

    def _extract_outputs_from_return_annotation(self, returns, *args, **kwargs):
        return super()._extract_outputs_from_return_annotation(returns=returns, enforce_shape=True)

    def _node_proxy_expr(self, ctx=ast.Load()):
        return ast.Name(id="#node_p", ctx=ctx)

    def _ts_inproxy_expr(self, id, ctx=ast.Load()):
        if isinstance(id, ast.Name) or isinstance(id, int):
            input_def = (
                self._signature.ts_input_by_id(id)
                if isinstance(id, int)
                else self._signature.input(id.id, allow_missing=True)
            )
            if not input_def:
                raise CspParseError(f"unrecognized input '{id.id}'", self._cur_node.lineno)
            if input_def.kind == ArgKind.SCALAR:
                raise CspParseError(f"expected '{id.id}' to be a timeseries input", self._cur_node.lineno)

            if input_def.kind.is_basket():
                pname = input_def.name
            else:
                pname = "#inp_%d" % input_def.ts_idx
            return ast.Name(id=pname, ctx=ctx)
        elif isinstance(id, ast.Subscript):
            # Basket input proxy, ie x[ 'a' ]
            basket_proxy = self._ts_inproxy_expr(id.value)
            index = ASTUtils.get_subscript_index(id)
            return ast.Call(
                func=ast.Attribute(value=basket_proxy, attr="_proxy", ctx=ast.Load()), args=[index], keywords=[]
            )
        elif isinstance(id, ast.Attribute) and id.attr == "shape":
            input_def = self._signature.input(id.value.id, allow_missing=True)
            if not input_def:
                raise CspParseError(f"unrecognized input '{id.value.id}'", self._cur_node.lineno)
            if not input_def.kind.is_dynamic_basket():
                raise CspParseError(
                    f"invalid use of .shape on input '{id.value.id}', .shape property is only available on dynamic basket inputs",
                    self._cur_node.lineno,
                )

            # shape proxy on dynamic basket is exposed as ._shapeproxy property
            newid = copy.copy(id)
            newid.attr = "_shapeproxy"
            return newid

    def _ts_outproxy_expr(self, name):
        if not isinstance(name, int):
            output_def = self._signature.output(name, allow_missing=True)
            if not output_def:
                raise CspParseError(f"unrecognized output '{name}'", self._cur_node.lineno)
            name = output_def.ts_idx
        pname = "#outp_%d" % name

        return ast.Name(id=pname, ctx=ast.Load())

    def visit(self, node):
        prev_node = self._cur_node
        self._cur_node = node
        new_node = super().visit(node)
        self._cur_node = prev_node
        return new_node

    def visit_Expr(self, node: ast.Expr):
        res = self.generic_visit(node)
        if isinstance(res.value, list) or isinstance(res.value, ast.Expr):
            return res.value
        else:
            return res

    def _visit_node_or_list(self, node_or_list):
        if isinstance(node_or_list, list):
            res = []
            for node in node_or_list:
                cur_transformed_node = self.generic_visit(node)
                if isinstance(cur_transformed_node, list):
                    res.extend(cur_transformed_node)
                else:
                    res.append(cur_transformed_node)
            return res
        else:
            return self.generic_visit(node_or_list)

    def visit_Call(self, node):
        is_special_func_call = self._is_csp_special_func_call(node, self._SPECIAL_BLOCKS)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "csp"
        ):
            if is_special_func_call:
                raise CspParseError(
                    f"Invalid usage of {node.func.attr}, it should appear at the beginning of the function (consult documentation for details)"
                )

            convertor = self.METHOD_MAP.get("csp." + node.func.attr, None)
            if convertor is None:
                raise CspParseError("invalid csp call csp.%s" % node.func.attr, node.lineno)

            # Looks like visit wont carry through to newly created nodes, so we apply visitor on it ourselves
            return self._visit_node_or_list(convertor(self, node))
        elif isinstance(node.func, ast.Name):
            if is_special_func_call:
                raise CspParseError(
                    f"Invalid usage of {node.func.id}, it should appear at the beginning of the function (consult documentation for details)"
                )

            convertor = self.METHOD_MAP.get(node.func.id, None)
            if convertor is not None:
                return self._visit_node_or_list(convertor(self, node))

        return self.generic_visit(node)

    def visit_Return(self, node):
        """outputs are done via the + operator on output proxies"""
        if self._inner_loop_count > 0:
            raise CspParseError("returning from within a for or while loop is not supported", node.lineno)

        if len(self._outputs) == 0 and node.value is not None:
            raise CspParseError("returning from node without any outputs defined", node.lineno)

        if node.value is None:
            return ast.Continue()

        # Single return value
        if len(self._outputs) == 1:
            self._returned_outputs.add(None)
            out_proxy = ast.Name(id="#outp_0", ctx=ast.Load())

            # Allow single named outputs in csp.output
            if BaseParser._is_csp_output_call(node.value):
                return [ast.Expr(self.visit_Call(node.value)), ast.Continue()]

            # We add continue to avoid evaluating the rest of the body
            if node.value is not None:
                return [
                    ast.Expr(
                        ast.BinOp(left=out_proxy, op=ast.Add(), right=self.visit(node.value)),
                        lineno=node.lineno,
                        end_lineno=node.end_lineno,
                        col_offset=node.col_offset,
                        end_col_offset=node.end_col_offset,
                    ),
                    ast.Continue(),
                ]
            else:
                return ast.Continue()

        # Multiple return values: parse proper output return syntax: return csp.output(out1=val1, out2=val2, ...)
        self._validate_output(node)
        return [ast.Expr(self.visit_Call(node.value)), ast.Continue()]

    def visit_For(self, node):
        self._inner_loop_count += 1
        next = super().generic_visit(node)
        self._inner_loop_count -= 1
        return next

    def visit_While(self, node):
        self._inner_loop_count += 1
        next = super().generic_visit(node)
        self._inner_loop_count -= 1
        return next

    @_pythonic_depr_warning
    def _parse_output_or_return(self, node, is_return):
        func_name = "return" if is_return else "csp.output"
        if len(node.args) > 2 or (len(node.args) and len(node.keywords)):
            raise CspParseError(
                f"{func_name} expects to be called with (output, value) or (output = value, output2 = value2)",
                node.lineno,
            )
        nodes = []
        if len(node.args) == 1:
            if len(self._signature._outputs) > 1 and self._signature.output(0).name is not None:
                raise CspParseError(
                    f"cannot {func_name} single unnamed arg in node returning %d outputs"
                    % len(self._signature._outputs),
                    node.lineno,
                )
            elif len(self._outputs) == 0:
                raise CspParseError("returning from node without any outputs defined", node.lineno)
            else:  # node only has one output
                output = self._signature.output(0)
                if isinstance(
                    output.typ, OutputBasketContainer
                ):  # if it's a basket, the arg is a dict: csp.output({k1:v1, k2:v2})
                    if isinstance(node.args[0], ast.Dict):  # unnamed dict: convert to subscript
                        nodes.extend(
                            ast.BinOp(
                                left=ast.Subscript(
                                    value=self._ts_outproxy_expr(output.name),
                                    slice=ASTUtils.create_subscript_index(k),
                                    ctx=ast.Load(),
                                ),
                                op=ast.Add(),
                                right=v,
                            )
                            for k, v in zip(node.args[0].keys, node.args[0].values)
                        )
                    else:  # named dict: pass to PyBasketOutputProxy for handling
                        nodes.append(
                            ast.BinOp(left=self._ts_outproxy_expr(output.name), op=ast.Add(), right=node.args[0])
                        )
                else:  # not a basket, the arg should be output normally
                    nodes.append(ast.BinOp(left=self._ts_outproxy_expr(0), op=ast.Add(), right=node.args[0]))
                self._returned_outputs.add(None)
        if len(node.args) == 2:
            if isinstance(
                node.args[0], ast.Subscript
            ):  # if node.args[0] is an ast.Subscript then node.args[0].value.id must name an output basket: csp.output(x[k], v)
                if is_return:
                    raise CspParseError(
                        f"Invalid use of {func_name} basket element returns is not possible with return"
                    )
                output_name = node.args[0].value.id
                if not isinstance(self._signature.output(output_name).typ, OutputBasketContainer):
                    raise CspParseError("csp.output(x[k],v) syntax can only be used on basket outputs", node.lineno)
                nodes.append(
                    ast.BinOp(
                        left=ast.Subscript(
                            value=self._ts_outproxy_expr(output_name), slice=node.args[0].slice, ctx=ast.Load()
                        ),
                        op=ast.Add(),
                        right=node.args[1],
                    )
                )
                self._returned_outputs.add(output_name)
            else:
                if not isinstance(node.args[0], ast.Name):
                    raise CspParseError(f"Invalid use of {func_name} please consult documentation", node.lineno)
                nodes.append(ast.BinOp(left=self._ts_outproxy_expr(node.args[0].id), op=ast.Add(), right=node.args[1]))
                self._returned_outputs.add(node.args[0].id)

        for arg in node.keywords:
            if self._signature.output(arg.arg, True) is None:
                raise CspParseError(f"unrecognized output '{arg.arg}'", node.lineno)
            output = self._signature.output(arg.arg)
            if isinstance(output.typ, OutputBasketContainer) and isinstance(arg.value, ast.Dict):
                nodes.extend(
                    ast.BinOp(
                        left=ast.Subscript(
                            value=self._ts_outproxy_expr(arg.arg),
                            slice=ASTUtils.create_subscript_index(k),
                            ctx=ast.Load(),
                        ),
                        op=ast.Add(),
                        right=v,
                    )
                    for k, v in zip(arg.value.keys, arg.value.values)
                )
            else:
                nodes.append(ast.BinOp(left=self._ts_outproxy_expr(arg.arg), op=ast.Add(), right=arg.value))
            self._returned_outputs.add(arg.arg)

        if len(nodes) == 0:
            if not is_return:
                raise CspParseError("Empty output is not allowed", node.lineno)
            res = ast.Pass(lineno=node.lineno, end_lineno=node.end_lineno)
        else:
            res = (
                ast.BoolOp(op=ast.Or(), values=nodes, lineno=node.lineno, end_lineno=node.end_lineno)
                if len(nodes) > 1
                else nodes[0]
            )
        if is_return:
            if isinstance(res, ast.Pass):
                return [res, ast.Continue(lineno=node.lineno, end_lineno=node.end_lineno)]
            else:
                return [
                    ast.Expr(res, lineno=node.lineno, end_lineno=node.end_lineno),
                    ast.Continue(lineno=node.lineno, end_lineno=node.end_lineno),
                ]

        return res

    def _parse_output(self, node):
        return self._parse_output_or_return(node=node, is_return=False)

    def _parse_return(self, node):
        return self._parse_output_or_return(node=node, is_return=True)

    def _parse_remove_dynamic_key(self, node):
        if len(self._signature._outputs) == 0:
            raise CspParseError("invalid call to csp.remove_dynamic_key on node with no outputs")

        is_named_outputs = self._signature.output(0).name is not None
        expected_args = 2 if is_named_outputs else 1
        if len(node.args) != expected_args:
            raise CspParseError(
                f"csp.remove_dynamic_key expects {expected_args} arguments for {'' if is_named_outputs else 'un'}named outputs, got {len(node.args)}"
            )

        if len(node.args) == 2:
            name = node.args[0].id
            output_def = self._signature.output(name, allow_missing=True)
            if not output_def:
                raise CspParseError("unrecognized output '%s'" % name, node.lineno)

            key = node.args[1]
        else:
            name = None
            output_def = self._signature.output(0)
            key = node.args[0]

        if output_def.kind != ArgKind.DYNAMIC_BASKET_TS:
            raise CspParseError(
                "output %sis not a dynamic basket output" % f"{name} " if name is not None else "", node.lineno
            )

        # Convert csp.remove_dynamic_key call to outputting dict of { key : csp.impl.REMOVE_DYNAMIC_KEY }
        value = ast.Dict(
            keys=[key],
            values=[
                ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="csp", ctx=ast.Load()), attr="impl", ctx=ast.Load()),
                    attr="REMOVE_DYNAMIC_KEY",
                    ctx=ast.Load(),
                )
            ],
        )

        return ast.BinOp(left=self._ts_outproxy_expr(output_def.ts_idx), op=ast.Add(), right=value)

    def _create_single_input_ticked_expression(self, arg):
        return ast.UnaryOp(op=ast.UAdd(), operand=self._ts_inproxy_expr(arg))

    def _parse_ticked(self, node):
        """ORed together all values ticked converted to unary add +"""
        exprs = [self._create_single_input_ticked_expression(arg) for arg in node.args]
        if len(exprs) == 1:
            return exprs[0]

        op = ast.BinOp(left=exprs[0], op=ast.BitOr(), right=exprs[1])
        for expr in exprs[2:]:
            op = ast.BinOp(left=op, op=ast.BitOr(), right=expr)
        return op

    def _create_single_input_valid_expression(self, arg):
        return ast.UnaryOp(op=ast.USub(), operand=self._ts_inproxy_expr(arg))

    def _parse_valid(self, node):
        """ANDed together all values valid converted to unary sub -"""
        exprs = [self._create_single_input_valid_expression(arg) for arg in node.args]
        if len(exprs) == 1:
            return exprs[0]

        op = ast.BinOp(left=exprs[0], op=ast.BitAnd(), right=exprs[1])
        for expr in exprs[2:]:
            op = ast.BinOp(left=op, op=ast.BitAnd(), right=expr)
        return op

    def _parse_now(self, node):
        if len(node.args) or len(node.keywords):
            raise CspParseError("csp.now takes no arguments", node.lineno)
        return ast.Call(
            func=ast.Name(id=self._CSP_NOW_FUNC, ctx=ast.Load()), args=[self._node_proxy_expr()], keywords=[]
        )

    def _parse_stop_engine(self, node):
        args = [self._node_proxy_expr()] + node.args
        return ast.Call(func=ast.Name(id=self._CSP_STOP_ENGINE_FUNC, ctx=ast.Load()), args=args, keywords=node.keywords)

    def _parse_num_ticks(self, node):
        if len(node.args) != 1:
            raise CspParseError("csp.num_ticks expects single ts argument", node.lineno)

        proxy = self._ts_inproxy_expr(node.args[0])
        return ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[proxy], keywords=[])

    def _parse_schedule_alarm_func(self, node, funcname, name_node=None):
        name_node = name_node or node.args[0]

        if not isinstance(name_node, ast.Name):
            raise CspParseError(f"csp.{funcname} expects alarm name as first argument", node.lineno)

        name = name_node.id

        input_def = self._signature.input(name, allow_missing=True)
        if not input_def:
            raise CspParseError(f"unrecognized alarm '{name}'", node.lineno)

        if input_def.kind != ArgKind.ALARM:
            raise CspParseError(f"cannot {funcname.replace('_', ' ')} on non-alarm input '{name}'", node.lineno)

        proxy = self._ts_inproxy_expr(node.args[0])
        return ast.Call(
            func=ast.Attribute(value=proxy, attr=funcname, ctx=ast.Load()), args=node.args[1:], keywords=node.keywords
        )

    def _parse_alarm(self, node, name_node):
        return self._parse_schedule_alarm_func(node, "schedule_alarm", name_node)

    def _parse_schedule_alarm(self, node):
        return self._parse_schedule_alarm_func(node, "schedule_alarm")

    def _parse_reschedule_alarm(self, node):
        return self._parse_schedule_alarm_func(node, "reschedule_alarm")

    def _parse_cancel_alarm(self, node):
        return self._parse_schedule_alarm_func(node, "cancel_alarm")

    def _parse_set_buffering_policy(self, node):
        proxy = self._ts_inproxy_expr(node.args[0])
        args = list(node.args[1:])
        # FIXME
        # kwargs = {kwd.arg: kwd.value for kwd in node.keywords}
        return ast.Call(
            func=ast.Attribute(value=proxy, attr="set_buffering_policy", ctx=ast.Load()),
            args=args,
            keywords=node.keywords,
        )

    def _parse_engine_start_time(self, node):
        if len(node.args) or len(node.keywords):
            raise CspParseError("csp.engine_start_time takes no arguments", node.lineno)
        return ast.Call(
            func=ast.Name(id=self._CSP_ENGINE_START_TIME_FUNC, ctx=ast.Load()),
            args=[self._node_proxy_expr()],
            keywords=[],
        )

    def _parse_engine_end_time(self, node):
        if len(node.args) or len(node.keywords):
            raise CspParseError("csp.engine_end_time takes no arguments", node.lineno)
        return ast.Call(
            func=ast.Name(id=self._CSP_ENGINE_END_TIME_FUNC, ctx=ast.Load()),
            args=[self._node_proxy_expr()],
            keywords=[],
        )

    def _parse_csp_engine_stats(self, node):
        if len(node.args) or len(node.keywords):
            raise CspParseError("csp.engine_stats takes no arguments", node.lineno)

        return ast.Call(
            func=ast.Name(id=self._CSP_ENGINE_STATS_FUNC, ctx=ast.Load()), args=[self._node_proxy_expr()], keywords=[]
        )

    def _validate_return_statements(self):
        """Ensure outputs in spec return stmts line up"""
        if len(self._outputs) and not len(self._returned_outputs):
            raise CspParseError("node has __outputs__ defined but no return or csp.output statements")

        for output in self._outputs:
            if output.name not in self._returned_outputs:
                raise CspParseError(f"output '{output.name}' is never returned")

    def _parse_special_blocks(self, body):
        # skip doc string
        for index, node in enumerate(body):
            if not isinstance(node, ast.Expr) or not (
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            ):
                break

        last_special_block = None

        def consume_block(*args, **kwargs):
            cur_index, cur_block_name = self._consume_special_block(*args, **kwargs)
            if cur_block_name:
                return cur_index, cur_block_name
            return cur_index, last_special_block

        # Ensure blocks are processed in proper order i.e. start before stop
        index, last_special_block = consume_block(
            body, index, "alarms", self._parse_alarms, allow_with_block=True, allow_flat_call=False
        )
        index, last_special_block = consume_block(
            body, index, "__alarms__", self._parse_alarms, allow_with_block=True, allow_flat_call=True
        )

        index, last_special_block = consume_block(
            body, index, "state", self._parse_state, allow_with_block=True, allow_flat_call=False
        )
        index, last_special_block = consume_block(
            body, index, "__state__", self._parse_state, allow_with_block=True, allow_flat_call=True
        )

        index, last_special_block = consume_block(
            body, index, "start", self._parse_start, allow_with_block=True, allow_flat_call=False
        )
        index, last_special_block = consume_block(
            body, index, "__start__", self._parse_start, allow_with_block=True, allow_flat_call=False
        )

        index, last_special_block = consume_block(
            body, index, "stop", self._parse_stop, allow_with_block=True, allow_flat_call=False
        )
        index, last_special_block = consume_block(
            body, index, "__stop__", self._parse_stop, allow_with_block=True, allow_flat_call=False
        )

        if index < len(body):
            first_body_elem = body[index]
            if self._is_special_with_or_call(first_body_elem, self._SPECIAL_BLOCKS):
                for special_block_name in self._SPECIAL_BLOCKS:
                    if self._is_special_with_or_call(first_body_elem, special_block_name):
                        # For sure we must have previous block, it's a matter of out of order. Otherwise we can't get there
                        raise CspParseError(
                            f"{special_block_name} must be declared before {last_special_block}",
                            lineno=first_body_elem.lineno,
                            file=self._func_filename,
                        )
                # We should never get here
                raise AssertionError("Special block error wasn't handled correctly")
        return index

    @classmethod
    def _create_ast_args(
        cls, posonlyargs=[], args=[], kwonlyargs=[], defaults=[], vararg=None, kwarg=None, kw_defaults=[]
    ):
        return ast.arguments(
            posonlyargs=posonlyargs,
            args=args,
            kwonlyargs=kwonlyargs,
            defaults=defaults,
            vararg=vararg,
            kwarg=kwarg,
            kw_defaults=kw_defaults,
        )

    def _is_ts_args_removed_from_signature(self):
        return True

    def _parse_impl(self):
        self._inputs, input_defaults, self._outputs = self.parse_func_signature(self._funcdef)
        idx = self._parse_special_blocks(self._funcdef.body)
        self._signature = Signature(
            self._name,
            self._inputs,
            self._outputs + self._special_outputs,
            input_defaults,
            special_outputs=self._special_outputs,
        )
        # Up front initialize timeseries inputs as local vars.  Regular TS need two vars, the proxy and
        # the python value instance. Baskets will only need the proxy.  We also provide a noderef for node
        # specific methods.  Values are assigned to tuple( name, index ) which is replaced in PyNode.cpp init
        node_proxy = [
            ast.Assign(
                targets=[self._node_proxy_expr(ast.Store())],
                value=ast.Tuple(elts=[ast.Constant(self._NODE_P_VARNAME), ast.Constant(0)], ctx=ast.Load()),
            )
        ]  # '#nodep'
        ts_in_proxies = []
        ts_out_proxies = []
        ts_vars = []

        for inp in self._inputs:
            if inp.kind.is_any_ts():
                proxy = self._ts_inproxy_expr(inp.ts_idx, ast.Store())  # '#inp_%d' % inp.ts_idx
                ts_in_proxies.append(
                    ast.Assign(
                        targets=[proxy],
                        value=ast.Tuple(
                            elts=[ast.Constant(self._INPUT_PROXY_VARNAME), ast.Constant(inp.ts_idx)], ctx=ast.Load()
                        ),
                    )
                )
                if not inp.kind.is_basket():
                    ts_vars.append(
                        ast.Assign(
                            targets=[ast.Name(id=inp.name, ctx=ast.Store())],
                            value=ast.Tuple(
                                elts=[ast.Constant(self._INPUT_VAR_VARNAME), ast.Constant(inp.ts_idx)], ctx=ast.Load()
                            ),
                        )
                    )

        for output in self._signature._outputs:
            name = f"#outp_{output.ts_idx}"
            # ts_out_proxies.append( ast.Expr( value=ast.Name( id = name, ctx = ast.Load() ) ) )
            ts_out_proxies.append(
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Tuple(
                        elts=[ast.Constant(self._OUTPUT_PROXY_VARNAME), ast.Constant(output.ts_idx)], ctx=ast.Load()
                    ),
                )
            )

        # innerbody is the while loop.  Start off with a yield, will get called when something ticks
        innerbody = [ast.Expr(value=ast.Yield(value=None))]

        for x in range(idx, len(self._funcdef.body)):
            func_body_transformed = self.visit(self._funcdef.body[x])
            if isinstance(func_body_transformed, list):
                innerbody.extend(func_body_transformed)
            else:
                innerbody.append(func_body_transformed)

        self._validate_return_statements()

        # apply transform to stateblock and startblock after signature is ready for alarms and passive calls
        self._stateblock = [self.visit(node) for node in self._stateblock]
        self._startblock = [self.visit(node) for node in self._startblock]

        init_block = node_proxy + ts_in_proxies + ts_out_proxies + ts_vars
        startblock = self._stateblock + self._startblock
        body = [ast.While(test=ast.Constant(value=True), orelse=[], body=innerbody)]

        if self._stopblock:
            self._stopblock = [self.visit(node) for node in self._stopblock]

            # For stop we wrap the body of a node in a try / finally
            # If the init block fails it's unrecoverable, and if the start block raises we don't want to stop that specific node
            start_and_body = startblock + [ast.Try(body=body, finalbody=self._stopblock, handlers=[], orelse=[])]

        else:
            start_and_body = startblock + body

        # delete ts_var variables *after* start so that they raise Unbound local exceptions if they get accessed before first tick
        del_vars = []
        for v in ts_vars:
            del_vars.append(ast.Delete(targets=[ast.Name(id=v.targets[0].id, ctx=ast.Del())]))
        # Yield before start block so we can setup stack frame before executing
        # However, this initial yield shouldn't be within the try-finally block, since if a node does not start, it's stop() logic should not be invoked
        # This avoids an issue where one node raises an exception upon start(), and then other nodes execute their stop() without having ever started
        start_and_body = [ast.Expr(value=ast.Yield(value=None))] + del_vars + start_and_body
        newbody = init_block + start_and_body

        newfuncdef = ast.FunctionDef(name=self._name, body=newbody, returns=None)
        newfuncdef.args = self._create_ast_args(
            posonlyargs=[], args=[], kwonlyargs=[], defaults=[], vararg=None, kwarg=None, kw_defaults=[]
        )
        newfuncdef.decorator_list = []

        # Scalars become the only args to the generator
        newfuncdef.args.args = self.get_non_ts_args()

        if self._DEBUG_PARSE or self._debug_print:
            import astor

            print(astor.to_source(newfuncdef))

        ast.fix_missing_locations(newfuncdef)
        ast.increment_lineno(newfuncdef, self._raw_func.__code__.co_firstlineno - 1)
        self._impl = self._compile_function(newfuncdef)
        self._postprocess_basket_outputs(newfuncdef)

    @classmethod
    def _make_single_proxy_arg_func_resolver(cls, func):
        resolver = _SingleProxyFuncArgResolver(func)

        def f(node_parser, node):
            if not len(node.args):
                raise CspParseError(f"{func.__name__} expects a timeseries as first positional argument", node.lineno)
            return resolver(node_parser._ts_inproxy_expr(node.args[0]), node)

        return f

    @classmethod
    def _init_internal_maps(cls):
        cls.METHOD_MAP = {
            "csp.now": cls._parse_now,
            "csp.stop_engine": cls._parse_stop_engine,
            "csp.ticked": cls._parse_ticked,
            "csp.valid": cls._parse_valid,
            "csp.make_passive": cls._make_single_proxy_arg_func_resolver(builtin_functions.make_passive),
            "csp.make_active": cls._make_single_proxy_arg_func_resolver(builtin_functions.make_active),
            "csp.remove_dynamic_key": cls._parse_remove_dynamic_key,
            # omit this as its handled in a special case
            # 'csp.alarm': cls._parse_alarm,
            "csp.schedule_alarm": cls._parse_schedule_alarm,
            "csp.reschedule_alarm": cls._parse_reschedule_alarm,
            "csp.cancel_alarm": cls._parse_cancel_alarm,
            "csp.output": cls._parse_output,
            "__return__": cls._parse_return,
            "csp.__return__": cls._parse_return,
            "csp.num_ticks": cls._parse_num_ticks,
            "csp.value_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.value_at),
            "csp.time_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.time_at),
            "csp.item_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.item_at),
            "csp.values_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.values_at),
            "csp.times_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.times_at),
            "csp.items_at": cls._make_single_proxy_arg_func_resolver(builtin_functions.items_at),
            "csp.set_buffering_policy": cls._parse_set_buffering_policy,
            "csp.engine_start_time": cls._parse_engine_start_time,
            "csp.engine_end_time": cls._parse_engine_end_time,
            "csp.engine_stats": cls._parse_csp_engine_stats,
        }


NodeParser._init_internal_maps()
