import ast
import copy
import typing
from ast import Attribute, Call, Name

from csp.impl.types.common_definitions import ArgKind
from csp.impl.wiring.node_parser import CspParseError, NodeParser
from csp.impl.wiring.signature import Signature


class _StateVarTransformer(ast.NodeTransformer):
    """Transform all usages of vars that are stored in state to be used from the state

    Example:
        Assume that x, is a state variable, y is a function argument (function arguments treated as state variable since they should persist) and z is local variable.
        x = x + y + z will be transformed to __csp_state__.x = __csp_state__.x +  __csp_state__.y +  z
    """

    def __init__(self, state_obj_name):
        self._state_obj_name = state_obj_name
        self.state_vars = set()

    def visit_Name(self, node: Name):
        if node.id in self.state_vars:
            return Attribute(value=Name(id=self._state_obj_name, ctx=ast.Load()), attr=node.id, ctx=node.ctx)
        else:
            return node


class NumbaNodeParser(NodeParser):
    _STATE_ARG_NAME = "__csp_state_ptr__"
    _STATE_INST_NAME = "__csp_state_inst__"

    def __init__(self, name, raw_func, func_frame, debug_print=False, state_types=None):
        from csp.impl.wiring.numba_utils.csp_cpp_numba_interface import CSP_NUMBA_CPP_FUNCTIONS
        from csp.impl.wiring.numba_utils.csp_numba_functions import (
            csp_print,
            csp_unsafe_cast_class_ptr_to_int,
            csp_unsafe_cast_int_to_class_ptr,
        )

        super().__init__(name=name, raw_func=raw_func, func_frame=func_frame, debug_print=debug_print)
        self._func_globals_modified.update(CSP_NUMBA_CPP_FUNCTIONS)
        self._func_globals_modified["csp_unsafe_cast_int_to_class_ptr"] = csp_unsafe_cast_int_to_class_ptr
        self._func_globals_modified["csp_unsafe_cast_class_ptr_to_int"] = csp_unsafe_cast_class_ptr_to_int
        self._func_globals_modified["__csp__print__"] = csp_print

        self._state_types = {} if state_types is None else state_types
        self._state_vars_transformer = _StateVarTransformer(self._STATE_INST_NAME)
        self._start_state_block_body = []
        self._state_class_name = f"__CSP__STATE__{raw_func.__name__}__"
        for k in self._state_types.keys():
            self._state_vars_transformer.state_vars.add(k)

    def _parse_state(self, node):
        if isinstance(node, ast.With):
            assert len(node.items) == 1
            call = node.items[0].context_expr
            body = node.body
        else:
            call = node
            body = None

        for kwd in call.keywords:
            if kwd.arg not in self._state_types:
                if isinstance(kwd.value, ast.Constant) and isinstance(kwd.value.value, (str, bool, int, float)):
                    if isinstance(kwd.value.value, str):
                        self._state_types[kwd.arg] = str
                    elif isinstance(kwd.value.value, bool):
                        self._state_types[kwd.arg] = bool
                    else:
                        self._state_types[kwd.arg] = type(kwd.value.value)
                else:
                    raise CspParseError(
                        f"Unable to resolve type of state variable {kwd.arg}, explicit type must be provided in numba_node decorator",
                        node.lineno,
                    )
            self._start_state_block_body.append(
                ast.Assign(targets=[ast.Name(id=kwd.arg, ctx=ast.Store())], value=kwd.value)
            )
            self._state_vars_transformer.state_vars.add(kwd.arg)
        if body:
            self._start_state_block_body += [self.visit(n) for n in body]

    def visit_Return(self, node):
        if len(self._outputs) > 1:
            raise CspParseError("returning from node with multiple outputs, use return csp.output(...)", node.lineno)

        res = []
        if node.value is not None:
            return_value = self.visit(node.value)
            res.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id=self.get_ts_out_value_return_name(0), ctx=ast.Load()),
                        args=[ast.Name(id="__csp_node__", ctx=ast.Load()), ast.Constant(0), return_value],
                        keywords=[],
                    )
                )
            )

        res.append(ast.Return(value=None))
        for i in res:
            i.lineno = node.lineno
            i.col_offset = node.col_offset
        return res

    def visit_Name(self, node):
        if self._signature is None or type(node.ctx) is not ast.Load:
            return node
        inp = self._signature.input(node.id, allow_missing=True)
        if inp is None or inp.kind == ArgKind.SCALAR:
            return node

        ts_idx = inp.ts_idx
        if ts_idx is not None:
            return ast.Call(
                func=ast.Name(id=self.get_ts_input_value_getter_name(ts_idx), ctx=ast.Load()),
                args=[ast.Name(id="__csp_node__", ctx=ast.Load()), ast.Constant(ts_idx)],
                keywords=[],
            )
        else:
            return node

    def visit_Call(self, node):
        # Hack to workaround numba print bug
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            node.func.id = "__csp__print__"

        return super().visit_Call(node)

    @classmethod
    def get_ts_input_value_getter_name(cls, ts_idx):
        return f"__csp__value_getter_inp_{ts_idx}__"

    @classmethod
    def get_ts_out_value_return_name(cls, ts_idx):
        return f"__csp__value_return_out_{ts_idx}__"

    @classmethod
    def get_arg_type_var_name(cls, arg_name):
        return f"__csp__{arg_name}_type__"

    @classmethod
    def get_state_type_var_name(cls, arg_name):
        return f"__csp__state__{arg_name}_type__"

    def _create_single_input_builtin_call(self, arg, func):
        input = self._signature.input(arg.id)
        ts_idx = input.ts_idx
        return ast.Call(
            func=ast.Name(id=func, ctx=ast.Load()),
            args=[ast.Name(id="__csp_node__", ctx=ast.Load()), ast.Constant(ts_idx)],
            keywords=[],
        )

    def _create_single_input_ticked_expression(self, arg):
        return self._create_single_input_builtin_call(arg, "__csp_numba_node_ticked__")

    def _create_single_input_valid_expression(self, arg):
        return self._create_single_input_builtin_call(arg, "__csp_numba_node_valid__")

    @classmethod
    def _csp_single_input_builtin_function_transformer(cls, cpp_function_name):
        """A generic function that is used to transform multiple builtin functions
        :param node: Call node
        :param cpp_function_name: The name of the cpp function to call
        """

        def _transform_csp_call(self, node):
            if len(node.args) != 1 or node.keywords:
                raise CspParseError("invalid csp call csp.%s" % node.func.attr, node.lineno)

            return self._create_single_input_builtin_call(node.args[0], cpp_function_name)

        return _transform_csp_call

    def _create_state_class(
        self, state_body: typing.List[typing.Any], args: typing.List[str], state_vars: typing.List[str]
    ):
        l_ctx = ast.Load()

        new_args = []
        for arg in args:
            new_arg = copy.copy(arg)
            new_arg.annotation = None
            new_args.append(new_arg)

        cls_field_types = [
            ast.Tuple(
                elts=[ast.Constant(arg.arg), ast.Name(id=self.get_arg_type_var_name(arg.arg), ctx=l_ctx)], ctx=l_ctx
            )
            for arg in self.get_non_ts_args()
        ]
        cls_field_types += [
            ast.Tuple(
                elts=[ast.Constant(state_var), ast.Name(id=self.get_state_type_var_name(state_var), ctx=l_ctx)],
                ctx=l_ctx,
            )
            for state_var in state_vars
        ]
        res = ast.ClassDef(
            name=self._state_class_name,
            bases=[],
            keywords=[],
            body=[
                ast.FunctionDef(
                    name="__init__",
                    args=self._create_ast_args(
                        posonlyargs=[],
                        args=[ast.arg(arg="self", annotation=None)] + new_args,
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[],
                    ),
                    body=state_body,
                    decorator_list=[],
                    returns=None,
                ),
                ast.FunctionDef(
                    name="get_ptr",
                    args=self._create_ast_args(
                        posonlyargs=[],
                        args=[ast.arg(arg="self", annotation=None)],
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[],
                    ),
                    body=[
                        ast.Return(
                            value=ast.Call(
                                func=ast.Name(id="csp_unsafe_cast_class_ptr_to_int", ctx=l_ctx),
                                args=[ast.Name(id="self", ctx=l_ctx)],
                                keywords=[],
                            )
                        )
                    ],
                    decorator_list=[],
                    returns=None,
                ),
            ],
            decorator_list=[
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__csp_numba__", ctx=l_ctx), attr="jitclass", ctx=l_ctx),
                    args=[ast.List(elts=cls_field_types, ctx=l_ctx)],
                    keywords=[],
                )
            ],
        )
        res = ast.fix_missing_locations(res)
        return res

    def _transform_state_block(self):
        self._stateblock = [self.visit(node) for node in self._stateblock]

        if self.has_state():
            if not self._stateblock:
                state_init_body = [ast.Pass()]
            else:
                state_init_body = self._stateblock

            self._stateblock = [
                self._create_state_class(
                    state_body=state_init_body, args=self.get_non_ts_args(), state_vars=list(self._state_types.keys())
                )
            ]

    def has_state(self):
        return self._state_types or self._stateblock

    def _create_init_block(self):
        init_block = []
        init_block.append(ast.Import(names=[ast.alias(name="numba", asname="__csp_numba__")]))
        return init_block

    @classmethod
    def _create_function_def(cls, name, body, args=None, decorator_list=None):
        res = ast.FunctionDef(name=name, body=body, returns=None)
        if args is None:
            res.args = cls._create_ast_args(
                posonlyargs=[], args=[], kwonlyargs=[], defaults=[], vararg=None, kwarg=None, kw_defaults=[]
            )
        else:
            res.args = args
        if decorator_list is None:
            res.decorator_list = []
        else:
            res.decorator_list = decorator_list
        return res

    def _create_jit_func(self, func_name, inner_body):
        args = self._create_ast_args(
            posonlyargs=[],
            args=[ast.arg(arg="__csp_node__", annotation=None), ast.arg(arg=self._STATE_ARG_NAME, annotation=None)],
            kwonlyargs=[],
            defaults=[],
            vararg=None,
            kwarg=None,
            kw_defaults=[],
        )
        if self.has_state():
            body = [
                ast.Assign(
                    targets=[Name(id="__csp_state_inst__", ctx=ast.Store())],
                    value=Call(
                        func=Name(id="csp_unsafe_cast_int_to_class_ptr", ctx=ast.Load()),
                        args=[
                            Name(id="__csp_state_ptr__", ctx=ast.Load()),
                            Name(id=self._state_class_name, ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                )
            ] + inner_body
        else:
            body = inner_body

        decorator_list = [
            Call(
                func=Attribute(value=Name(id="__csp_numba__", ctx=ast.Load()), attr="cfunc", ctx=ast.Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Attribute(
                                value=Name(id="__csp_numba__", ctx=ast.Load()), attr="types", ctx=ast.Load()
                            ),
                            attr="void",
                            ctx=ast.Load(),
                        ),
                        args=[
                            Attribute(
                                value=Attribute(
                                    value=Name(id="__csp_numba__", ctx=ast.Load()), attr="types", ctx=ast.Load()
                                ),
                                attr="int64",
                                ctx=ast.Load(),
                            ),
                            Attribute(
                                value=Attribute(
                                    value=Name(id="__csp_numba__", ctx=ast.Load()), attr="types", ctx=ast.Load()
                                ),
                                attr="int64",
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    )
                ],
                keywords=[
                    ast.keyword(arg="cache", value=ast.Constant(value=False)),
                    ast.keyword(arg="debug", value=ast.Constant(value=False)),
                ],
            )
        ]
        inner_body_func = self._create_function_def(name=func_name, args=args, body=body, decorator_list=decorator_list)
        return inner_body_func

    def _create_wrapped_inner_body(self, inner_body):
        if self._start_state_block_body:
            state_block_body = [self._state_vars_transformer.visit(n) for n in self._start_state_block_body]
        else:
            state_block_body = [ast.Pass()]
        inner_body = [self._state_vars_transformer.visit(n) for n in inner_body]

        init_state_func = self._create_jit_func("__csp__init__state__", state_block_body)
        inner_body_func = self._create_jit_func("__csp__impl__", inner_body)
        return [init_state_func, inner_body_func]

    def _finalize_func_body(self, newbody):
        if self.has_state():
            state_cls = ast.Name(id=self._state_class_name, ctx=ast.Load())
        else:
            state_cls = ast.Constant(value=None)

        newbody.append(
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        state_cls,
                        ast.Name(id="__csp__init__state__", ctx=ast.Load()),
                        ast.Name(id="__csp__impl__", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            )
        )
        return newbody

    def _create_top_level_func_def(self, body):
        res = self._create_function_def(name=self._funcdef.name, body=body)
        res.args.args += self.get_non_ts_args()

        for inp in self._inputs:
            if inp.kind.is_single_ts():
                res.args.args.append(ast.arg(arg=self.get_ts_input_value_getter_name(inp.ts_idx), annotation=None))
        for out in self._outputs:
            if out.kind.is_single_ts():
                res.args.args.append(ast.arg(arg=self.get_ts_out_value_return_name(out.ts_idx), annotation=None))

        for arg in self.get_non_ts_args():
            res.args.args.append(ast.arg(arg=self.get_arg_type_var_name(arg.arg), annotation=None))

        for state_var in self._state_types.keys():
            res.args.args.append(ast.arg(arg=self.get_state_type_var_name(state_var), annotation=None))

        return res

    def _is_ts_args_removed_from_signature(self):
        return True

    def _parse_impl(self):
        self._inputs, input_defaults, self._outputs = self.parse_func_signature(self._funcdef)
        # Should have inputs and outputs at this point
        inner_block_start_idx = self._parse_special_blocks(self._funcdef.body)
        self._signature = Signature(self._name, self._inputs, self._outputs, input_defaults)
        for arg in self.get_non_ts_args():
            self._stateblock.append(
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=arg.arg, ctx=ast.Store())],
                    value=ast.Name(id=arg.arg, ctx=ast.Load()),
                )
            )
            self._state_vars_transformer.state_vars.add(arg.arg)
        # apply transform to stateblock after signature is ready for alarms and passive calls
        self._transform_state_block()

        init_block = self._create_init_block()

        # inner_body is the while loop.  Start off with a yield, will get called when something ticks
        inner_body = []
        for x in range(inner_block_start_idx, len(self._funcdef.body)):
            res = self.visit(self._funcdef.body[x])
            if isinstance(res, list):
                inner_body.extend(res)
            else:
                inner_body.append(res)

        wrapped_inner_block = self._create_wrapped_inner_body(inner_body)

        if isinstance(wrapped_inner_block, list):
            start_and_body = self._stateblock + wrapped_inner_block
        else:
            start_and_body = self._stateblock + [wrapped_inner_block]

        if self._stopblock:
            # For stop we wrap start and body in a try / finally ( not the init block, if that fails it's unrecoverable )
            start_and_body = [ast.Try(body=start_and_body, finalbody=self._stopblock, handlers=[], orelse=[])]

        if init_block is None:
            newbody = start_and_body
        else:
            newbody = init_block + start_and_body

        newbody = self._finalize_func_body(newbody)

        newfuncdef = self._create_top_level_func_def(newbody)

        if self._DEBUG_PARSE or self._debug_print:
            import astor

            print(astor.to_source(newfuncdef))

        newfuncdef = ast.fix_missing_locations(newfuncdef)
        ast.increment_lineno(newfuncdef, self._raw_func.__code__.co_firstlineno - 1)
        modast = self._create_ast_module([newfuncdef])

        comp = compile(modast, filename=self._func_filename, mode="exec")
        _locals = {}
        eval(comp, self._func_globals_modified, _locals)
        self._impl = _locals[newfuncdef.name]

    @classmethod
    def _init_internal_maps(cls):
        super()._init_internal_maps()
        cls.METHOD_MAP["csp.make_passive"] = cls._csp_single_input_builtin_function_transformer(
            "__csp_numba_node_make_passive__"
        )
        cls.METHOD_MAP["csp.make_active"] = cls._csp_single_input_builtin_function_transformer(
            "__csp_numba_node_make_active__"
        )


NumbaNodeParser._init_internal_maps()
