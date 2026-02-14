import ast
import copy
import inspect
import textwrap
import typing
from abc import ABCMeta, abstractmethod
from warnings import warn

import csp
from csp.impl.types import tstype
from csp.impl.types.common_definitions import (
    ArgKind,
    BasketKind,
    InputDef,
    OutputBasket,
    OutputBasketContainer,
    OutputDef,
    Outputs,
    OutputTypeError,
)
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.type_annotation_normalizer_transformer import TypeAnnotationNormalizerTransformer
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.warnings import WARN_PYTHONIC

LEGACY_METHODS = {"__alarms__", "__state__", "__start__", "__stop__", "__outputs__", "__return__"}


class CspParseError(Exception):
    def __init__(self, msg, lineno=None, file=None, frame=None):
        self.msg = msg
        self.filename = file
        self.lineno = lineno
        self.frame = frame

    def __str__(self):
        return self.msg


def _pythonic_depr_warning(func):
    def wrapper(*args, **kwargs):
        if not WARN_PYTHONIC:
            return func(*args, **kwargs)
        # Check name being used
        if len(args) > 1:
            node = args[1]
        else:
            node = kwargs["node"]

        if isinstance(node, ast.With):
            fn = node.items[0].context_expr.func
            if isinstance(fn, ast.Name):
                name = fn.id  # function
            else:
                name = fn.attr  # attribute
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                # should never get here
                try:
                    raise CspParseError(f"Error parsing csp builtin: {ast.unparse(node)}")
                except AttributeError:
                    raise CspParseError("Error parsing csp builtin function")
        else:
            # should never get here either
            raise CspParseError("Unable to parse csp builtin function")

        if name in LEGACY_METHODS:
            if name == "__return__":
                warn(
                    "Calling __return__ is deprecated. Instead, call return csp.output(v1=va1, v2=val2 ...)",
                    DeprecationWarning,
                )
            else:
                warn(
                    "Calling __{0}__ is deprecated: instead, use a 'with csp.{0}()' context (see the csp docs)".format(
                        name[2:-2]
                    ),
                    DeprecationWarning,
                )
        return func(*args, **kwargs)

    return wrapper


class BaseParser(ast.NodeTransformer, metaclass=ABCMeta):
    _DEBUG_PARSE = False

    def __init__(self, name, raw_func, func_frame, debug_print=False):
        self._name = name
        self._outputs = []
        self._special_outputs = tuple()
        self._func_frame = func_frame
        self._func_filename = raw_func.__code__.co_filename
        self._func_lineno = raw_func.__code__.co_firstlineno
        self._debug_print = debug_print
        self._docstring = raw_func.__doc__
        self._raw_func = raw_func
        self._inputs = []
        self._signature = None
        self._type_annotation_normalizer = TypeAnnotationNormalizerTransformer()
        self._cur_node = None
        self._func_globals_modified = dict()
        self._func_globals_modified["typing"] = typing
        self._func_globals_modified["csp"] = csp
        self._func_globals_modified.update(self._func_frame.f_globals)

        source = textwrap.dedent(inspect.getsource(raw_func))
        body = ast.parse(source)
        self._funcdef = body.body[0]
        self._type_annotation_normalizer.normalize_type_annotations(self._funcdef)

    def _eval_expr(self, exp):
        return eval(
            compile(ast.Expression(body=exp), filename="<csp>", mode="eval"),
            self._func_globals_modified,
            self._func_frame.f_locals,
        )

    @classmethod
    def _resolve_input_type_kind(cls, typ):
        # TODO: handle alarms here?
        if tstype.isTsType(typ):
            return ArgKind.TS, None
        elif CspTypingUtils.is_generic_container(typ):
            origin = CspTypingUtils.get_origin(typ)
            if origin is typing.List:
                if tstype.isTsType(typ.__args__[0]):
                    return ArgKind.BASKET_TS, BasketKind.LIST
            elif origin is typing.Dict:
                if tstype.isTsType(typ.__args__[1]):
                    if tstype.isTsType(typ.__args__[0]):
                        return ArgKind.DYNAMIC_BASKET_TS, BasketKind.DYNAMIC_DICT
                    return ArgKind.BASKET_TS, BasketKind.DICT
        elif CspTypingUtils.is_union_type(typ):
            args = [arg for arg in typing.get_args(typ) if arg is not None and arg is not type(None)]
            args_ts_status = [
                argkind.is_any_ts() for argkind, _basket_kind in [cls._resolve_input_type_kind(arg) for arg in args]
            ]
            if all(args_ts_status):
                return ArgKind.TS, None
            elif not any(args_ts_status):
                return ArgKind.SCALAR, None
            raise ValueError(f"Cannot mix TS and non-TS types in a union {typ}")
        return ArgKind.SCALAR, None

    @staticmethod
    def _is_pass_or_ellipsis(node):
        return isinstance(node, ast.Pass) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value == ...
        )

    @staticmethod
    def _is_csp_output_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "csp"
            and node.func.attr == "output"
        )

    def _get_kind_from_arg_type(self, typ):
        if tstype.isTsType(typ):
            return ArgKind.TS
        elif tstype.isTsBasket(typ):
            if tstype.isTsDynamicBasket(typ):
                return ArgKind.DYNAMIC_BASKET_TS
            else:
                return ArgKind.BASKET_TS
        return None

    @classmethod
    def _validate_output(cls, node):
        if not BaseParser._is_csp_output_call(node.value) or (len(node.value.args) + len(node.value.keywords)) == 0:
            raise CspParseError(
                "Returning multiple outputs must use the following syntax: return csp.output(out1=val1, ...)",
                node.lineno,
            )

    def _parse_single_output_definition(self, name, arg_type_node, ts_idx, typ=None):
        if typ is None:
            arg_type_node = self._type_annotation_normalizer.normalize_single_type_annotation(arg_type_node)
            typ = self._eval_expr(arg_type_node)

        kind = self._get_kind_from_arg_type(typ)
        if not kind:
            raise CspParseError(f"outputs must be ts[] or basket types, got {typ}", arg_type_node.lineno)

        return OutputDef(name=name, typ=typ, kind=kind, ts_idx=ts_idx, shape=None)

    def _parse_single_output_definition_with_shapes(
        self, name, arg_type_node, ts_idx, typ=None, enforce_shape_for_baskets=True
    ):
        assert typ is None
        if (
            isinstance(arg_type_node, ast.Call)
            and isinstance(arg_type_node.func, ast.Attribute)
            and arg_type_node.func.attr in OutputBasketContainer.SHAPE_FUNCS
        ):
            value = self._eval_expr(
                self._type_annotation_normalizer.normalize_single_type_annotation(arg_type_node.func.value)
            )
            if tstype.isTsBasket(value):
                if arg_type_node.keywords or len(arg_type_node.args) != 1:
                    raise CspParseError(f"Invalid use of '{arg_type_node.func.attr}'", arg_type_node.lineno)
                if tstype.isTsDynamicBasket(value):
                    raise CspParseError("dynamic baskets cannot declare shape", arg_type_node.lineno)
                typ = OutputBasketContainer.SHAPE_FUNCS[arg_type_node.func.attr](
                    typ=value,
                    shape=arg_type_node.args[0],
                    lineno=arg_type_node.lineno,
                    end_lineno=arg_type_node.end_lineno,
                    col_offset=arg_type_node.col_offset,
                    end_col_offset=arg_type_node.end_col_offset,
                )
                kind = ArgKind.BASKET_TS

        if typ is None:
            typ = self._eval_expr(self._type_annotation_normalizer.normalize_single_type_annotation(arg_type_node))
            if tstype.isTsBasket(typ):
                if tstype.isTsDynamicBasket(typ):
                    return OutputDef(name=name, typ=typ, kind=ArgKind.DYNAMIC_BASKET_TS, ts_idx=ts_idx, shape=None)
                elif enforce_shape_for_baskets:
                    raise CspParseError("output baskets must define shape using with_shape or with_shape_of")
            return BaseParser._parse_single_output_definition(self, name, arg_type_node, ts_idx, typ=typ)
        else:
            assert kind is not None
            return OutputDef(name=name, typ=typ, kind=kind, ts_idx=ts_idx, shape=None)

    def _parse_output_def(self, node):
        if WARN_PYTHONIC:
            warn(
                "Declaring __outputs__ is deprecated; instead, use output annotation syntax (func -> Outputs(...)). Consult the csp docs.",
                DeprecationWarning,
            )

    def parse_outputs(self, node: ast.Call):
        if node.args and node.keywords:
            raise CspParseError("__outputs__ must all be named or be single output, cant be both", node.lineno)

        if len(node.args) > 1:
            raise CspParseError("__outputs__ single unnamed arg only", node.lineno)

        if node.keywords:
            return tuple(
                self._parse_single_output_definition(k.arg, k.value, idx) for idx, k in enumerate(node.keywords)
            )
        else:
            name = None
            arg_type = node.args[0]
            ts_idx = 0
            return (self._parse_single_output_definition(name, arg_type, ts_idx),)

    def _extract_outputs_from_body(self, body):
        outputs_call = None
        # last_doc_string_item = -1
        lineno = None
        for i, body_item in enumerate(body):
            lineno = body_item.lineno
            if (
                isinstance(body_item, ast.Expr)
                and isinstance(body_item.value, ast.Constant)
                and isinstance(body_item.value.value, str)
            ):
                # last_doc_string_item = i
                continue
            break

        if i < len(body):
            _, res = self._consume_special_block(
                body, i, "__outputs__", self._parse_output_def, allow_with_block=False, allow_flat_call=True
            )
            if res:
                outputs_call = body[i].value

        if outputs_call is not None:
            body.pop(i)
            if not body:
                raise CspParseError('Node body can not be empty add "pass" for an empty node', lineno)
            return self.parse_outputs(outputs_call)
        else:
            return tuple()

    def _extract_single_output_definition(self, name, typ, idx, basket_container=None, enforce_shape=False):
        kind = self._get_kind_from_arg_type(typ)

        if not kind:
            raise CspParseError(f"outputs must be ts[] or basket types, got {typ}")

        if tstype.isTsDynamicBasket(typ):
            raise NotImplementedError()

        if basket_container:
            # replace typ with parsed out basket container
            typ = basket_container

        return OutputDef(name=name, typ=typ, kind=kind, ts_idx=idx, shape=None)

    def _extract_outputs_from_return_annotation(self, returns, enforce_shape=False):
        """Extract output types from the return type annotation of the node or graph.
        NOTE: the node_parser will overload this function to set enforce_shape to true
        """
        # evaluate the returns statement
        ret_type = self._eval_expr(returns)

        # Handle -> None annotation: explicitly return empty tuple for no outputs
        if ret_type is None:
            return tuple()

        output_dictionary_type = ContainerTypeNormalizer.normalize_type(ret_type)

        if not (isinstance(output_dictionary_type, type) and issubclass(output_dictionary_type, Outputs)):
            # try to wrap in outputs
            try:
                output_dictionary_type = Outputs(output_dictionary_type)
            except OutputTypeError:
                # if that didn't work, output type was wrong
                raise CspParseError(f"Output type must subclass `csp.Outputs`, got {output_dictionary_type}")

        # prepare return values
        ret = []

        # annotations are here, extract them (and remove self reference)
        type_annotations = output_dictionary_type.__annotations__

        # pop with default as subclasses will not have annotations-inside-annotations
        type_annotations.pop("__annotations__", None)
        for idx, (name, typ) in enumerate(type_annotations.items()):
            basket_container = None
            is_dynamic_basket = False
            if isinstance(typ, type) and issubclass(typ, OutputBasket):
                # replace the basket holder with the underlying type
                basket = typ
                typ = basket.typ
                shape = basket.shape

                is_dynamic_basket = tstype.isTsDynamicBasket(typ)
                if enforce_shape and not shape and not is_dynamic_basket:
                    raise CspParseError("output baskets must define shape using with_shape or with_shape_of")

                elif shape:
                    # install the proper shaping
                    basket_container = OutputBasketContainer.SHAPE_FUNCS[basket.shape_func](
                        typ=typ,
                        shape=shape,
                        lineno=returns.lineno,
                        col_offset=returns.col_offset,
                        end_lineno=returns.end_lineno,
                        end_col_offset=returns.end_col_offset,
                    )

            if is_dynamic_basket:
                ret.append(OutputDef(name=name, typ=typ, kind=ArgKind.DYNAMIC_BASKET_TS, ts_idx=idx, shape=None))
            else:
                ret.append(
                    self._extract_single_output_definition(
                        name=name,
                        typ=ContainerTypeNormalizer.normalize_type(typ),
                        idx=idx,
                        basket_container=basket_container,
                        enforce_shape=enforce_shape,
                    )
                )
        return tuple(ret)

    def parse_func_signature(self, funcdef):
        args = funcdef.args

        if args.vararg or args.kwarg:
            raise CspParseError("*args and **kwargs arguments are not supported in csp nodes")

        if args.posonlyargs:
            raise CspParseError("position only arguments are not supported in csp nodes")

        inputs = []
        tsidx = 0
        for arg_idx, arg in enumerate(args.args):
            if arg.annotation is None:
                # Let's allow missing annotation on self, make it "object"
                # It's UGLY, but we didn't find a better way to do it.
                # We will allow only the FIRST argument with name self to not be annotated
                # self._raw_func.__qualname__ != self._raw_func.__name__ checks that the function is nested (within another function or a class)
                # self._raw_func.__qualname__.split('.')[-2] != '<locals>') checks that the nesting is in fact within a class and not in function
                # For example in the test we see qualified name of a method as 'TestParsing.test_method_parsing.<locals>.C.f' while
                # qualified name of a nested function is 'TestParsing.test_method_parsing.<locals>.f', so the parent scope of the nested function
                # seems to be <locals>
                if (
                    arg.arg in ("self", "cls")
                    and arg_idx == 0
                    and self._raw_func.__qualname__ != self._raw_func.__name__
                    and self._raw_func.__qualname__.split(".")[-2] != "<locals>"
                ):
                    typ = object
                else:
                    raise CspParseError("csp.node and csp.graph args must be type annotated")
            else:
                typ = self._eval_expr(arg.annotation)
            arg_kind, basket_kind = self._resolve_input_type_kind(typ)

            inputs.append(InputDef(arg.arg, typ, arg_kind, basket_kind, tsidx, arg_idx))

            if arg_kind.is_any_ts():
                tsidx += 1

        input_idx = len(inputs) - len(args.defaults)
        defaults = {}
        for d in args.defaults:
            defaults[inputs[input_idx].name] = self._eval_expr(d)
            input_idx += 1

        if funcdef.returns is not None:
            annotation_outputs = self._extract_outputs_from_return_annotation(funcdef.returns)
        else:
            annotation_outputs = tuple()

        body_outputs = self._extract_outputs_from_body(self._funcdef.body)

        # TODO eventually we should disallow body outputs
        if annotation_outputs and body_outputs:
            raise CspParseError(
                "csp.node and csp.graph outputs must be via return annotation or __outputs__ call, not both"
            )
        return inputs, defaults, annotation_outputs or body_outputs

    def visit(self, node):
        prev_node = self._cur_node
        self._cur_node = node
        new_node = super().visit(node)
        self._cur_node = prev_node
        return new_node

    def get_non_ts_args(self):
        return [self._funcdef.args.args[input.arg_idx] for input in self._inputs if input.kind == ArgKind.SCALAR]

    @classmethod
    def _create_ast_module(cls, body):
        return ast.Module(body, [])

    def _compile_function(self, newfuncdef):
        modast = self._create_ast_module([newfuncdef])
        comp = compile(modast, filename=self._func_filename, mode="exec")
        _globals = dict(self._func_globals_modified)
        _globals.update(**self._func_frame.f_locals)
        _locals = {}
        eval(comp, _globals, _locals)
        impl = _locals[newfuncdef.name]
        return impl

    def parse(self):
        try:
            self._parse_impl()
        except CspParseError as err:
            err.frame = self._func_frame
            err.filename = self._func_filename
            err.lineno = self._func_lineno + (1 if err.lineno is None else err.lineno - 1)
            raise err

    @abstractmethod
    def _parse_impl(self):
        raise NotImplementedError()

    @classmethod
    def _is_csp_special_func_call(cls, call: ast.Call, func_name_or_list):
        name_pred = (
            (lambda x: x == func_name_or_list)
            if isinstance(func_name_or_list, str)
            else (lambda x: x in func_name_or_list)
        )

        # Calls can take the form:
        #   csp.__state__()  /  __state__()
        #        or
        #  with __state__():  with csp.__state__():
        if isinstance(call, ast.Call) and (
            (isinstance(call.func, ast.Name) and name_pred(call.func.id))
            or (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "csp"
                and name_pred(call.func.attr)
            )
        ):
            return True
        # raise exception if trying to use __state__ or ...
        elif isinstance(call, ast.Name) and name_pred(call.id):
            raise CspParseError(f"{call.id} must be called, cannot use as a bare name")
        # ... csp.__state__  without invocation
        elif isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name) and name_pred(call.attr):
            raise CspParseError(f"csp.{call.attr} must be called, cannot use as a bare name")
        return False

    def _is_special_with_block(cls, node, func_name_or_list):
        if isinstance(node, ast.With):
            if len(node.items) == 1:
                expr = node.items[0].context_expr
                if cls._is_csp_special_func_call(expr, func_name_or_list):
                    return True
        return False

    def _is_special_with_or_call(cls, node, func_name_or_list):
        if cls._is_special_with_block(node, func_name_or_list) or (
            isinstance(node, ast.Expr) and cls._is_csp_special_func_call(node.value, func_name_or_list)
        ):
            return True
        return False

    def _consume_special_block(self, body, index, name, handler_function, allow_with_block, allow_flat_call):
        if len(body) <= index:
            return index, None
        node = body[index]
        if self._is_special_with_block(node, name):
            if not allow_with_block:
                raise CspParseError(f"{name} can not be used in a with statement")
            handler_function(node)
            return index + 1, name
        if isinstance(node, ast.Expr) and self._is_csp_special_func_call(node.value, name):
            if not allow_flat_call:
                raise CspParseError(f"{name} must be used in a with statement")
            handler_function(node.value)
            return index + 1, name
        return index, None

    @abstractmethod
    def _is_ts_args_removed_from_signature(self):
        raise NotImplementedError()

    def _postprocess_basket_outputs(self, main_func_signature, enforce_shape_for_baskets=True):
        if self._outputs:
            for output in self._outputs:
                if output.kind == ArgKind.BASKET_TS:
                    if enforce_shape_for_baskets or isinstance(output.typ, OutputBasketContainer):
                        name = output.name if output.name else "arg0"

                        args = copy.deepcopy(main_func_signature.args)
                        # For nodes the timeseries args are no longer part of the function signature while for graphs they're
                        # we need to handle this here
                        if not self._is_ts_args_removed_from_signature():
                            args.args = [
                                arg
                                for input, arg in zip(self._signature.raw_inputs(), args.args)
                                if input.kind.is_scalar()
                            ]
                        for input in self._signature.raw_inputs():
                            if input.kind == ArgKind.BASKET_TS:
                                args.args.append(ast.arg(arg=input.name))

                        if isinstance(output.typ.shape, ast.AST):
                            shape_func = ast.FunctionDef(
                                name=f"{name}_shape_eval",
                                args=args,
                                body=[ast.Return(value=output.typ.shape)],
                                decorator_list=[],
                                returns=None,
                                **output.typ.ast_kwargs,
                            )
                            ast.fix_missing_locations(shape_func)
                            output.typ.shape_func = self._compile_function(shape_func)
                        else:
                            output.typ.shape_func = lambda *args, s=output.typ.shape: s
