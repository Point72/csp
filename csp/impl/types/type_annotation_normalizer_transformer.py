import ast


class InvalidTypeAnnotationError(TypeError):
    def __init__(self, function_name, argument_name, annotation):
        super().__init__(f"Invalid type annotation for argument {argument_name} in {function_name} : {annotation} ")


class TypeAnnotationNormalizerTransformer(ast.NodeTransformer):
    """A utility class to normalize   AST annotation types to standard typing annotations:
    Example:
        def f(w: int, x: [int], y:{float}, z:{'T':int) ...
        will be converted to:
        def f(w: int, x : typing.List[int], y : typing.Set[float], z:typing.Dict[typing.TypeVar('T'), int])...
    """

    def __init__(self):
        self._cur_arg = None
        self._f = None

    def normalize_type_annotations(self, f: ast.FunctionDef):
        """Normalize the given function ast  and return the function ast with normalized type annotations.

        NOTE:  normalization is done in-place, the original AST is modifed.
        :param f:  Function ast
        """
        self._f = f
        self._cur_arg = None
        self.visit(f.args)
        self._f = None

    def normalize_single_type_annotation(self, node):
        """Normalize the given function ast  and return the function ast with normalized type annotations.

        NOTE:  normalization is done in-place, the original AST is modifed.
        :param f:  Function ast
        """
        self._cur_arg = node
        res = self.visit(node)
        ast.fix_missing_locations(res)
        return res

    def visit_arg(self, node):
        if node.annotation:
            self._cur_arg = node

            node.annotation = self.visit(node.annotation)

            self._cur_arg = None
            node = ast.fix_missing_locations(node)
        return node

    def visit_Subscript(self, node):
        # We choose to avoid parsing here
        # to maintain current behavior of allowing empty lists in our types
        return node

    def visit_List(self, node):
        if not self._cur_arg:
            return node
        if len(node.elts) != 1:
            raise InvalidTypeAnnotationError(self._f.name, self._cur_arg, node.elts)
        node = ast.Subscript(
            value=ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()), attr="List", ctx=ast.Load()),
            slice=self.visit(node.elts[0]),
            ctx=ast.Load(),
        )
        return node

    def visit_Set(self, node):
        if not self._cur_arg:
            return node

        if len(node.elts) != 1:
            raise InvalidTypeAnnotationError(self._f.name, node.arg, node.elts)
        node = ast.Subscript(
            value=ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()), attr="Set", ctx=ast.Load()),
            slice=self.visit(node.elts[0]),
            ctx=ast.Load(),
        )
        return node

    def visit_Dict(self, node):
        if not self._cur_arg:
            return node

        if len(node.keys) != 1:
            raise InvalidTypeAnnotationError(self._f.name, node.arg, node)
        node = ast.Subscript(
            value=ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()), attr="Dict", ctx=ast.Load()),
            slice=ast.Tuple(elts=[self.visit(node.keys[0]), self.visit(node.values[0])], ctx=ast.Load()),
            ctx=ast.Load(),
        )
        return node

    def visit_Call(self, node):
        return node

    def visit_Constant(self, node):
        if not self._cur_arg or not isinstance(node.value, str):
            return node
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()), attr="TypeVar", ctx=ast.Load()),
            args=[node],
            keywords=[],
        )
