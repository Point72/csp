import inspect


class Edge:
    __slots__ = ["tstype", "nodedef", "output_idx", "basket_idx"]

    def __init__(self, tstype, nodedef, output_idx, basket_idx=-1):
        self.tstype = tstype
        self.nodedef = nodedef
        self.output_idx = output_idx
        self.basket_idx = basket_idx

    def __repr__(self):
        return f"Edge( tstype={self.tstype}, nodedef={self.nodedef}, output_idx={self.output_idx}, basket_idx={self.basket_idx} )"

    def __bool__(self):
        raise ValueError("boolean evaluation of an edge is not supported")

    def __wrap_binary_method(self, other, method):
        import csp

        if isinstance(other, Edge):
            return method(self, other)
        try:
            return method(self, csp.const(other))
        except TypeError:
            # Need to return NotImplemented so that python will try calling the reverse operator on the other type.
            return NotImplemented

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.sub)

    def __rsub__(self, other):
        import csp

        return csp.sub(csp.const(other), self)

    def __mul__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.divide)

    def __rtruediv__(self, other):
        import csp

        return csp.divide(csp.const(other), self)

    def __floordiv__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.floordiv)

    def __rfloordiv__(self, other):
        import csp

        return csp.floordiv(csp.const(other), self)

    def __pow__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.pow)

    def __rpow__(self, other):
        import csp

        return csp.pow(csp.const(other), self)

    def __mod__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.mod)

    def __rmod__(self, other):
        import csp

        return csp.mod(csp.const(other), self)

    def __gt__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.gt)

    def __ge__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.ge)

    def __lt__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.lt)

    def __le__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.le)

    def __eq__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.eq)

    def __ne__(self, other):
        import csp

        return self.__wrap_binary_method(other, csp.ne)

    def __invert__(self):
        import csp

        if self.tstype.typ is int:
            return csp.bitwise_not(self)
        raise TypeError(f"Cannot call invert with a ts[{self.tstype.typ.__name__}], not an integer type")

    def __pos__(self):
        import csp

        return csp.pos(self)

    def __neg__(self):
        import csp

        return csp.neg(self)

    def __abs__(self):
        import csp

        return csp.abs(self)

    def abs(self):
        import csp

        return csp.abs(self)

    # def __ceil__(self):
    # def __floor__(self):
    # def __round__(self):
    # def __trunc__(self):
    # def __lshift__(self):
    # def __rshift__(self):
    # def __pos__(self):
    # def __xor__(self):

    def ln(self):
        import csp

        return csp.ln(self)

    def log2(self):
        import csp

        return csp.log2(self)

    def log10(self):
        import csp

        return csp.log10(self)

    def exp(self):
        import csp

        return csp.exp(self)

    def sqrt(self):
        import csp

        return csp.sqrt(self)

    def erf(self):
        import csp

        return csp.erf(self)

    def __getattr__(self, key):
        from csp.impl.struct import Struct

        typ = super().__getattribute__("tstype").typ
        if inspect.isclass(typ) and issubclass(typ, Struct):
            import csp

            elemtype = typ.metadata(typed=True).get(key)
            if elemtype is None:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.tstype.typ.__name__, key))
            return csp.struct_field(self, key, elemtype)

        raise AttributeError("'Edge' object has no attribute '%s'" % (key))

    def apply(self, func, *args, **kwargs):
        """
        :param func: A scalar function that will be applied on each value of the Edge. If a different output type
            is returned, pass a tuple (f, typ), where typ is the output type of f.
        :param args: Positional arguments passed into func
        :param kwargs: Dictionary of keyword arguments passed into func
        :return: A time series that ticks on each tick of self. Each item in the result time series is the return value of f applied on x. All values should match
        the specified result_type
        """
        import csp.baselib

        if isinstance(func, tuple):
            func, typ = func
        else:
            typ = self.tstype.typ
        if args or kwargs:
            f = lambda x: func(x, *args, **kwargs)  # noqa: E731
        else:
            f = func
        return csp.baselib.apply(self, f, typ)

    def pipe(self, node, *args, **kwargs):
        """
        :param node: A graph node that will be applied to the Edge, which is passed into node as the first argument.
            Alternatively, a (node, edge_keyword) tuple where edge_keyword is a string indicating the keyword of node
            that expects the edge.
        :param args: Positional arguments passed into node
        :param kwargs: Dictionary of keyword arguments passed into node
        :return: The return type of node
        """
        if isinstance(node, tuple):
            node, kwarg = node
            kwargs[kwarg] = self
            return node(*args, **kwargs)
        else:
            return node(self, *args, **kwargs)

    def run(self, *args, **kwargs):
        """
        :param args: Positional arguments passed into csp.run
        :param kwargs: Dictionary of keyword arguments passed into csp.run
        :return: The output of csp.run(self, *args, **kwargs)
        """
        from csp.impl.wiring.runtime import run

        return run(self, *args, **kwargs)[0]
