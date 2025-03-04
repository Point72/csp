import copy
import functools
import operator
import re
from pydoc import locate
from typing import Any, List, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_integer, is_list_like
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin, IntervalArray
from pandas.core.dtypes.dtypes import PandasExtensionDtype, register_extension_dtype
from pandas.core.indexers import check_array_indexer

import csp
from csp.impl.types.tstype import TsType, isTsType, ts
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.node import node

str_type = str
T = TypeVar("T")


@register_extension_dtype
class TsDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for TsType data (based on the code for IntervalDtype)

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, type
        The type of the TsType, i.e. "float" in ts[float]

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> TsDtype(float)
    ts[float]
    """

    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 999  # Not sure what to put here
    _metadata = ("subtype",)
    _match = re.compile(r"ts\[(?P<subtype>.+)\]")
    _cache_dtypes: dict = {}

    def __new__(cls, subtype=None):
        from pandas.core.dtypes.common import pandas_dtype

        if isinstance(subtype, TsDtype):
            return subtype
        elif isinstance(subtype, type):
            subtype = subtype
        elif isTsType(subtype):
            subtype = subtype.typ
        elif subtype is None:
            # we are called as an empty constructor
            # generally for pickle compat
            u = object.__new__(cls)
            u._subtype = None
            return u
        else:
            if isinstance(subtype, str):
                m = cls._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd["subtype"]
            # Turn string into type object
            subtype = locate(subtype)
            if subtype is None:
                try:
                    subtype = pandas_dtype(subtype)
                except TypeError as err:
                    raise TypeError("could not construct TsDtype") from err

        key = f"{subtype}+{id(subtype)}"
        try:
            return cls._cache_dtypes[key]
        except KeyError:
            u = object.__new__(cls)
            u._subtype = subtype
            cls._cache_dtypes[key] = u
            return u

    @classmethod
    def construct_array_type(cls) -> Type["IntervalArray"]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """

        return TsArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> "TsDtype":
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")

        if cls._match.search(string) is not None:
            return cls(string)

        msg = (
            f"Cannot construct a 'TsDtype' from '{string}'.\n\n"
            "Incorrectly formatted string passed to constructor. "
            "Valid formats include ts[typ] "
            "where typ is any type"
        )
        raise TypeError(msg)

    @property
    def type(self) -> type:
        return TsType

    @property
    def name(self) -> str_type:
        return f"ts[{self._subtype.__name__}]"

    @property
    def subtype(self) -> type:
        return self._subtype

    @property
    def tstype(self):
        return TsType[self.subtype]

    def __str__(self) -> str_type:
        return self.name

    @property
    def na_value(self) -> float:
        return np.nan

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return other.lower() == str(self).lower()

        return isinstance(other, TsDtype) and self.subtype == other.subtype

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __setstate__(self, state):
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._subtype = state["subtype"]

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if cls._match.search(dtype) is not None:
                try:
                    if cls.construct_from_string(dtype) is not None:
                        return True
                    else:
                        return False
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def _get_common_dtype(self, dtypes: List["TsDtype"]) -> Union["TsDtype", None]:
        if all(isinstance(x, TsDtype) for x in dtypes):
            subtypes = [cast("TsDtype", x).subtype for x in dtypes]
            if len(set(subtypes)) == 1:
                return self
        return None


# -----------------------------------------------------------------------------
# Extension Container
# -----------------------------------------------------------------------------


class _NumPyBackedExtensionArrayMixin(ExtensionArray, ExtensionScalarOpsMixin):
    # Code inspired by https://github.com/hgrecco/pint-pandas/blob/master/pint_pandas/pint_array.py
    # and https://github.com/ContinuumIO/cyberpandas/blob/master/cyberpandas/ip_array.py
    _data: np.ndarray

    @property
    def dtype(self) -> "TsDtype":
        """An instance of 'TsDtype'."""
        return self._dtype

    def __len__(self) -> int:
        """Length of this array
        Returns
        -------
        length : int
        """
        return len(self._data)

    def __getitem__(self, item) -> Any:
        """Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar or TsArray
        """
        if is_integer(item):
            return self._data[item]

        item = check_array_indexer(self, item)

        return self.__class__(self._data[item], self.dtype)

    def __setitem__(self, key, value):
        # need to not use `not value` on numpy arrays
        if isinstance(value, (list, tuple)) and (not value):
            # doing nothing here seems to be ok
            return

        key = check_array_indexer(self, key)
        if is_integer(key) and is_list_like(value):
            raise ValueError("Value length does not match key.")

        try:
            self._data[key] = value
        except IndexError as e:
            msg = "Mask is wrong length. {}".format(e)
            raise IndexError(msg)

    def isna(self) -> np.ndarray:
        """Return a Boolean NumPy array indicating if each value is missing.
        Returns
        -------
        missing : np.array
        """
        return ~np.array([isinstance(i, Edge) for i in self._data], dtype=bool)

    def take(self, indices, allow_fill=False, fill_value=None):
        """Take elements from an array.
        # type: (Sequence[int], bool, Optional[Any]) -> TsArray
        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.
        Returns
        -------
        TsArray
        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        Notes
        -----
        TsArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignemnt, with a `fill_value`.
        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        Examples
        --------
        """
        from pandas.core.algorithms import take

        data = self._data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indices, fill_value=fill_value, allow_fill=allow_fill)

        return type(self)(result, dtype=self.dtype)

    def copy(self, deep=False):
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        else:
            data = data.copy()

        return type(self)(data, dtype=self.dtype)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        dtype = to_concat[0].dtype  # Know there will be at least one argument
        return cls(np.concatenate([array.data for array in to_concat]), dtype=dtype)

    def tolist(self):
        return self._data.tolist()

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def data(self):
        return self._data

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.
        Every category will have an entry, even those with a count of 0.
        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaN.
        Returns
        -------
        counts : Series
        See Also
        --------
        Series.value_counts
        """

        from collections import Counter

        data = self._data
        if dropna:
            data = data[~self.isna()]

        return pd.Series(Counter(data.tolist()))


class TsArray(_NumPyBackedExtensionArrayMixin):
    """Holder for CSP Ts edges.
    It satisfies pandas' extension array interface, and so can be stored inside
    :class:`pandas.Series` and :class:`pandas.DataFrame`.
    """

    __array_priority__ = 1000
    _valuetype = Edge  # Actual value type for isinstance checks
    ndim = 1
    can_hold_na = True

    def __init__(self, values, dtype=None, copy=False):
        if copy:
            values = values.copy()

        if isinstance(values, TsArray):
            self._data = values._data
        else:
            self._data = np.atleast_1d(np.asarray(values, dtype="O"))

        if dtype is None:
            try:
                dtype = next(i.tstype.typ for i in self._data if isinstance(i, Edge))
            except StopIteration:
                raise ValueError("Either a csp.Edge must be provided or a dtype")

        self._dtype = TsDtype(dtype)

        # Validation
        mask = self.isna()
        if not all(i.tstype.typ == self._dtype.subtype for i in self._data[~mask]):
            raise ValueError("All edges not of type {}".format(self._dtype.type))
        # All other values should be set to self._dtype.na_value
        self._data[mask] = self._dtype.na_value

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.
        This is used in the default '__repr__'. The returned formatting
        function receives scalar Quantities.
        # type: (bool) -> Callable[[Any], Optional[str]]
        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).
        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """

        def formatting_function(x):
            if isinstance(x, self._valuetype):
                return "<{}>".format(x.nodedef.__class__.__name__)
            elif np.isnan(x):
                return str(pd.NA)
            else:
                # The below throws some other weird error if hit
                raise ValueError("Unexpected value for format: {}".format(x))

        return formatting_function

    def _reduce(self, name, skipna=True, **kwds):
        """
        Return a scalar result of performing the reduction operation.
        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values (both missing Edges and nan's inside of Edges)
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.
        Returns
        -------
        scalar
        Raises
        ------
        TypeError : subclass does not define reductions
        """
        import pandas.core.nanops as nanops

        functions = {
            "all": functools.partial(_reduce_bool, func=nanops.nanall),
            "any": functools.partial(_reduce_bool, func=nanops.nanany),
            "min": functools.partial(_reduce, func=nanops.nanmin),
            "max": functools.partial(_reduce, func=nanops.nanmax),
            "sum": functools.partial(_reduce, func=nanops.nansum),
            "prod": functools.partial(_reduce, func=nanops.nanprod),
            "mean": functools.partial(_reduce_float, func=nanops.nanmean),
            "median": functools.partial(_reduce_float, func=nanops.nanmedian),
            "std": functools.partial(_reduce_float, func=nanops.nanstd),
            "var": functools.partial(_reduce_float, func=nanops.nanvar),
            "sem": functools.partial(_reduce_float, func=nanops.nansem),
            "skew": functools.partial(_reduce_float, func=nanops.nanskew),
            "kurt": functools.partial(_reduce_float, func=nanops.nankurt),
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            quantity = self.dropna()._data
        else:
            quantity = self._data

        if len(quantity) == 0:
            return None

        return functions[name](quantity.tolist(), self.dtype.subtype, kwargs=dict(skipna=skipna))

    @classmethod
    def _create_comparison_method(cls, op):
        # Override from base class to make sure result dtype is correct
        return cls._create_method(op, coerce_to_dtype=True, result_dtype=TsDtype(bool))

    @classmethod
    def _create_logical_method(cls, op):
        # Override from base class to make sure result dtype is correct
        out = functools.partial(_binary_op, op=op)
        out.__name__ = op.__name__.rstrip("_")
        return cls._create_method(op, coerce_to_dtype=True, result_dtype=TsDtype(bool))

    def __invert__(self):
        """
        Element-wise inverse of this array.
        """
        data = [_unary_op(d, operator.invert) for d in self._data]
        return type(self)(data, dtype=self.dtype)

    def astype(self, dtype, copy=True):
        """Cast to a NumPy array with 'dtype'.
        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.
        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, str) and dtype.startswith("ts["):
            dtype = TsDtype(dtype)
        if isinstance(dtype, TsDtype):
            if dtype == self._dtype:
                if copy:
                    return self.copy()
                else:
                    return self
            else:
                values = np.array([_cast(edge, dtype.subtype) for edge in self._data])
                return TsArray(values, dtype)
        return super(TsArray, self).astype(dtype, copy=copy)


@node
def _cast(x: ts[object], typ: "T") -> ts["T"]:
    if csp.ticked(x):
        return np.array(x).astype(typ).item()


@node
def _binary_op(x: ts["T"], y: ts["T"], op: object) -> ts["T"]:
    if csp.ticked(x, y) and csp.valid(x, y):
        return op(x, y)


@node
def _unary_op(x: ts["T"], op: object) -> ts["T"]:
    if csp.ticked(x):
        return op(x)


@node
def _reduce(x: List[ts["T"]], typ: "T", func: object, args: object = (), kwargs: object = {}) -> ts["T"]:
    # The choice was made to only emit values if all basket elements are valid.
    # If one wanted to reduce only over valid elements, then in many cases you could pre-apply csp.default
    # with a nan/sentinal value, and then apply a function which ignores these values
    # (as many of the common reduce functions do).
    # However, if it was implemented over valid elements by default, it would be significantly more difficult to
    # get it to wait for everything to be valid.

    if csp.valid(x):
        data = np.fromiter(x.validvalues(), dtype=typ)
        if data.dtype.kind in ("S", "U"):
            # Because pd.core.nanops functions on strings throw "TypeError: cannot perform reduce with flexible type",
            # but work just find on object type...
            data = data.astype("O")

        out = func(data, *args, **kwargs)
        if isinstance(out, (np.int64, np.int32)):
            return int(out)
        return out


@node
def _reduce_float(x: List[ts["T"]], typ: "T", func: object, args: object = (), kwargs: object = {}) -> ts[float]:
    if csp.valid(x):
        data = np.fromiter(x.validvalues(), dtype=typ)
        return func(data, *args, **kwargs)


@node
def _reduce_bool(x: List[ts["T"]], typ: "T", func: object, args: object = (), kwargs: object = {}) -> ts[bool]:
    if csp.valid(x):
        data = np.fromiter(x.validvalues(), dtype=typ)
        return bool(func(data, *args, **kwargs))


def is_csp_type(arr_or_dtype) -> bool:
    """Check whether the provided array or dtype is of a TsDtype dtype."""
    t = getattr(arr_or_dtype, "dtype", arr_or_dtype)
    try:
        return isinstance(t, TsDtype) or issubclass(t, TsDtype)
    except Exception:
        return False


TsArray._add_arithmetic_ops()
TsArray._add_comparison_ops()
TsArray._add_logical_ops()
setattr(TsArray, "__pos__", TsArray._create_arithmetic_method(operator.pos))
setattr(TsArray, "__neg__", TsArray._create_arithmetic_method(operator.neg))
setattr(TsArray, "__abs__", TsArray._create_arithmetic_method(operator.abs))
