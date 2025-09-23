import operator
from datetime import datetime

import numpy as np
import pandas as pd
import pandas._testing as pd_test
import pandas.tests.extension.base as base
import pytest
from pandas import Series
from pandas.tests.extension.conftest import *

import csp
from csp.impl.pandas_ext_type import TsArray, TsDtype
from csp.impl.wiring.edge import Edge

PANDAS_BASE_EXTENSION_TESTS = base.base.BaseExtensionTests if pd.__version__ < "2.1.0" else object


@pytest.fixture
def pytype():
    """This is the python type used for the tests."""
    # While we only test one type in prod, feel please test different types as part of development
    # Most tests (with the exception or arithmetic/comparison/reduce) should work on all types
    # Depending on the type, a subset of the arithmetic/comparison/reduce tests should still work (and this is a
    # good source of corner cases)
    return int


@pytest.fixture(autouse=True)
def edge_equality_mock(monkeypatch):
    """Because csp defines Edge equality to return another Edge, but the provided pandas tests rely on a standard
    equality test (returning a bool) to confirm that the right things are happening, we monkeypatch edge equality
    so that we can ree-use the extensive pandas testing logic for extension types."""

    def mock_eq(self, other):
        # This works for some tests, but not all (i.e. two different ways of arriving at the same logical Edge)
        # return id(self) == id(other)
        return self.run(starttime=datetime(2000, 1, 1), endtime=datetime(2000, 1, 1)) == other.run(
            starttime=datetime(2000, 1, 1), endtime=datetime(2000, 1, 1)
        )

    def mock_ne(self, other):
        return not mock_eq(self, other)

    monkeypatch.setattr(Edge, "__eq__", mock_eq)
    monkeypatch.setattr(Edge, "__ne__", mock_ne)


@pytest.fixture
def dtype(pytype):
    """A fixture providing the ExtensionDtype to validate."""
    return TsDtype(pytype)


@pytest.fixture
def data(pytype):
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return TsArray([csp.const(pytype(i + 1)) for i in range(100)])


@pytest.fixture
def data_missing(pytype):
    """Length-2 array with [NA, Valid]"""
    return TsArray([np.nan, csp.const(pytype(1))])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_grouping(dtype, pytype):
    """
    Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    a = csp.const(pytype(0))
    b = csp.const(pytype(1))
    c = csp.const(pytype(2))
    na = np.nan
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)


@pytest.fixture
def data_for_twos(pytype):
    """Length-100 array in which all the elements are two."""
    return TsArray([csp.const(pytype(2)) for _ in range(100)])


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return np.nan


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    # "__mod__",
    # "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(
    params=[
        # Note: we do not test eq and ne here, because we have monkeypatched eq and ne on edges so that the
        # pandas-provided unit tests can use equality comparisons on Edge types. Presumably, if the other operators
        # are implemented correctly, then eq and ne will be as well.
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
)
def comparison_op(request):
    """
    Fixture for operator module comparison functions.
    """
    return request.param


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestTsDtype(base.BaseDtypeTests):
    # Add additional test cases as there are multiple possible TsDtype sub-types
    def test_equality(self):
        assert TsDtype(int) == TsDtype(int)
        assert TsDtype(int) == "ts[int]"
        assert TsDtype(int) != TsDtype(float)
        assert TsDtype(int) != "ts[float]"

    def test_caching(self):
        class MyClass:
            pass

        my_class = MyClass
        typ = TsDtype(MyClass)
        typ2 = TsDtype(MyClass)
        assert typ2 is typ

        class MyClass:  # Redefine with same name in an attempt to fool caching
            pass

        assert MyClass is not my_class
        assert TsDtype(MyClass) != typ
        assert TsDtype(MyClass).subtype is MyClass


class TestCasting(base.BaseCastingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    def test_getitem_scalar(self, data):
        """Overwrite from parent because Edges are not actually of type TsType"""
        result = data[0]
        assert result.tstype == data.dtype.tstype

        result = pd.Series(data)[0]
        assert result.tstype == data.dtype.tstype


class TestGroupby(PANDAS_BASE_EXTENSION_TESTS):
    test_groupby_apply_identity = base.BaseGroupbyTests.test_groupby_apply_identity
    # test_in_numeric_groupby = base.BaseGroupbyTests.test_in_numeric_groupby

    def test_in_numeric_groupby(self, data_for_grouping):
        # This is not actually testing anything today, but simply documenting
        # the difference in behavior.

        # Groupby with extensions has different behavior in Pandas 1.5 and 2
        # In pandas 2 it will always call into our implementation of _reduce
        # In pandas 1.5, if the column's dtype is not labeled as numeric,
        # then it will just omit the column in the reduction (and never call
        # overridden version of _reduce).
        # Ultimately, our extension type should only probably only implement
        # this functionality for a Ts[T] if the aggregation makes sense for
        # the type T. Unfortunately, marking the type as numeric also has
        # broader implications on the types of operations that can be
        # performaned on it from pandas and appears to be breaking in other
        # ways.

        import pandas._testing as tm
        from pandas.core.dtypes.common import is_numeric_dtype

        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        dtype = data_for_grouping.dtype
        result = df.groupby("A").sum().columns

        if data_for_grouping.dtype._is_numeric:
            expected = pd.Index(["B", "C"])
        else:
            expected = pd.Index(["C"])

        # tm.assert_index_equal(result, expected)

    def test_groupby_agg_extension(self, data_for_grouping):
        # GH#38980 groupby agg on extension type fails for non-numeric types
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        expected = df.iloc[[0, 2, 4, 7]]
        expected = expected.set_index("A")

        result = df.groupby("A").agg({"B": "first"})
        pd_test.assert_frame_equal(result, expected)

        result = df.groupby("A").agg("first")
        pd_test.assert_frame_equal(result, expected)

        result = df.groupby("A").first()
        pd_test.assert_frame_equal(result, expected)

    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        df.groupby("A", group_keys=False).apply(groupby_apply_op)
        df.groupby("A", group_keys=False).B.apply(groupby_apply_op)


class TestInterface(base.BaseInterfaceTests):
    def test_contains(self, data, data_missing):
        # Overwrite from parent as it's not expected to pass with edges.
        pass

    def test_array_interface_copy(self, data):
        # Overwrite from parent as it's not expected to pass on numpy 2
        pass


class TestMethods(PANDAS_BASE_EXTENSION_TESTS):
    """Selected tests copied from base.BaseMethodsTests, because many of them are not expected to work."""

    test_count = base.BaseMethodsTests.test_count
    test_series_count = base.BaseMethodsTests.test_series_count
    test_apply_simple_series = base.BaseMethodsTests.test_apply_simple_series
    test_unique = base.BaseMethodsTests.test_unique
    test_fillna_copy_frame = base.BaseMethodsTests.test_fillna_copy_frame
    test_fillna_copy_series = base.BaseMethodsTests.test_fillna_copy_series
    test_fillna_length_mismatch = base.BaseMethodsTests.test_fillna_length_mismatch
    test_combine_first = base.BaseMethodsTests.test_combine_first
    test_container_shift = base.BaseMethodsTests.test_container_shift
    test_shift_0_periods = base.BaseMethodsTests.test_shift_0_periods
    test_shift_non_empty_array = base.BaseMethodsTests.test_shift_non_empty_array
    test_shift_empty_array = base.BaseMethodsTests.test_shift_empty_array
    test_shift_zero_copies = base.BaseMethodsTests.test_shift_zero_copies
    test_shift_fill_value = base.BaseMethodsTests.test_shift_fill_value
    test_not_hashable = base.BaseMethodsTests.test_not_hashable
    test_hash_pandas_object_works = base.BaseMethodsTests.test_hash_pandas_object_works
    test_where_series = base.BaseMethodsTests.test_where_series
    test_repeat = base.BaseMethodsTests.test_repeat
    test_repeat_raises = base.BaseMethodsTests.test_repeat_raises

    # These two value-count tests aren't used because the implementation relies on sorting by edge.
    # See the following tests below instead
    # test_value_counts = base.BaseMethodsTests.test_value_counts
    # test_value_counts_with_normalize = base.BaseMethodsTests.test_value_counts_with_normalize

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, data_for_grouping, dropna):
        value_counts = data_for_grouping.value_counts(dropna)
        # Recall value counts follows the pattern [B, B, NA, NA, A, A, B, C]
        assert len(value_counts) == 3 if dropna else 4
        assert value_counts[data_for_grouping[0]] == 3
        assert value_counts[data_for_grouping[4]] == 2
        assert value_counts[data_for_grouping[-1]] == 1

    def test_describe(self, data_for_grouping):
        out = pd.Series(data_for_grouping).describe()
        assert len(data_for_grouping) == 8
        assert out["count"] == 6
        assert out["unique"] == 3


class TestMissing(base.BaseMissingTests):  # Good
    pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None
    divmod_exc = TypeError

    def test_error(self, data, all_arithmetic_operators):
        """This is supposed to be overridden when things work. See https://github.com/pandas-dev/pandas/pull/39386"""
        pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    # The pandas test is incorrectly written in old versions, but fixed with this pull request:
    # https://github.com/pandas-dev/pandas/pull/44004
    # Below we paste the fix. This can all be removed once we use the pandas version that contains this change.
    def _compare_other(self, ser: pd.Series, data, op, other):
        if op.__name__ in ["eq", "ne"]:
            # comparison should match point-wise comparisons
            result = op(ser, other)
            expected = ser.combine(other, op)
            pd_test.assert_series_equal(result, expected)

        else:
            exc = None
            try:
                result = op(ser, other)
            except Exception as err:
                exc = err

            if exc is None:
                # Didn't error, then should match pointwise behavior
                expected = ser.combine(other, op)
                pd_test.assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, op)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0)

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)
        other = pd.Series([data[0]] * len(data))
        self._compare_other(ser, data, comparison_op, other)


class TestLogicalOps(base.ops.BaseOpsUtil):
    """Various Series and DataFrame logical ops methods."""

    # Note: This test doesn't exist in the pandas test suite, but should!

    def _compare_other(self, ser: pd.Series, data, op, other):
        result = op(ser, other)

        # Didn't error, then should match pointwise behavior
        expected = ser.combine(other, op)
        pd_test.assert_series_equal(result, expected)

    def test_logical_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0)

    def test_logical_array(self, data, comparison_op):
        ser = pd.Series(data)
        other = pd.Series([data[0]] * len(data))
        self._compare_other(ser, data, comparison_op, other)


class TestUnaryOps(base.BaseUnaryOpsTests):
    pass


class TestPrinting(base.BasePrintingTests):  # Good
    pass


def check_reduce(self, s, op_name, skipna=None):
    import pandas._testing as tm

    result = getattr(s, op_name)(skipna=skipna)
    result = result.run(starttime=datetime(2000, 1, 1), endtime=datetime(2000, 1, 1))[0][1]
    data = pd.Series([i.run(starttime=datetime(2000, 1, 1), endtime=datetime(2000, 1, 1))[0][1] for i in s])
    expected = getattr(data, op_name)(skipna=skipna)
    tm.assert_almost_equal(result, expected)


class TestBooleanReduce(base.BaseBooleanReduceTests):
    def check_reduce(self, s, op_name, skipna):
        return check_reduce(self, s, op_name, skipna)


class TestNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        return check_reduce(self, s, op_name, skipna)


class TestReshaping(base.BaseReshapingTests):
    def test_merge_on_extension_array(self, data):
        """ "Do not implement because merge behavior not well defined"""
        pass

    def test_merge_on_extension_array_duplicates(self, data):
        """ "Do not implement because merge behavior not well defined"""
        pass


class TestSetitem(base.BaseSetitemTests):
    def test_setitem_invalid(self):
        pass


class TestAsType(PANDAS_BASE_EXTENSION_TESTS):
    """This is not part of the pandas test suite, but needed to test the astype functionality."""

    def _check_astype(self, data, out, dtype, equality=True):
        assert out.dtype == dtype
        assert len(out) == len(data)
        assert out[0] == data[0]

    def test_as_object(self, data):
        out = data.astype(object)
        self._check_astype(data, out, np.dtype("O"))

    def test_as_ts_float(self, data):
        out = data.astype(TsDtype(float))
        self._check_astype(data, out, TsDtype(float))

    def test_as_ts_object(self, data):
        out = data.astype(TsDtype(object))
        self._check_astype(data, out, TsDtype(object))

    def test_as_string(self, data):
        out = data.astype(str)
        assert len(out) == len(data)
        assert out[0].startswith("Edge(")
