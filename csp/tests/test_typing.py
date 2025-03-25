import json
import math
import unittest

import numpy as np
from pydantic import TypeAdapter, ValidationError

from csp import Enum, Struct
from csp.typing import Numpy1DArray, NumpyNDArray


class TestNumPyArrayTypes(unittest.TestCase):
    def test_Numpy1DArray(self):
        ta = TypeAdapter(Numpy1DArray[float])

        # Test basic validation with different dtypes
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))
        ta.validate_python([])
        self.assertRaises(ValidationError, ta.validate_python, 12)  # This gets turned into a scalar array
        self.assertRaises(ValidationError, ta.validate_python, np.array([[1.0]]))  # 2D array should fail
        self.assertRaises(ValidationError, ta.validate_python, np.array([[1.0, 2.0]]))  # 2D array should fail
        self.assertRaises(ValidationError, ta.validate_python, np.array(["foo"]))  # string array should fail

        # Test type coercion
        result = ta.validate_python(np.array([1, 2]))  # integer array gets coerced to float
        self.assertEqual(result.dtype, np.float64)

        result = ta.validate_python(np.array([1.0, 2.0], dtype=np.float32))  # float32 gets coerced to float64
        self.assertEqual(result.dtype, np.float64)

        # Test with NaN and Inf values
        special_arr = ta.validate_python(np.array([1.0, np.nan, np.inf]))
        self.assertTrue(np.isnan(special_arr[1]))
        self.assertTrue(np.isinf(special_arr[2]))

    def test_Numpy1DArray_json_serialization(self):
        ta = TypeAdapter(Numpy1DArray[float])

        # Test serialization to JSON
        arr = np.array([1.0, 2.0, 3.0])
        json_data = ta.dump_json(arr)
        self.assertEqual(json.loads(json_data), [1.0, 2.0, 3.0])

        # Test round-trip serialization
        arr_roundtrip = ta.validate_json(json_data)
        np.testing.assert_array_equal(arr, arr_roundtrip)

        # Test with empty array
        empty_arr = np.array([], dtype=np.float64)
        empty_json = ta.dump_json(empty_arr)
        self.assertEqual(json.loads(empty_json), [])
        empty_roundtrip = ta.validate_json(empty_json)
        self.assertEqual(len(empty_roundtrip), 0)

        # Test with special values (NaN and Inf become null in JSON)
        special_arr = np.array([1.0, np.nan, np.inf])
        special_json = ta.dump_json(special_arr)
        json_loaded = json.loads(special_json)
        self.assertEqual(json_loaded[0], 1.0)
        self.assertIsNone(json_loaded[1])  # NaN becomes null
        self.assertIsNone(json_loaded[2])  # Inf becomes null

    def test_NumpyNDArray(self):
        ta = TypeAdapter(NumpyNDArray[float])

        # Test 1D arrays (which are valid n-dimensional arrays)
        ta.validate_python(np.array([1.0, 2.0]))
        ta.validate_python(np.array([1.0, 2.0], dtype=np.float64))

        # Test 2D arrays
        ta.validate_python(np.array([[1.0, 2.0]]))
        ta.validate_python(np.array([[1.0, 2.0]], dtype=np.float64))

        # Test 3D arrays
        arr_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        result = ta.validate_python(arr_3d)
        self.assertEqual(result.shape, (1, 2, 2))

        # Test invalid arrays
        self.assertRaises(ValidationError, ta.validate_python, np.array(["foo"]))

        # Test type coercion
        result = ta.validate_python(np.array([1, 2]))
        self.assertEqual(result.dtype, np.float64)

        result = ta.validate_python(np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(result.dtype, np.float64)

        result = ta.validate_python(12)  # this generates a scalar of 0-dimension!
        self.assertEqual(result, np.array(12, dtype=float))

    def test_NumpyNDArray_json_serialization(self):
        ta = TypeAdapter(NumpyNDArray[float])

        # Test 1D array serialization
        arr_1d = np.array([1.0, 2.0, 3.0])
        json_1d = ta.dump_json(arr_1d)
        self.assertEqual(json.loads(json_1d), [1.0, 2.0, 3.0])
        arr_1d_roundtrip = ta.validate_json(json_1d)
        np.testing.assert_array_equal(arr_1d, arr_1d_roundtrip)

        # Test 2D array serialization
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        json_2d = ta.dump_json(arr_2d)
        self.assertEqual(json.loads(json_2d), [[1.0, 2.0], [3.0, 4.0]])
        arr_2d_roundtrip = ta.validate_json(json_2d)
        np.testing.assert_array_equal(arr_2d, arr_2d_roundtrip)

        # Test 3D array serialization
        arr_3d = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        json_3d = ta.dump_json(arr_3d)
        expected_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        self.assertEqual(json.loads(json_3d), expected_3d)
        arr_3d_roundtrip = ta.validate_json(json_3d)
        np.testing.assert_array_equal(arr_3d, arr_3d_roundtrip)

        # Test empty array serialization
        empty_arr = np.array([[], []], dtype=np.float64)
        empty_json = ta.dump_json(empty_arr)
        self.assertEqual(json.loads(empty_json), [[], []])
        empty_arr_roundtrip = ta.validate_json(empty_json)
        self.assertEqual(empty_arr_roundtrip.shape, (2, 0))
        self.assertEqual(empty_arr_roundtrip.dtype, np.float64)

    def test_NumpyNDArray_with_Enum(self):
        class MyEnum(Enum):
            A = Enum.auto()
            B = Enum.auto()

        ta = TypeAdapter(NumpyNDArray[MyEnum])

        # Test validation
        arr_enum = np.array([MyEnum.A, MyEnum.B])
        result = ta.validate_python(arr_enum)
        self.assertEqual(result[0], MyEnum.A)
        self.assertEqual(result[1], MyEnum.B)

        # Test JSON serialization (enums are serialized by name)
        json_data = ta.dump_json(arr_enum)
        self.assertEqual(json.loads(json_data), ["A", "B"])

        # Test round-trip serialization
        arr_roundtrip = ta.validate_json(json_data)
        self.assertEqual(arr_roundtrip[0], MyEnum.A)
        self.assertEqual(arr_roundtrip[1], MyEnum.B)

        # Test 2D array with enums
        arr_2d = np.array([[MyEnum.A], [MyEnum.B]])
        json_2d = ta.dump_json(arr_2d)
        self.assertEqual(json.loads(json_2d), [["A"], ["B"]])
        arr_2d_roundtrip = ta.validate_json(json_2d)
        self.assertEqual(arr_2d_roundtrip[0][0], MyEnum.A)
        self.assertEqual(arr_2d_roundtrip[1][0], MyEnum.B)

        from_raw_values = ta.validate_python(np.array([0, 1]))
        self.assertEqual(from_raw_values[0], MyEnum.A)
        self.assertEqual(from_raw_values[1], MyEnum.B)

    def test_integration_with_struct(self):
        class ArrayStruct(Struct):
            arr_1d: Numpy1DArray[float]
            arr_nd: NumpyNDArray[float]

        # Test creating and serializing a struct with numpy arrays
        test_struct = ArrayStruct(arr_1d=np.array([1.0, 2.0, 3.0]), arr_nd=np.array([[1.0, 2.0], [3.0, 4.0]]))

        struct_ta = TypeAdapter(ArrayStruct)
        json_data = struct_ta.dump_json(test_struct)
        loaded_json = json.loads(json_data)

        self.assertEqual(loaded_json["arr_1d"], [1.0, 2.0, 3.0])
        self.assertEqual(loaded_json["arr_nd"], [[1.0, 2.0], [3.0, 4.0]])

        # Test round-trip
        test_struct_roundtrip = struct_ta.validate_json(json_data)
        np.testing.assert_array_equal(test_struct.arr_1d, test_struct_roundtrip.arr_1d)
        np.testing.assert_array_equal(test_struct.arr_nd, test_struct_roundtrip.arr_nd)

        ta_array_structs = TypeAdapter(Numpy1DArray[ArrayStruct])
        array_structs = np.array([test_struct, test_struct])
        json_array_structs = ta_array_structs.dump_json(array_structs)
        loaded_array_structs = ta_array_structs.validate_json(json_array_structs)
        self.assertEqual(len(loaded_array_structs), 2)
        np.testing.assert_array_equal(loaded_array_structs[0].arr_1d, test_struct.arr_1d)
        np.testing.assert_array_equal(loaded_array_structs[0].arr_nd, test_struct.arr_nd)

    def test_Numpy1DArray_edge_cases(self):
        """Test edge cases for Numpy1DArray validation."""
        ta = TypeAdapter(Numpy1DArray[float])

        # Empty arrays
        self.assertEqual(len(ta.validate_python(np.array([], dtype=np.float64))), 0)
        self.assertEqual(len(ta.validate_python([])), 0)

        # Special values (NaN, Inf, -Inf)
        special_values = np.array([np.nan, np.inf, -np.inf, 1.0])
        validated_special = ta.validate_python(special_values)
        self.assertTrue(np.isnan(validated_special[0]))
        self.assertTrue(np.isinf(validated_special[1]))
        self.assertTrue(np.isinf(validated_special[2]) and validated_special[2] < 0)

        # 0-dimensional arrays (scalars) - should fail for 1D
        scalar_array = np.array(5.0)
        with self.assertRaises(ValidationError):
            ta.validate_python(scalar_array)

        # 3-dimensional arrays - should fail for 1D
        array_3d = np.ones((2, 2, 2))
        with self.assertRaises(ValidationError):
            ta.validate_python(array_3d)

        # Single element array - should work
        single_element = np.array([42.0])
        validated_single = ta.validate_python(single_element)
        self.assertEqual(len(validated_single), 1)
        self.assertEqual(validated_single[0], 42.0)

    def test_Numpy1DArray_comprehensive_type_validation(self):
        """Test validation with different types for Numpy1DArray."""
        # Test with different specified types
        ta_int = TypeAdapter(Numpy1DArray[int])
        ta_float = TypeAdapter(Numpy1DArray[float])
        ta_str = TypeAdapter(Numpy1DArray[str])
        ta_bool = TypeAdapter(Numpy1DArray[bool])
        ta_complex = TypeAdapter(Numpy1DArray[complex])

        # Test int arrays
        int_array = np.array([1, 2, 3])
        validated_int = ta_int.validate_python(int_array)
        self.assertTrue(np.issubdtype(validated_int.dtype, np.integer))

        # Int arrays should convert to float
        float_from_int = ta_float.validate_python(int_array)
        self.assertTrue(np.issubdtype(float_from_int.dtype, np.floating))

        float_array = np.array([1.5, 2.5, 3.6])
        # We rely on numpy typing, so we get this unexpected result
        self.assertEqual(ta_int.validate_python(float_array).tolist(), [1, 2, 3])

        # String arrays
        str_array = np.array(["a", "b", "c"])
        validated_str = ta_str.validate_python(str_array)
        self.assertEqual(validated_str.dtype.kind, "U")

        # Bool arrays
        bool_array = np.array([True, False, True])
        validated_bool = ta_bool.validate_python(bool_array)
        self.assertTrue(np.issubdtype(validated_bool.dtype, np.bool_))

        # Complex arrays
        complex_array = np.array([1 + 2j, 3 + 4j])
        validated_complex = ta_complex.validate_python(complex_array)
        self.assertTrue(np.issubdtype(validated_complex.dtype, np.complexfloating))

        # Mixed type arrays should fail appropriate validation
        mixed_array = np.array([1, "a", 3.0], dtype=object)
        with self.assertRaises(ValidationError):
            ta_int.validate_python(mixed_array)

        # Empty arrays with type conversion
        empty_int_list = []
        validated_empty = ta_float.validate_python(empty_int_list)
        self.assertTrue(np.issubdtype(validated_empty.dtype, np.floating))
        self.assertEqual(len(validated_empty), 0)

    def test_Numpy1DArray_numpy_dtypes(self):
        """Test Numpy1DArray with various specific NumPy dtypes."""

        # Define type adapters for different NumPy dtypes
        ta_float16 = TypeAdapter(Numpy1DArray[np.float16])
        ta_float32 = TypeAdapter(Numpy1DArray[np.float32])
        ta_int8 = TypeAdapter(Numpy1DArray[np.int8])
        ta_int16 = TypeAdapter(Numpy1DArray[np.int16])
        ta_int64 = TypeAdapter(Numpy1DArray[np.int64])
        ta_uint8 = TypeAdapter(Numpy1DArray[np.uint8])
        ta_uint64 = TypeAdapter(Numpy1DArray[np.uint64])
        ta_complex64 = TypeAdapter(Numpy1DArray[np.complex64])

        # Test basic validation - each dtype should accept its own type
        float32_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        validated_float32 = ta_float32.validate_python(float32_array)
        self.assertEqual(validated_float32.dtype, np.float32)

        int64_array = np.array([1, 2, 3], dtype=np.int64)
        validated_int64 = ta_int64.validate_python(int64_array)
        self.assertEqual(validated_int64.dtype, np.int64)

        complex64_array = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        validated_complex64 = ta_complex64.validate_python(complex64_array)
        self.assertEqual(validated_complex64.dtype, np.complex64)

        # Test type coercion between different numpy dtypes
        # float32 -> float16 (potential precision loss)
        float16_from_float32 = ta_float16.validate_python(float32_array)
        self.assertEqual(float16_from_float32.dtype, np.float16)

        # int32 -> int64 (upcast)
        int32_array = np.array([1, 2, 3], dtype=np.int32)
        int64_from_int32 = ta_int64.validate_python(int32_array)
        self.assertEqual(int64_from_int32.dtype, np.int64)

        # int32 -> float32 (type conversion)
        float32_from_int32 = ta_float32.validate_python(int32_array)
        self.assertEqual(float32_from_int32.dtype, np.float32)

        # Test edge cases
        # 1. Overflow for smaller integer types
        large_int = np.array([300], dtype=np.int16)  # larger than int8 can handle
        self.assertEqual(ta_int8.validate_python(large_int)[0], 44)  # 300 % 256

        # 2. Underflow for unsigned types
        negative_int = np.array([-1], dtype=np.int32)
        self.assertEqual(ta_uint8.validate_python(negative_int)[0], 255)  # -1 % 256

        # 3. Special values in float types
        special_floats = np.array([np.inf, np.nan, -np.inf], dtype=np.float64)
        validated_special_float32 = ta_float32.validate_python(special_floats)
        self.assertEqual(validated_special_float32.dtype, np.float32)
        self.assertTrue(np.isnan(validated_special_float32[1]))
        self.assertTrue(np.isinf(validated_special_float32[0]))

        # 4. Very large values for float16 (potential overflow)
        large_float = np.array([1e20], dtype=np.float64)  # too large for float16
        float16_large = ta_float16.validate_python(large_float)
        self.assertTrue(np.isinf(float16_large[0]))  # Should be inf in float16

        # Test JSON serialization and deserialization
        # float32 serialization
        json_float32 = ta_float32.dump_json(float32_array)
        float32_roundtrip = ta_float32.validate_json(json_float32)
        np.testing.assert_almost_equal(float32_array, float32_roundtrip)
        self.assertEqual(float32_roundtrip.dtype, np.float32)

        # int64 serialization
        json_int64 = ta_int64.dump_json(int64_array)
        int64_roundtrip = ta_int64.validate_json(json_int64)
        np.testing.assert_array_equal(int64_array, int64_roundtrip)
        self.assertEqual(int64_roundtrip.dtype, np.int64)

        # complex64 serialization (will be serialized as [real, imag] pairs)
        json_complex64 = ta_complex64.dump_json(complex64_array)
        # JSON doesn't have native complex numbers, check the loaded JSON format
        loaded_complex = json.loads(json_complex64)
        # Each complex number should be serialized as a list of [real, imaginary]
        self.assertEqual(len(loaded_complex), len(complex64_array))

        # Test empty arrays with specific dtypes
        empty_float32 = np.array([], dtype=np.float32)
        validated_empty_float32 = ta_float32.validate_python(empty_float32)
        self.assertEqual(validated_empty_float32.dtype, np.float32)
        self.assertEqual(len(validated_empty_float32), 0)

        # Test with Python lists
        float32_from_list = ta_float32.validate_python([1.0, 2.0, 3.0])
        self.assertEqual(float32_from_list.dtype, np.float32)

        int16_from_list = ta_int16.validate_python([1, 2, 3])
        self.assertEqual(int16_from_list.dtype, np.int16)

        # Test uint64 max value
        max_uint64 = np.array([2**64 - 1], dtype=np.uint64)
        validated_uint64 = ta_uint64.validate_python(max_uint64)
        self.assertEqual(validated_uint64[0], 2**64 - 1)
        self.assertEqual(validated_uint64.dtype, np.uint64)

    def test_NumpyNDArray_numpy_dtypes(self):
        # Define type adapters for different NumPy dtypes
        ta_float16 = TypeAdapter(NumpyNDArray[np.float16])
        ta_float32 = TypeAdapter(NumpyNDArray[np.float32])
        ta_int8 = TypeAdapter(NumpyNDArray[np.int8])
        ta_int16 = TypeAdapter(NumpyNDArray[np.int16])
        ta_int32 = TypeAdapter(NumpyNDArray[np.int32])
        ta_int64 = TypeAdapter(NumpyNDArray[np.int64])
        ta_uint8 = TypeAdapter(NumpyNDArray[np.uint8])
        ta_complex64 = TypeAdapter(NumpyNDArray[np.complex64])

        # Test basic validation with N-dimensional arrays
        # 1D arrays
        float32_array_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        validated_float32_1d = ta_float32.validate_python(float32_array_1d)
        self.assertEqual(validated_float32_1d.dtype, np.float32)

        # 2D arrays
        float32_array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        validated_float32_2d = ta_float32.validate_python(float32_array_2d)
        self.assertEqual(validated_float32_2d.dtype, np.float32)
        self.assertEqual(validated_float32_2d.shape, (2, 2))

        # 3D arrays
        int16_array_3d = np.ones((2, 3, 4), dtype=np.int16)
        validated_int16_3d = ta_int16.validate_python(int16_array_3d)
        self.assertEqual(validated_int16_3d.dtype, np.int16)
        self.assertEqual(validated_int16_3d.shape, (2, 3, 4))

        # Test type coercion between different numpy dtypes
        # float32 -> float16 (potential precision loss)
        float16_from_float32_2d = ta_float16.validate_python(float32_array_2d)
        self.assertEqual(float16_from_float32_2d.dtype, np.float16)

        # int32 -> int64 (upcast)
        int32_array_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
        int64_from_int32_2d = ta_int64.validate_python(int32_array_2d)
        self.assertEqual(int64_from_int32_2d.dtype, np.int64)

        # Test with scalar input (should create 0-dimensional array)
        scalar_int32 = ta_int32.validate_python(5)
        self.assertEqual(scalar_int32.dtype, np.int32)
        self.assertEqual(scalar_int32.ndim, 0)  # 0-dimensional array
        self.assertEqual(scalar_int32.item(), 5)

        # Test JSON serialization and deserialization with multidimensional arrays
        # 2D float32 array
        json_float32_2d = ta_float32.dump_json(float32_array_2d)
        float32_2d_roundtrip = ta_float32.validate_json(json_float32_2d)
        np.testing.assert_almost_equal(float32_array_2d, float32_2d_roundtrip)
        self.assertEqual(float32_2d_roundtrip.dtype, np.float32)

        # 3D int16 array
        json_int16_3d = ta_int16.dump_json(int16_array_3d)
        int16_3d_roundtrip = ta_int16.validate_json(json_int16_3d)
        np.testing.assert_array_equal(int16_array_3d, int16_3d_roundtrip)
        self.assertEqual(int16_3d_roundtrip.dtype, np.int16)

        # Test edge cases
        # 1. Empty arrays with dimensions
        empty_float32_2d = np.zeros((0, 3), dtype=np.float32)
        validated_empty_float32_2d = ta_float32.validate_python(empty_float32_2d)
        self.assertEqual(validated_empty_float32_2d.dtype, np.float32)
        self.assertEqual(validated_empty_float32_2d.shape, (0, 3))

        # 2. Arrays with mixed dtypes (should convert to specified type)
        mixed_array = np.array([[1, 2], [3.5, 4.2]])  # Default dtype is float64
        int8_from_mixed = ta_int8.validate_python(mixed_array)
        self.assertEqual(int8_from_mixed.dtype, np.int8)
        # Values should be truncated to integers
        np.testing.assert_array_equal(int8_from_mixed, np.array([[1, 2], [3, 4]], dtype=np.int8))

        # 3. Complex numbers in float arrays (should fail or convert real part)
        complex_array = np.array([[1 + 1j, 2 + 2j]], dtype=np.complex64)
        # Test if conversion happens or raises error
        float32_from_complex = ta_float32.validate_python(complex_array)
        # If it converts, it should take only the real part
        np.testing.assert_array_equal(float32_from_complex, np.array([[1.0, 2.0]], dtype=np.float32))

        # 4. Test uint8 (common for image processing)
        # Create a small "image" as uint8
        image = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],  # 2x2x3 RGB image
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        validated_image = ta_uint8.validate_python(image)
        self.assertEqual(validated_image.dtype, np.uint8)
        self.assertEqual(validated_image.shape, (2, 2, 3))

        # Test from Python nested lists to specific dtype
        int64_from_lists = ta_int64.validate_python([[1, 2], [3, 4]])
        self.assertEqual(int64_from_lists.dtype, np.int64)
        self.assertEqual(int64_from_lists.shape, (2, 2))

        # Test complex64 validation and serialization
        complex64_2d = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
        validated_complex64 = ta_complex64.validate_python(complex64_2d)
        self.assertEqual(validated_complex64.dtype, np.complex64)

        # Test serialization round-trip of complex numbers
        json_complex64 = ta_complex64.dump_json(complex64_2d)
        complex64_roundtrip = ta_complex64.validate_json(json_complex64)
        # Compare real and imaginary parts to avoid float precision issues
        np.testing.assert_almost_equal(complex64_2d.real, complex64_roundtrip.real)
        np.testing.assert_almost_equal(complex64_2d.imag, complex64_roundtrip.imag)
        self.assertEqual(complex64_roundtrip.dtype, np.complex64)

    def test_Numpy1DArray_from_list_validation(self):
        """Test Numpy1DArray validation from Python lists."""
        ta_float = TypeAdapter(Numpy1DArray[float])
        ta_int = TypeAdapter(Numpy1DArray[int])
        ta_str = TypeAdapter(Numpy1DArray[str])

        # Simple lists
        float_list = [1.0, 2.0, 3.0]
        validated_float = ta_float.validate_python(float_list)
        self.assertEqual(validated_float.dtype, np.float64)
        self.assertEqual(validated_float.shape, (3,))

        # Mixed list that can be converted to float
        mixed_list = [1, 2.5, 3]
        validated_mixed = ta_float.validate_python(mixed_list)
        self.assertEqual(validated_mixed.dtype, np.float64)

        # List with non-convertible elements
        with self.assertRaises(ValidationError):
            ta_int.validate_python([1, 2, "not_an_int"])

        # Empty list
        validated_empty = ta_str.validate_python([])
        self.assertEqual(validated_empty.shape, (0,))
        self.assertEqual(validated_empty.dtype.kind, "U")

        # Nested list should fail for 1D array
        with self.assertRaises(ValidationError):
            ta_float.validate_python([[1.0], [2.0]])

    def test_Numpy1DArray_serialization_edge_cases(self):
        """Test serialization edge cases for Numpy1DArray."""
        ta = TypeAdapter(Numpy1DArray[float])

        # Test with very large values
        large_vals = np.array([1e100, 1e-100])
        json_large = ta.dump_json(large_vals)
        large_roundtrip = ta.validate_json(json_large)
        np.testing.assert_array_equal(large_vals, large_roundtrip)

        # Test with special string values when using string arrays
        ta_str = TypeAdapter(Numpy1DArray[str])
        special_strs = np.array(['{"quoted": "json"}', "\\backslash", '"quotes"'])
        json_strs = ta_str.dump_json(special_strs)
        strs_roundtrip = ta_str.validate_json(json_strs)
        np.testing.assert_array_equal(special_strs, strs_roundtrip)

        # Test with bool arrays
        ta_bool = TypeAdapter(Numpy1DArray[bool])
        bool_arr = np.array([True, False, True])
        json_bool = ta_bool.dump_json(bool_arr)
        self.assertEqual(json.loads(json_bool), [True, False, True])
        bool_roundtrip = ta_bool.validate_json(json_bool)
        np.testing.assert_array_equal(bool_arr, bool_roundtrip)

    def test_NumpyNDArray_ragged_arrays(self):
        """Test handling of ragged arrays with NumpyNDArray."""
        ta = TypeAdapter(NumpyNDArray[float])

        # Ragged Python list (uneven dimensions)
        ragged_list = [[1.0, 2.0], [3.0, 4.0, 5.0]]

        # unequal lengths
        self.assertRaises(ValidationError, ta.validate_python, ragged_list)

    def test_NumpyNDArray_with_complex_types(self):
        """Test NumpyNDArray with complex data types."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __eq__(self, other):
                if not isinstance(other, Point):
                    return False
                return self.x == other.x and self.y == other.y

        # Try with custom class
        ta = TypeAdapter(NumpyNDArray[Point])

        # Create array of Points
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        point_array = np.array([p1, p2], dtype=object)

        validated_points = ta.validate_python(point_array)
        self.assertEqual(validated_points[0].x, 1)
        self.assertEqual(validated_points[1].y, 4)

        # Test with dict type
        ta_dict = TypeAdapter(NumpyNDArray[dict])
        dict_array = np.array([{"a": 1}, {"b": 2}], dtype=object)
        validated_dicts = ta_dict.validate_python(dict_array)
        self.assertEqual(validated_dicts[0]["a"], 1)
        self.assertEqual(validated_dicts[1]["b"], 2)

    def test_array_with_none_values(self):
        ta = TypeAdapter(Numpy1DArray[float])

        # Create array with None values
        data = np.array([1.0, None, 3.0], dtype=object)

        # Validate
        validated_data = ta.validate_python(data)
        self.assertEqual(validated_data.dtype, np.float64)
        self.assertEqual(validated_data[0], 1.0)
        self.assertTrue(math.isnan(validated_data[1]))
        self.assertEqual(validated_data[2], 3.0)

        # Test with all None values
        all_none = np.array([None, None], dtype=object)
        for raw_none in (all_none, [None, None]):
            validated_none = ta.validate_python(raw_none)
            # Note we convert the value
            self.assertTrue(math.isnan(validated_none[0]))
            self.assertTrue(math.isnan(validated_none[1]))

        # Test JSON serialization
        json_data = ta.dump_json(validated_data)
        loaded_json = json.loads(json_data)
        self.assertEqual(loaded_json, [1.0, None, 3.0])

        # Test round-trip
        validated_roundtrip = ta.validate_json(json_data)
        np.testing.assert_equal(validated_roundtrip[0], 1.0)
        self.assertTrue(math.isnan(validated_roundtrip[1]))
        np.testing.assert_equal(validated_roundtrip[2], 3.0)
