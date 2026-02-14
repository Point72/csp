import unittest

from pydantic import BaseModel, Field, ValidationError

from csp.impl.error_handling import (
    INPUT_VALUE_TRUNCATE_LENGTH,
    fmt_errors,
    truncate_input_value,
)


class TestTruncateInputValue(unittest.TestCase):
    def test_short_value_unchanged(self):
        short_value = "short"
        self.assertEqual(truncate_input_value(short_value), short_value)

    def test_exact_length_unchanged(self):
        exact_value = "x" * INPUT_VALUE_TRUNCATE_LENGTH
        self.assertEqual(truncate_input_value(exact_value), exact_value)

    def test_long_value_truncated(self):
        long_value = "START" + "x" * INPUT_VALUE_TRUNCATE_LENGTH + "END"
        result = truncate_input_value(long_value)
        self.assertLess(len(result), INPUT_VALUE_TRUNCATE_LENGTH)
        self.assertIn("...", result)
        self.assertTrue(result.startswith("START"))
        self.assertTrue(result.endswith("END"))


class TestFmtErrors(unittest.TestCase):
    """
    Tests that verify our custom formatting matches pydantic's standard error messages.
    """

    def test_string_type_error_matches_pydantic(self):
        class Model(BaseModel):
            field: str

        try:
            Model(field=123)  # type: ignore
        except ValidationError as e:
            pydantic_msg = str(e)
            custom_msg = fmt_errors(e, "")
            self.assertEqual(pydantic_msg, custom_msg)

    def test_list_type_error_matches_pydantic(self):
        class Model(BaseModel):
            items: list[int]

        try:
            Model(items="not_a_list")  # type: ignore
        except ValidationError as e:
            pydantic_msg = str(e)
            custom_msg = fmt_errors(e, "")
            self.assertEqual(pydantic_msg, custom_msg)

    def test_nested_model_error_matches_pydantic(self):
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        try:
            Outer(inner={"value": "not_an_int"})  # type: ignore
        except ValidationError as e:
            pydantic_msg = str(e)
            custom_msg = fmt_errors(e, "")
            self.assertEqual(pydantic_msg, custom_msg)

    def test_list_item_error_location_matches_pydantic(self):
        class Model(BaseModel):
            items: list[int]

        try:
            Model(items=[1, "bad", 3])  # type: ignore
        except ValidationError as e:
            pydantic_msg = str(e)
            custom_msg = fmt_errors(e, "")
            self.assertEqual(pydantic_msg, custom_msg)

    def test_default_factory_not_called_matches_pydantic(self):
        class Model(BaseModel):
            a: int = Field(gt=10)
            b: int = Field(default_factory=lambda data: data["a"])

        try:
            Model(a=1)
            breakpoint()
        except ValidationError as e:
            pydantic_msg = str(e)
            custom_msg = fmt_errors(e, "")
            self.assertEqual(pydantic_msg, custom_msg)
            print(pydantic_msg)
