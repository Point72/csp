import importlib
import os.path
import sys

import pytest

EXAMPLES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))

# make examples importable without keeping in source tree
sys.path.append(EXAMPLES_ROOT)


def _get_module(folder, filename):
    # Don't fail during test gathering
    try:
        return importlib.import_module(f"{folder}.{filename.replace('.py', '')}")
    except Exception:
        return None


def _get_modules_to_test(*folders):
    folder = ".".join(folders) if len(folders) > 0 else folders[0]
    return [
        (file, _get_module(folder, file))
        for file in os.listdir(os.path.join(EXAMPLES_ROOT, *folders))
        if file.endswith(".py")
    ]


def _no_examples_folder_or_running_sdist_tests():
    return os.environ.get("CSP_TEST_SKIP_EXAMPLES", None) is not None or not os.path.exists(EXAMPLES_ROOT)


@pytest.mark.skipif(_no_examples_folder_or_running_sdist_tests(), reason="no examples present or manually skipping")
class TestExamples:
    @pytest.mark.parametrize("filename,module", _get_modules_to_test("01_basics"))
    def test_1_basics(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("02_intermediate"))
    def test_2_intermediate(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("03_using_adapters", "parquet"))
    def test_3_adapters_parquet(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("04_writing_adapters"))
    def test_4_writing_adapters(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("06_advanced"))
    def test_6_advanced(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("98_just_for_fun"))
    def test_98_just_for_fun(self, filename, module):
        assert module.main
        module.main()

    @pytest.mark.parametrize("filename,module", _get_modules_to_test("99_developer_tools"))
    def test_99_developer_tools(self, filename, module):
        assert module.main
        module.main()
