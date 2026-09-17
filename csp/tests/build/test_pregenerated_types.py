import sys
from pathlib import Path
from subprocess import call
from tempfile import TemporaryDirectory

AUTOGEN_SCRIPT = (Path(__file__).parent / ".." / ".." / "build" / "csp_autogen.py").resolve()


class TestPregeneratedTypes:
    def test_engine(self):
        pregenerated_folder = (
            Path(__file__).parent / ".." / ".." / ".." / "cpp" / "csp" / "engine" / "csp_autogen"
        ).resolve()
        pregenerated_header = pregenerated_folder / "autogen_types.h"
        pregenerated_cpp = pregenerated_folder / "autogen_types.cpp"

        with TemporaryDirectory() as td:
            call(
                [
                    sys.executable,
                    str(AUTOGEN_SCRIPT),
                    "-m",
                    "csp.impl.types.autogen_types",
                    "-d",
                    td,
                    "-o",
                    "autogen_types",
                    "--omit_asserts",
                ]
            )
            generated_header = Path(td) / "autogen_types.h"
            generated_cpp = Path(td) / "autogen_types.cpp"

            # Path is different, so skip the lines
            # that show the command
            assert pregenerated_header.read_text().split("\n")[4:] == generated_header.read_text().split("\n")[4:]
            assert pregenerated_cpp.read_text().split("\n")[4:] == generated_cpp.read_text().split("\n")[4:]

    def test_websocket(self):
        pregenerated_folder = (
            Path(__file__).parent / ".." / ".." / ".." / "cpp" / "csp" / "adapters" / "websocket" / "csp_autogen"
        ).resolve()
        pregenerated_header = pregenerated_folder / "websocket_types.h"
        pregenerated_cpp = pregenerated_folder / "websocket_types.cpp"

        with TemporaryDirectory() as td:
            call(
                [
                    sys.executable,
                    str(AUTOGEN_SCRIPT),
                    "-m",
                    "csp.adapters.websocket_types",
                    "-d",
                    td,
                    "-o",
                    "websocket_types",
                    "--omit_asserts",
                ]
            )
            generated_header = Path(td) / "websocket_types.h"
            generated_cpp = Path(td) / "websocket_types.cpp"

            # Path is different, so skip the lines
            # that show the command
            assert pregenerated_header.read_text().split("\n")[4:] == generated_header.read_text().split("\n")[4:]
            assert pregenerated_cpp.read_text().split("\n")[4:] == generated_cpp.read_text().split("\n")[4:]
