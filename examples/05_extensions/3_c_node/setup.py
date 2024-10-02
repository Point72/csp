import csp
from setuptools import Extension, setup


setup(
    name="csp-example-piglatin",
    version="0.0.1",
    packages=["piglatin"],
    ext_modules=[
        Extension(
            name="piglatin._piglatin",
            sources=["piglatin.c"],
            include_dirs=[csp.get_include_path()],
            library_dirs=[csp.get_lib_path()],
        ),
    ],
)
