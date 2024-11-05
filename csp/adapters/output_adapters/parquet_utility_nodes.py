import typing

import numpy

import csp

T = typing.TypeVar("T")


@csp.node
def flatten_numpy_array(x: csp.ts[csp.typing.NumpyNDArray["T"]]) -> csp.Outputs(
    value=csp.ts[csp.typing.Numpy1DArray["T"]], shape=csp.ts[csp.typing.Numpy1DArray[int]]
):
    """Flattens the arrays in the given ts to 1d arrays
    :param x: The time series of arrays to flatten
    :return: "value" - the flattened arrays ts and "shape" the original shapes of the arrays that could be used to reshape the data back
    """
    if csp.ticked(x):
        return csp.output(value=x.reshape(-1), shape=numpy.array(x.shape, dtype=int))


@csp.node
def reshape_numpy_array(
    value: csp.ts[csp.typing.Numpy1DArray["T"]], shape: csp.ts[csp.typing.Numpy1DArray[int]]
) -> csp.ts[csp.typing.NumpyNDArray["T"]]:
    """Reshapes the value arrays (1d numpy arrays) using the shape values. The assumption is that both ts must tick synchronously
    :param value: The ts of value arrays to be reshaped
    :param shape: The ts of the shapes to which the values should be reshaped
    :return: A single ts of reshaped arrays
    """
    assert csp.ticked(value)
    assert csp.ticked(shape)
    return value.reshape(shape)
