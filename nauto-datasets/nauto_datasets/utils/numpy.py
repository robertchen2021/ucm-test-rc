from typing import Generic, Type, TypeVar, Union

import numpy as np
import typing_inspect

D = TypeVar('D', bound=np.generic)


class NDArrayT(Generic[D]):
    """This is a utility type used for providing dtype annotations
    for numpy arrays.

    This class is not meant to be instantiated, as it is destined
    only for type introspection related to numpy arrays.

    The presence of this class is cause by the lack of numpy
    typing stubs, which could provide information on
    underlying dtypes.
    """

    def __init__(self) -> None:
        raise AssertionError(
            'This class should never be instantiated')


NDArray = NDArrayT[D]


def is_ndarray_type(type_t: type) -> bool:
    return (
        typing_inspect.is_generic_type(type_t)
        and typing_inspect.get_origin(type_t) is NDArrayT
    )


def get_underlying_dtype(ndarray_t: Type[NDArray]) -> np.dtype:
    """Returns an assumed np.dtype associated with given NDArray type"""
    return np.dtype(typing_inspect.get_args(ndarray_t)[0])


def datetime64_to_float_seconds(
        dt: Union[NDArray[np.datetime64], np.datetime64]) -> np.float64:
    return dt.astype('datetime64[ns]').astype(np.float64) / 1e+9


def datetime64_to_nano_seconds(
        dt: Union[NDArray[np.datetime64], np.datetime64]) -> np.int64:
    return dt.astype('datetime64[ns]').astype(np.int64)
