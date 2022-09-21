from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nauto_datasets.utils.numpy import NDArray, get_underlying_dtype, is_ndarray_type
from nauto_datasets.utils.tuples import NamedTupleRecreator


class StreamMixin:
    """Mixin adding some stream/array related utility methods to a NamedTuple

    This mixin is intended to be used with NamedTuples consisting
    only of fields being numpy arrays or lists of the same lengths or length equal to 0.

    The first field is considered an indexing field. It is used for the size
    calculation. The other fields are either the size of the indexing field or
    empty.
    """

    def __init__(
            self, *arrays: List[np.ndarray],
            **arr_dict: Dict[str, np.ndarray]
    ) -> None:
        arrays = list(arrays) + list(arr_dict.values())
        if len(arrays) == 0:
            raise ValueError('Cannot create a stream with no arrays')

        index_stream = self._fields[0]
        index_arr = arr_dict.get(index_stream)
        if index_arr is None:
            index_arr = arrays[0]

        if index_arr is None:
            raise ValueError('No indexing array provided')

        if np.any([
                (len(arr) != len(index_arr) and (len(arr) != 0))
                for arr in arrays]):
            raise ValueError('Stream arrays have to be empty or of equal size')

    def _size(self) -> int:
        return len(getattr(self, self._fields[0]))

    def _is_empty(self) -> bool:
        return self._size() == 0

    def _to_df(self) -> pd.DataFrame:
        size = self._size()
        # do not include empty values
        columns = {
            name: vals for name, vals in self._asdict().items()
            if len(vals) == size
        }
        return pd.DataFrame(data=columns)

    def _range(self, arr_name: str) -> Optional[Tuple[Any, Any]]:
        if self._is_empty():
            return None
        else:
            arr = getattr(self, arr_name)
            return (arr[0], arr[-1])

    @classmethod
    def concat(cls, streams: List['StreamMixin']) -> 'StreamMixin':
        def concatenate(arrays, field_t) -> Union[np.ndarray, List]:
            if is_ndarray_type(field_t):
                return np.concatenate(arrays)
            else:
                return sum(arrays, [])
        arrays = {
            name: concatenate(
                [getattr(stream, name) for stream in streams],
                field_t)
            for name, field_t in cls._field_types.items()
        }
        return cls(**arrays)

    @classmethod
    def empty(cls) -> 'StreamMixin':
        def maybe_arr(field_t) -> Union[np.ndarray, List]:
            if is_ndarray_type(field_t):
                return np.array([], dtype=get_underlying_dtype(field_t))
            else:
                return []
        arrays = {
            name: maybe_arr(field_t)
            for name, field_t in cls._field_types.items()
        }
        return cls(**arrays)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'StreamMixin':
        def array_or_list(field_t, vals: np.ndarray) -> Union[np.ndarray, List]:
            if is_ndarray_type(field_t):
                return np.array(vals, dtype=get_underlying_dtype(field_t))
            else:
                return list(vals)

        columns = dict(zip(df.T.index, df.T.values))
        for name, field_t in cls._field_types.items():
            if name not in columns:
                # add empty values for missing columns
                columns[name] = array_or_list(field_t, [])
            else:
                columns[name] = array_or_list(field_t, columns[name])
        return cls(**columns)


class CombinedStreamMixin:
    """Mixin adding meta operation on a stream consting of several
    streams combined together.

    This mixin should be used in rather indirect way through the function
    @create_combined_stream_type.
    """

    def __init__(self, stream: StreamMixin, lengths: List[int]) -> None:
        if stream._size() != np.sum(lengths):
            raise ValueError(
                'Stream length is not equal to the sum of lengths')

    @classmethod
    def from_substreams(cls,
                        streams: List[StreamMixin]) -> 'CombinedStreamMixin':
        if len(streams) == 0:
            raise ValueError('streams list must be non empty')
        stream_t = type(streams[0])
        return cls(
            stream=stream_t.concat(streams),
            lengths=np.array(
                [stream._size() for stream in streams], dtype=np.int32))

    def _substreams_count(self) -> int:
        return len(self.lengths)

    def _substreams_offsets(self) -> np.ndarray:
        return np.cumsum(self.lengths)

    def _substreams(self) -> List[StreamMixin]:
        offsets = self._substreams_offsets()
        substreams = []
        for i in range(self._substreams_count()):
            beg = 0 if i == 0 else offsets[i-1]
            end = offsets[i]
            substreams.append(
                self.stream._replace(
                    **{
                        name: getattr(self.stream, name)[beg:end]
                        for name in self.stream._fields
                        if len(getattr(self.stream, name)) != 0
                    }))
        return substreams

    def _ith_substream(self, id: int) -> StreamMixin:
        offsets = self._substreams_offsets()
        beg = 0 if id == 0 else offsets[id-1]
        end = offsets[id]
        return self.stream._replace(
            **{
                name: getattr(self.stream, name)[beg:end]
                for name in self.stream._fields
                if len(getattr(self.stream, name)) != 0
            })


def create_combined_stream_type(
        name: str,
        base_stream_t: type,
        bases: Optional[List[type]] = None,
        module: Optional[str] = None
) -> type:
    """Returns a type representing a combination of streams of
    type `base_stream_t`

    Resulting type will be a `NamedTuple` with fields:
        stream: of type `base_stream_t`
        lengths: a int32 numpy array with lengths of every substream
        module: name of the module of the new type. If not provided it
           will be equal to the `base_stream_t`'s module

    In addition to that the new type will have `CombinedStreamMixin`
    class mixed into it.
    """
    if not issubclass(base_stream_t, StreamMixin):
        raise ValueError('base_stream_t should be a StreamMixin')

    bases = bases or []

    tuple_t = NamedTuple(
        'Com{0}Base'.format(base_stream_t.__name__),
        [
            ('stream', base_stream_t),
            ('lengths', NDArray[np.int32])
        ])
    new_t = type(
        name,
        (NamedTupleRecreator, CombinedStreamMixin, tuple_t, *bases),
        {})

    if module is None:
        module = base_stream_t.__module__
    new_t.__module__ = module
    return new_t
