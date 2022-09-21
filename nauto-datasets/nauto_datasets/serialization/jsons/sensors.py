import inspect
from typing import Any, Dict, List, NamedTuple

import numpy as np

from nauto_datasets.core.sensors import CombinedRecording
from nauto_datasets.core.streams import CombinedStreamMixin
from nauto_datasets.utils import numpy as np_utils


def _named_tuple_to_json_dict(nt_object: NamedTuple) -> Dict[str, Any]:

    str_json = {}
    for field_name, field_t in nt_object._field_types.items():
        field = getattr(nt_object, field_name)
        if np_utils.is_ndarray_type(field_t) or field_t == np.ndarray:
            str_json[field_name] = field.tolist()
        elif isinstance(field, np.generic):
            str_json[field_name] = field.item()
        elif isinstance(field, list):
            # NOTE: if List[T] then  T is assumed to be a NamedTuple
            # all primitive types should be NDArrays
            str_json[field_name] = [
                _named_tuple_to_json_dict(item) for item in field
            ]
        else:
            str_json[field_name] = field
    return str_json


def _get_data(com_rec: CombinedRecording) -> List[Dict[str, Any]]:
    data = []
    for field_name, field_t in com_rec._field_types.items():
        if not (inspect.isclass(field_t) and issubclass(field_t, CombinedStreamMixin)):
            continue

        com_stream = getattr(com_rec, field_name)
        stream_json = {
            'data': _named_tuple_to_json_dict(com_stream.stream),
            'tag': field_name,
            'lengths': com_stream.lengths.tolist()
        }

        if field_name == 'ekf':
            configs = [
                _named_tuple_to_json_dict(ekf_conf)
                for ekf_conf in com_rec.ekf_configs
            ]
            stream_json['configs'] = configs

        data.append(stream_json)

    return data


def _get_metadata(com_rec: CombinedRecording) -> Dict[str, Any]:
    file_info = []
    for metadata in com_rec.metadatas:
        serialized_meta = _named_tuple_to_json_dict(metadata)
        serialized_meta['sensorfile_version'] = metadata.version
        del serialized_meta['version']
        file_info.append(serialized_meta)

    return {'file_info': file_info}


def combined_recording_to_json(com_rec: CombinedRecording) -> Dict[str, Any]:
    return dict(
        data=_get_data(com_rec),
        metadata=_get_metadata(com_rec)
    )
