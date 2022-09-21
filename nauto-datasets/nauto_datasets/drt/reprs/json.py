import json
import numpy as np
from typing import Any, Dict, List, Optional

from pyspark.sql import Column
from pyspark.sql import functions as F

from nauto_datasets.drt import events, judgments, media
from nauto_datasets.drt import types as drt_types
from nauto_datasets.drt.data_source import DRTConfig
from nauto_datasets.reprs.sensor_jsons import JsonValueExtractor
from nauto_datasets.utils.numpy import NDArray


def get_event_medatada_extractor() -> JsonValueExtractor:

    def add_event_metadata_information(
            sensor_json: Dict[str, Any],
            event_id: int,
            device_id: int,
            message_id: int,
            fleet_id: str,
            region: str
    ) -> None:
        metadata = sensor_json['metadata']
        metadata['event_id'] = f'{np.uint64(event_id):x}'
        metadata['device_id'] = f'{np.uint64(device_id):x}'
        metadata['message_id'] = f'{np.uint64(message_id):x}'
        metadata['fleet_id'] = fleet_id
        metadata['region'] = region

    return JsonValueExtractor(
        arg_columns=[
            events.EventColumns.ID.name,
            events.EventColumns.DEVICE_ID.name,
            events.EventColumns.MESSAGE_ID.name,
            events.EventColumns.FLEET_ID.name,
            events.EventColumns.REGION.name
        ],
        extractor_fn=add_event_metadata_information)


def get_judgment_label_extractor(
        judgment_type: drt_types.JudgmentType
) -> JsonValueExtractor:
    arg_cols = [
        judgments.get_label_column(judgment_type).name,
        judgments.get_info_column(judgment_type).name
    ]

    def add_labels_information(
            sensor_json: Dict[str, Any],
            label: bool,
            info: Optional[str]
    ) -> None:
        if 'labels' not in sensor_json:
            sensor_json['labels'] = {}
        labels = sensor_json['labels']
        labels[arg_cols[0]] = label
        info_json = None
        if info is not None:
            info_json = json.loads(info)
        labels[arg_cols[1]] = info_json

    return JsonValueExtractor(
        arg_columns=arg_cols,
        extractor_fn=add_labels_information
    )


def get_media_extractor(media_type: drt_types.MediaType):

    arg_cols = [
        media.get_paths_column(media_type, True).name,
        media.get_ids_column(media_type, True).name
    ]

    def add_media_information(
            sensor_json: Dict[str, Any],
            media_paths: List[str],
            message_ids: List[int]
    ) -> None:
        metadata = sensor_json['metadata']
        if 'medias' not in metadata:
            metadata['medias'] = {}
        medias = metadata['medias']
        medias[arg_cols[0]] = media_paths
        medias[arg_cols[1]] = message_ids

    return JsonValueExtractor(
        arg_columns=arg_cols,
        extractor_fn=add_media_information)


def sensor_paths_column() -> str:
    return media.get_paths_column(drt_types.MediaType.SENSOR, True).name


def json_value_extractors(dc: DRTConfig) -> List[JsonValueExtractor]:

    extractors = [get_event_medatada_extractor()]

    judgment_types = [
        drt_types.EventToMainJudgment[dc.event_type]] + dc.judgment_tags
    for jt in judgment_types:
        extractors.append(get_judgment_label_extractor(jt))

    if drt_types.MediaType.SENSOR not in dc.media_types:
        raise ValueError(
            'Cannot create json value extractor for dataset with sensors')
    for mt in dc.media_types:
        extractors.append(get_media_extractor(mt))

    return extractors


def file_name(column_data: Dict[str, Any]) -> str:
    device_id = column_data[events.EventColumns.DEVICE_ID.name]
    message_id = column_data[events.EventColumns.MESSAGE_ID.name]
    return f'{np.uint64(device_id):x}_{np.uint64(message_id):x}.json'
