from typing import Any, Dict, List, NamedTuple, Optional, Union, Tuple
import tensorflow as tf
from enum import Enum
import numpy as np

from nauto_datasets.core import spark
from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.drt import events, judgments, media
from nauto_datasets.drt.data_source import DRTConfig
from nauto_datasets.drt.types import (EventToMainJudgment, JudgmentType,
                                      MediaType)
from nauto_datasets.reprs import tfrecords as tf_repr
from nauto_datasets.serialization.tfrecords import decoding as tf_decoding
from nauto_datasets.serialization.tfrecords import encoding as tf_encoding
from nauto_datasets.utils import protobuf
from nauto_datasets.utils.numpy import NDArray
from nauto_datasets.utils.category import Monoid
from sensor import sensor_pb2


def _enum_name(et: Enum) -> str:
    return et.name.lower()


class JudgmentInfoParser:
    """Parser of judgment info column.
    It is used to add additional subfields to the relevant labels field
    in the record named tuple, e.g. for the COLLISION judgment there will be
    created a 'collision_info' field of the type returned by 'info_type'

    Note that `info_type` should return a type serializable and parsable by
    tfrecords encoders and decoders.
    """

    @staticmethod
    def info_type() -> type:
        raise NotImplementedError()

    @staticmethod
    def from_info(info: Optional[str]) -> Any:
        """Parses info column and returns an instance of 'info_type()'.
        """
        raise NotImplementedError()


class MediaMessagesInfo(NamedTuple):
    message_ids: NDArray[np.int64]
    message_paths: NDArray[np.str]


def create_data_named_tuple(
        media_types: List[MediaType],
        alt_recording_type: Optional[type] = None
) -> NamedTuple:
    """Returns NamedTuple class with field for each media type

    Args:
        media_types: types to include in the resulting tuple
        alt_recording_type: a named tuple type, with a static constructor
           `from_recording(CombinedRecording)`.
           If provided by the user, this type will be used to serialize the
           recording instead of regular `CombinedRecording` type.
           This can be used to modify or select some fields from the
           original `CombinedRecording` before serialization

    Returns: NamedTuple with a subset of the fields:
            - 'sensor': CombinedRecording
            - 'snapshot_in': NDArray[bytes]
            - 'snapshot_out': NDArray[bytes]
            - 'video_in': NDArray[bytes]
            - 'video_out': NDArray[bytes]
    Raises:
        ValueError: when some media type is not supported
    """
    fields = {}
    for mt in media_types:
        if mt == MediaType.SENSOR:
            fields[_enum_name(mt)] = alt_recording_type or CombinedRecording
        else:
            # list of byte chunks representing videos or snapshots
            fields[_enum_name(mt)] = NDArray[bytes]

    return NamedTuple('Data', **fields)


def create_labels_named_tuple(
        judgment_types: List[JudgmentType],
        info_parsers: Optional[Dict[JudgmentType, JudgmentInfoParser]] = None
) -> NamedTuple:
    """Returns NamedTuple class with boolean field for judgment type

    Args:
        judgment_types: types to include in the resulting tuple

    Returns:
        NamedTuple with a subset of the fields:
            - distraction: bool
            - distraction_info: 'info_parser[DISTRACTION].info_type()'
            - tailgating: bool
            - tailgating_info: 'info_parser[TAILATING].info_type()'
            - collision: bool
            - collision_info: 'info_parser[COLLISION].info_type()'
            - ...
    """
    fields = dict((_enum_name(jt), bool) for jt in judgment_types)
    if info_parsers is not None:
        for jt, parser in info_parsers.items():
            fields[f'{_enum_name(jt)}_info'] = parser.info_type()
    return NamedTuple('Labels', **fields)


def create_metadata_named_tuple(media_types: List[MediaType]) -> NamedTuple:
    """Returns NamedTuple class with fields containin metadata

    Returned tuple contains fields for every event column, as well as
    additional field 'media' with information about messages for each
    event type:
        NamedTuple:
            - 'event_id': np.int64
            ...
            - media: NamedTuple:
                 - sensor: MediaMessagesInfo:
                     - message_ids
                     - message_paths
                 - video_in: ...
                 ...
    """
    media_fields = dict(
        (_enum_name(mt), MediaMessagesInfo) for mt in media_types
    )
    medias_tuple_t = NamedTuple('MediaInfos', **media_fields)

    fields = dict(
        (col.name.lower(), spark.spark_dt_to_numpy_dt(col.dataType))
        for col in events.EventColumns.all())
    fields['media'] = medias_tuple_t

    return NamedTuple('Metadata', **fields)


def create_record_named_tuple(
        media_types: List[MediaType],
        judgment_types: List[JudgmentType],
        judgment_info_parsers: Optional[Dict[JudgmentType, JudgmentInfoParser]] = None,
        alt_recording_type: Optional[type] = None
) -> NamedTuple:
    """Returns NamedTuple class with metadata, data and labels fields

    Args:
        media_types: media types to be part of the data
        judgment_types: judgment types for which labels should be created
    Returns:
        NamedTuple instance constructor
    """
    metadata_nt = create_metadata_named_tuple(media_types)
    data_nt = create_data_named_tuple(media_types, alt_recording_type)
    labels_nt = create_labels_named_tuple(judgment_types, judgment_info_parsers)

    return NamedTuple(
        'DrtTfRecord',
        metadata=metadata_nt,
        data=data_nt,
        labels=labels_nt)


def _parse_data(
        media_types: List[MediaType],
        data_dict: Dict[str, List[bytes]],
        alt_recording_type: Optional[type] = None
) -> Dict[str, Any]:
    parsed_data = {}
    for mt in media_types:
        if mt == MediaType.SENSOR:
            recordings = [
                Recording.from_pb(
                    protobuf.parse_message_from_gzipped_bytes(
                        sensor_pb2.Recording, msg_bytes))
                for msg_bytes in data_dict[media.get_paths_column(mt).name]
            ]
            recording = CombinedRecording.from_recordings(recordings)
            if alt_recording_type is not None:
                parsed_data[_enum_name(mt)] = \
                    alt_recording_type.from_recording(recording)
            else:
                parsed_data[_enum_name(mt)] = recording
        else:
            parsed_data[_enum_name(mt)] = np.array(
                data_dict[media.get_paths_column(mt).name],
                dtype=bytes)

    return parsed_data


class DRTRecordAggregator(Monoid):

    @staticmethod
    def zero() -> 'DRTRecordAggregator':
        return DRTRecordAggregator()

    @staticmethod
    def add(
        agg_1: 'DRTRecordAggregator',
        agg_2: 'DRTRecordAggregator'
    ) -> 'DRTRecordAggregator':
        return agg_1

    @staticmethod
    def from_record_named_tuple(record_data: NamedTuple) -> 'DRTRecordAggregator':
        """Initialize aggregator.
        Args:
            record_data: this is a NamedTuple, the same as the one returned from
                `create_record_named_tuple`. Note, that this type is created
                dynamically based on the config used to create the dataset.
        """
        return DRTRecordAggregator()


def get_tf_features_config(
        dc: DRTConfig,
        media_types: List[MediaType],
        judgment_info_parsers: Optional[Dict[JudgmentType, JudgmentInfoParser]] = None,
        aggregator_t: type = DRTRecordAggregator,
        use_tensor_protos: bool = False,
        ignore_sequence_features: bool = False,
        alt_recording_type: Optional[type] = None,
        tf_record_options: Optional[tf.io.TFRecordOptions] = None
) -> tf_repr.TfFeaturesConfig:
    """Given chosen media types and `DRTConfig` produce the `TfFeaturesCreator`

    Args:
        dc: DRTConfig for which parsers have to be created
        media_types: types of the media expected to be parts of the tf records
        judgment_info_parsers: parsers of labels info columns
        aggregator_t: a subclass of DRTRecordAggregator providing
           'from_record_named_tuple' static method. Can be provided by the user
           to aggregate some values accross the entire dataset split
        use_tensor_protos: whether tensors should be serialized indirectly
            as tensor protos
        ignore_sequence_features: whether sequence tensors should be ignored
            - only context tensors will be serialized
        alt_recording_type: a named tuple type, with a static constructor
           `from_recording(CombinedRecording)`.
           If provided by the user, this type will be used to serialize the
           recording instead of regular `CombinedRecording` type.
           This can be used to modify or select some fields from the
           original `CombinedRecording` before serialization
        tf_record_options: options passed to `TFRecordWriter`

    Returns:
        TfFeaturesCreator responsible for combining provided metadata, labels
             and data features into a single `tf.Example`/`tf.SequenceExample`
    """
    if not set(dc.media_types).issuperset(set(media_types)):
        raise ValueError(
            'Required media_types are not a subset of provided media types')

    judgment_types = list(set(
        [EventToMainJudgment[dc.event_type]] + dc.judgment_tags))

    event_columns = [
        col.name for col in events.EventColumns.all()
    ]

    media_paths_columns = [media.get_paths_column(mt).name for mt in media_types]
    media_ids_columns = [media.get_ids_column(mt).name for mt in media_types]

    labels_columns = [
        judgments.get_label_column(jt).name for jt in judgment_types
    ]
    info_columns = [
        judgments.get_info_column(jt).name for jt in judgment_types
    ]

    def get_tf_features_producer():
        # have to be created inside the map job in spark, because of the
        # problems with pickling/serialization of this dynamically created
        # classes
        record_nt = create_record_named_tuple(
            media_types,
            judgment_types,
            judgment_info_parsers,
            alt_recording_type)
        metadata_nt = record_nt._field_types['metadata']
        metadata_media_nt = metadata_nt._field_types['media']
        data_nt = record_nt._field_types['data']
        labels_nt = record_nt._field_types['labels']

        def produce_tf_features(
                column_data: Dict[str, Any],
                media_data: Dict[str, Union[List[bytes], bytes]]
        ) -> Tuple[tf_encoding.TfFeatures, DRTRecordAggregator]:

            event_values = [column_data[col_name] for col_name in event_columns]
            media_paths_values = [column_data[col_name] for col_name in media_paths_columns]
            media_ids_values = [column_data[col_name] for col_name in media_ids_columns]
            labels_values = [column_data[col_name] for col_name in labels_columns]

            # --- metadata fields ----
            metadata_media_args = {
                _enum_name(mt): MediaMessagesInfo(
                    message_ids=np.array(ids, np.int64),
                    message_paths=np.array(paths, np.str))
                for mt, ids, paths
                in zip(media_types, media_ids_values, media_paths_values)
            }
            metadata_media = metadata_media_nt(**metadata_media_args)
            event_args = dict(zip(event_columns, event_values))
            # cast values to appropriate types
            event_args = {
                name: metadata_nt._field_types[name](val)
                for name, val in event_args.items()
            }
            metadata_args = dict(media=metadata_media, **event_args)
            metadata = metadata_nt(**metadata_args)

            # ---- labels fields ----
            labels_args = dict(zip(
                [_enum_name(jt) for jt in judgment_types],
                labels_values))
            # cast values to appropriate types
            labels_args = {
                name: labels_nt._field_types[name](val)
                for name, val in labels_args.items()
            }
            for jt, info_parser in judgment_info_parsers.items():
                parsed = info_parser.from_info(
                    column_data[judgments.get_info_column(jt).name])
                labels_args[f'{_enum_name(jt)}_info'] = parsed
            labels = labels_nt(**labels_args)

            # ---- data fields -----
            data_args = _parse_data(media_types, media_data, alt_recording_type)
            data = data_nt(**data_args)

            # ---- drt record ----
            drt_record = record_nt(metadata=metadata, data=data, labels=labels)

            return (
                tf_encoding.structure_to_features(
                    drt_record,
                    use_tensor_protos=use_tensor_protos,
                    ignore_sequence_features=ignore_sequence_features),
                aggregator_t.from_record_named_tuple(drt_record)
            )

        return produce_tf_features

    return tf_repr.TfFeaturesConfig(
        get_features_producer=get_tf_features_producer,
        tf_record_options=tf_record_options,
        use_sequence_examples=not ignore_sequence_features,
        aggregator_t=aggregator_t,
        columns_to_fetch=media_paths_columns
    )


def get_tf_features_parsers(
        dc: DRTConfig,
        media_types: List[MediaType],
        judgment_info_parsers: Optional[Dict[JudgmentType, JudgmentInfoParser]] = None,
        parse_tensor_protos: bool = False,
        ignore_sequence_features: bool = False,
        alt_recording_type: Optional[type] = None
) -> tf_decoding.TfFeatureParsers:
    """Given chosen media types and `DRTConfig` produce the TfRecord parsers

    Args:
        dc: DRTConfig for which parsers have to be created
        media_types: types of the media expected to be parts of the tf records
        judgment_info_parsers: parsers of judgment info columns
        parse_tensor_protos: whether tensors are serialized as Tensor Protos
        ignore_sequence_features: whether sequence tensors should be ignored
            - only context tensors will be considered during parsing
        alt_recording_type: a named tuple type, with a static constructor
           `from_recording(CombinedRecording)`.
           If provided by the user, this type will be used to serialize the
           recording instead of regular `CombinedRecording` type.
           This can be used to modify or select some fields from the
           original `CombinedRecording` before serialization
    Returns:
        `TfFeatureParsers` to used read serialized tfrecord
        examples/sequence examples
    """
    judgment_types = list(set(
        [EventToMainJudgment[dc.event_type]] + dc.judgment_tags))

    return tf_decoding.nested_type_to_feature_parsers(
        create_record_named_tuple(
            media_types,
            judgment_types,
            judgment_info_parsers,
            alt_recording_type),
        parse_tensor_protos=parse_tensor_protos,
        ignore_sequence_features=ignore_sequence_features)

