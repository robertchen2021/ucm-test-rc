from nauto_datasets.core.sensors import CombinedRecording
from nauto_datasets.serialization.tfrecords import decoding as tf_decoding
from nauto_datasets.serialization.tfrecords import encoding as tf_encoding


def combined_recording_to_features(
        com_recording: CombinedRecording,
        ignore_sequence_features: bool = False,
        use_tensor_protos: bool = False
) -> tf_encoding.TfFeatures:
    """Builds tf record features from `CombinedRecording`

    The resulting `TfFeatures` instance has the following form:
        `context`: features extracted from `CombinedStreamMixin`s, e.g.
             'acc/stream/x' or 'grv/lengths'
        `feature_lists`: features lists extracted from the lists of values
             including `EKFConfig`s and `RecordingMetadata`s, e.g.
             'ekf_configs/rot_angle_x` or 'metadatas/utc_boot_time_offset_ns'

    Args:
        com_recording: CombinedRecording instance
        ignore_sequence_features: whether sequence features should be produced
            from list items in CombinedRecording, e.g. metadatas
        use_tensor_protos: whether featueres should be encapsulated in
            serialized TensorProtos
    """
    return tf_encoding.structure_to_features(
        com_recording,
        ignore_sequence_features=ignore_sequence_features,
        use_tensor_protos=use_tensor_protos)


def combined_recording_parsers(
        ignore_sequence_features: bool = False,
        parse_tensor_protos: bool = False
) -> tf_decoding.TfFeatureParsers:
    """"Returns feature descriptions and handlers for `CombinedRecording`


    The resulting `TfFeatureParsers` instance has the following form:
        `context`: feature descriptions for fields extracted from
             `CombinedStreamMixin`s, e.g. 'acc/stream/x' or 'grv/lengths'
        `context_handlers`: `FeatureCreators` with producers returning
             features with the same names as associated **context** feature
             descriptions. These features might be results of casting or
             transforming sparse tensors to dense.
        `sequence`: features descriptions for extracted from the lists of
             values including `EKFConfig`s and `RecordingMetadata`s, e.g.
             'ekf_configs/rot_angle_x` or 'metadatas/utc_boot_time_offset_ns'
        `sequence_handlers`: `FeatureCreators` with producers returning
             features with the same names as associated **sequence** feature
             descriptions. These features might be results of casting or
             transforming sparse tensors to dense.

    Args:
        ignore_sequence_features: whether sequence features should be parsed
        parse_tensor_protos: when true, all context features will be assumed
            to be previousle serialized as TensorProtos
    """
    return tf_decoding.nested_type_to_feature_parsers(
        CombinedRecording,
        ignore_sequence_features=ignore_sequence_features,
        parse_tensor_protos=parse_tensor_protos)
