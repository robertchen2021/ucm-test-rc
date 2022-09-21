from typing import Dict, List, Any
from .mappings import Mapping, SOURCE_TYPE_CONSTANT, SOURCE_TYPE_MEDIA
from nauto_zoo import ModelInput

# These are copy-pasted from datareviewer lib that is discontinued. Todo: find a better place
VIDEO_IN_MEDIA_TYPE = 'video-in'
VIDEO_OUT_MEDIA_TYPE = 'video-out'
SNAPSHOT_IN_MEDIA_TYPE = 'snapshot-in'
SNAPSHOT_OUT_MEDIA_TYPE = 'snapshot-out'
SENSOR_MEDIA_TYPE = 'sensor'


class TfRecordsMergePreprocessor():
    def __init__(self, mappings: List[Mapping]):
        self._mappings = mappings

    def preprocess_merge(self, model_input: ModelInput) -> bytes:
        from nauto_datasets.serialization.tfrecords import encoding as tf_encoding
        features_struct = self.media_to_features_struct(model_input.as_dict())
        return tf_encoding \
            .structure_to_features(features_struct, use_tensor_protos=False, ignore_sequence_features=True) \
            .to_example() \
            .SerializeToString()

    def media_to_features_struct(self, media: Dict[str, Any]) -> Dict:
        features_struct = {}
        for mapping in self._mappings:
            if mapping.source_type == SOURCE_TYPE_CONSTANT:
                features_struct[mapping.target] = mapping.source
            elif mapping.source_type == SOURCE_TYPE_MEDIA:
                if mapping.source in [VIDEO_IN_MEDIA_TYPE, VIDEO_OUT_MEDIA_TYPE, SNAPSHOT_OUT_MEDIA_TYPE,
                                      SNAPSHOT_IN_MEDIA_TYPE, SENSOR_MEDIA_TYPE]:
                    if mapping.source not in media:
                        raise RuntimeError(f"Media source `{mapping.source}` is not available")
                    features_struct[mapping.target] = media[mapping.source]
                else:
                    raise RuntimeError(f"Got unrecognized media source `{mapping.source}`")
            else:
                raise RuntimeError(f"Got unrecognized source type `{mapping.source_type}`")
        return features_struct
