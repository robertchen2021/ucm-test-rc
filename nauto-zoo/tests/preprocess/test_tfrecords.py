import tensorflow as tf
import pytest
from nauto_zoo.preprocess.tfrecords import TfRecordsMergePreprocessor
from nauto_zoo.preprocess.mappings import SOURCE_TYPE_CONSTANT, SOURCE_TYPE_MEDIA, Mapping
from nauto_zoo import ModelInput


def test_should_map_constants_tfrecord():
    preprocessor = TfRecordsMergePreprocessor(mappings=[
        Mapping(source='foo', source_type=SOURCE_TYPE_CONSTANT, target='bad')
    ])
    preprocessed = preprocessor.preprocess_merge(ModelInput())
    assert type(preprocessed) is bytes
    example: tf.train.Example = tf.train.Example.FromString(preprocessed)
    assert 'bad' in repr(example)
    assert 'foo' in repr(example)


def test_should_map_media_tfrecord():
    preprocessor = TfRecordsMergePreprocessor(mappings=[
        Mapping(source='sensor', source_type=SOURCE_TYPE_MEDIA, target='sensor_target'),
        Mapping(source='constant_source', source_type=SOURCE_TYPE_CONSTANT, target='constant_target'),
    ])
    model_input = ModelInput()
    model_input.sensor = 'sensor_value'
    preprocessed = preprocessor.preprocess_merge(model_input)
    assert type(preprocessed) is bytes
    example: tf.train.Example = tf.train.Example.FromString(preprocessed)
    assert 'sensor_target' in repr(example)
    assert 'sensor_value' in repr(example)
    assert 'constant_source' in repr(example)
    assert 'constant_target' in repr(example)


def test_should_raise_on_unavailable_media_source():
    preprocessor = TfRecordsMergePreprocessor(mappings=[
        Mapping(source='video-in', source_type=SOURCE_TYPE_MEDIA, target='bad')
    ])
    model_input = ModelInput()
    model_input.sensor = 'sensor_value'
    with pytest.raises(RuntimeError) as excinfo:
        preprocessor.preprocess_merge(model_input)
    assert str(excinfo.value) == 'Media source `video-in` is not available'
