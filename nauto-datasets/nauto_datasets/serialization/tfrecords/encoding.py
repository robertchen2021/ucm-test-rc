from typing import Any, Dict, List, NamedTuple, Union

import numpy as np
import tensorflow as tf

from nauto_datasets.utils import dicts as dict_utils
from nauto_datasets.utils import numpy as np_utils

FeaturesDict = Dict[str, tf.train.Feature]
FeatureListsDict = Dict[str, List[tf.train.Feature]]


class TfFeatures(NamedTuple):
    """Tuple representing a structure of tensorflow record features,
    which can be directly translated to @{tf.train.SequenceExample}
    """
    context: FeaturesDict
    feature_lists: FeatureListsDict

    def to_sequence_example(self) -> tf.train.SequenceExample:
        """Transforms this named tuple to an instance of SequenceExample"""
        return tf.train.SequenceExample(
            context=tf.train.Features(feature=self.context),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    name: tf.train.FeatureList(feature=f_list)
                    for name, f_list in self.feature_lists.items()
                }))

    def to_example(self) -> tf.train.Example:
        """Transforms only the `self.context` field to Example"""
        return tf.train.Example(
            features=tf.train.Features(feature=self.context))

    def with_prefix(self, prefix: str) -> 'TfFeatures':
        """Adds `prefix` to each key name in both `context`
           and `feature_lists`

        Returns:
            a new instance of `TfFeatures` with prefixed names
        """
        return TfFeatures(
            context={
                f'{prefix}/{name}': val for name, val in
                self.context.items()
            },
            feature_lists={
                f'{prefix}/{name}': val for name, val in
                self.feature_lists.items()
            })


def bytes_scalar_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_array_feature(values: List[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def int64_scalar_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_array_feature(
        values: Union[List[int], np.ndarray]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_scalar_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_array_feature(
        values: Union[List[float], np.ndarray]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def array_to_feature(array: np.ndarray) -> tf.train.Feature:
    """Translates a numpy array into Feature

    Raises:
        ValueError: when array is not 1-dimensional
        TypeError: when array's kind is not in (i, u, f, S, U, M)
    """
    if array.ndim != 1:
        raise ValueError('array should be one-dimensional')
    if array.dtype.kind in ('i', 'u', 'b'):
        return int64_array_feature(array.astype(np.int64))
    elif array.dtype.kind == 'f':
        return float_array_feature(array.astype(np.float32))
    elif array.dtype.kind in ('S', 'a'):
        return bytes_array_feature(array)
    elif array.dtype.kind == 'U':
        return bytes_array_feature(array.astype(np.bytes_))
    elif array.dtype.kind == 'M':
        # encode datetime as nanoseconds
        return int64_array_feature(np_utils.datetime64_to_nano_seconds(array))
    else:
        raise TypeError(f'arrays kind: {array.dtype.kind} is not supported')


def scalar_to_feature(value: Any) -> tf.train.Feature:
    """Translates a scalar into Feature

    Raises:
        TypeError: when value's dtype kind is not in (i, u, f, S, U, M)
    """
    dt = np.dtype(type(value))
    if dt.kind in ('i', 'u', 'b'):
        return int64_scalar_feature(np.int64(value))
    elif dt.kind == 'f':
        return float_scalar_feature(np.float32(value))
    elif dt.kind in ('S', 'a'):
        return bytes_scalar_feature(value)
    elif dt.kind == 'U':
        return bytes_scalar_feature(value.encode())
    elif dt.kind == 'M':
        # encode datetime as nanoseconds
        return int64_scalar_feature(np_utils.datetime64_to_nano_seconds(value))
    else:
        raise TypeError(f'value type: {type(value)} is not supported')


def array_to_proto_feature(array: np.ndarray) -> tf.train.Feature:
    """Creates a @{tf.train.Feature} by serializing array to TensorProto

    Note! datetime arrays are cast to int64 nanoseonds
    Note! uint32 and uint64 arrays are cast to int64, because of lacking
          support for these types from tensorflow

    Args:
        value: numpy array
    Returns:
        `tf.train.Feature` with non-empty bytes_list field containing
        a serialized `TensorProto`
    Raises:
        TypeError: when `array` cannot be packed in TensorProto
    """
    if array.dtype.kind == 'M':
        # necessary of datetime
        array = np_utils.datetime64_to_nano_seconds(array)
    elif array.dtype in (np.uint32, np.uint64):
        array = array.astype(np.int64)

    arr_proto = tf.io.serialize_tensor(array).numpy()
    return scalar_to_feature(arr_proto)


def scalar_to_proto_feature(value: Any) -> tf.train.Feature:
    """Creates a @{tf.train.Feature} by serializing scalar to TensorProto

    Note! datetime values are cast to int64 nanoseonds
    Note! uint32 and uint64 values are cast to int64, because of lacking
          support for these types from tensorflow

    Args:
        value: should be a single scalar value
    Returns:
        `tf.train.Feature` with non-empty bytes_list field containing
        a serialized `TensorProto`
    Raises:
        TypeError: when `value` cannot be packed in TensorProto
    """
    dtype = np.dtype(type(value))
    if dtype.kind == 'M':
        # necessary of datetime
        value = np_utils.datetime64_to_nano_seconds(value)
    elif dtype in (np.uint32, np.uint64):
        value = np.int64(value)

    arr_proto = tf.io.serialize_tensor(value).numpy()
    return scalar_to_feature(arr_proto)


NestedFeatures = Union[tf.train.Feature, Dict[str, Any]]


def _recursive_structure_to_features(
        value: Any,
        use_tensor_protos: bool = False,
        ignore_sequence_features: bool = False,
        in_sequence_mode: bool = False
) -> TfFeatures:
    def build_tffeatures(value: Any) -> TfFeatures:
        if in_sequence_mode:
            if ignore_sequence_features:
                return TfFeatures({}, {})
            else:
                return TfFeatures(context={}, feature_lists=value)
        else:
            return TfFeatures(context=value, feature_lists={})

    if isinstance(value, np.ndarray):
        if use_tensor_protos:
            return build_tffeatures(array_to_proto_feature(value))
        else:
            return build_tffeatures(array_to_feature(value))

    elif hasattr(value, '_asdict'):
        return _recursive_structure_to_features(
            value._asdict(),
            use_tensor_protos=use_tensor_protos,
            ignore_sequence_features=ignore_sequence_features,
            in_sequence_mode=in_sequence_mode)

    elif isinstance(value, dict):
        features_dict = {}
        for name, element in value.items():
            if isinstance(element, list):
                if in_sequence_mode:
                    raise TypeError('Detected nested lists')
                if ignore_sequence_features:
                    continue
                tffeatures_list = [
                    _recursive_structure_to_features(
                        e,
                        use_tensor_protos=False,
                        ignore_sequence_features=ignore_sequence_features,
                        in_sequence_mode=True
                    ).feature_lists
                    for e in element
                ]
                if len(tffeatures_list) > 0:
                    if not isinstance(tffeatures_list[0], dict):
                        features_dict[name] = TfFeatures(
                            context={},
                            feature_lists=tffeatures_list)
                    else:
                        features_dict[name] = TfFeatures(
                            context={},
                            feature_lists=dict_utils.concat_dicts(
                                tffeatures_list))
                else:
                    continue
            else:
                features_dict[name] = _recursive_structure_to_features(
                    element,
                    use_tensor_protos=use_tensor_protos,
                    ignore_sequence_features=ignore_sequence_features,
                    in_sequence_mode=in_sequence_mode)

        context, feature_lists = dict_utils.unzip_dict(features_dict)
        return TfFeatures(
            context=dict_utils.flatten_nested_dict(context),
            feature_lists=dict_utils.flatten_nested_dict(feature_lists))

    # assume scalars
    elif use_tensor_protos:
        return build_tffeatures(scalar_to_proto_feature(value))
    else:
        return build_tffeatures(scalar_to_feature(value))


def structure_to_features(
        value: Any,
        use_tensor_protos: bool = False,
        ignore_sequence_features: bool = False,
) -> TfFeatures:
    """Transforms a potentially nested structure (dict, named tuple) of values to
    `TfFeatures`

    Resulting @{TfFeatures} will have dictionaries with keys being in the form of
    a paths to the value in questions, e.g.:
    ```
    class InnerTuple(NamedTuple):
         string_val: str

    class SomeTuple(NamedTuple):
        int_val: int
        inner_tuple: InnerTuple
        inner_tuples: List[InnerTuple]
        arrays: List[np.ndarray]
    ```
    Encoding SomeTuple will result in TfFeatures, where:
        - keys in the `context` are equal to
          {'int_val', 'inner_tuple/string_val'}`
        - keys in the `feature_lists` are equal to
          {'inner_tuples/string_val', 'arrays'}`

    Args:
        value: can be a dictionary or namedtuple with fields of types:
            - scalar value (float, int, bytes, str)
            - numpy array of relevant kinds
            - dictionary str -> value
            - named tuple of values
            - list of values
            These fields should also fulfill the same constraints recursively.
        use_tensor_protos: when True the data will be wrapped
            wrapped in TensorProto before creating serialized
            features. This avoids the problems of casting to other dtypes
            of int64, float or bytes and makes it simpler to serialize
            multidimensional arrays.
            NOTE: when using protos, tensorflow eager execution has to be enabled!
        ignore_sequence_features: when True `feature_lists` dictionary will be empty
    Returns:
        TfFeatures representation of the structure
    Raises:
        TypeError: when some value type is not supported
        TypeError: when detected nested lists - lists which contain structers with
            other lists
    """
    if use_tensor_protos:
        assert tf.executing_eagerly(), \
            'In order to serialize the structure to tensor protos, ' \
            'eager execution has to be enabled.'
    if not (isinstance(value, dict) or hasattr(value, '_asdict')):
        raise TypeError('value should be either dictionary or NamedTuple')

    return _recursive_structure_to_features(
        value,
        use_tensor_protos=use_tensor_protos,
        ignore_sequence_features=ignore_sequence_features,
        in_sequence_mode=False)
