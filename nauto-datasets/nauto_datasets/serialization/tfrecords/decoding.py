import logging
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

import numpy as np
import tensorflow as tf

import typing_inspect
from nauto_datasets.utils import dicts as dict_utils
from nauto_datasets.utils import numpy as np_utils

"""Function producing a single tensor given multiple tensor inputs"""
FeatureProducerFn = Callable[..., Union[tf.Tensor, tf.SparseTensor]]


def tensor_proto_parser_producer(dt: tf.DType) -> FeatureProducerFn:
    """Returns feature producer parsing feature serialized as TensorProto"""
    def parse_feature(feature: tf.Tensor) -> tf.Tensor:
        return tf.io.parse_tensor(feature, out_type=dt)
    return parse_feature


def cast_to_dense_tensor_producer(dt: tf.DType) -> FeatureProducerFn:
    """Returns feature producer casting sparse tensor to a dense counterpart"""
    def parse_feature(feature: tf.SparseTensor) -> tf.Tensor:
        default_value = '' if dt == tf.string else 0
        return tf.cast(
            tf.sparse.to_dense(feature, default_value),
            dt)

    return parse_feature


def cast_tensor_producer(dt: tf.DType) -> FeatureProducerFn:
    """Returns feature producer casting tensor to a different type"""
    def parse_feature(
            feature: Union[tf.SparseTensor, tf.Tensor]
    ) -> Union[tf.Tensor, tf.SparseTensor]:
        return tf.cast(feature, dt)
    return parse_feature


class FeatureCreator(NamedTuple):
    """
    Attributes:
        keys: a list of names of the features from the serialized example
            that need to be provided to `feature_producer` callable
        feature_producer: a callable taking a parsed features tensors
            for `keys` and returning tensor representing a single feature
    """
    keys: List[str]
    feature_producer: FeatureProducerFn


FeaturesDict = Dict[str, Union[tf.Tensor, tf.SparseTensor]]

ContextFeatureDescription = Union[
    tf.io.VarLenFeature, tf.io.FixedLenFeature]
SequenceFeatureDescription = Union[
    tf.io.VarLenFeature, tf.io.FixedLenSequenceFeature]
FeatureDescription = Union[
    ContextFeatureDescription, SequenceFeatureDescription]


def parse_features(
        features: FeaturesDict,
        handlers: Dict[str, FeatureCreator]
) -> FeaturesDict:
    return {
        feature_name: feature_creator.feature_producer(
            *[features[name] for name in feature_creator.keys])
        for feature_name, feature_creator in handlers.items()
    }


class TfFeatureParsers(NamedTuple):
    """Represents dictionaries of features descriptions used when
    parsing @{tf.train.Example} or @{tf.train.SequenceExample}

    For more information, please look up:
    https://www.tensorflow.org/api_docs/python/tf/parse_single_example
    https://www.tensorflow.org/api_docs/python/tf/parse_single_sequence_example

    Attributes:
        context: a dictionary of features descriptions for regular
            @{tf.train.Example}'s or for `context` field in
            @{tf.train.SequenceExample}
        sequence: a dictionary of features descriptions for `feature_lists`
            @{tf.train.SequenceExample}
        context_handlers: a mapping from the produced feature names to the
            relevant feature creators for context feautres
        sequence_handlers: a mapping from the produced feature names to the
            relevant feature creators for sequence feautres
    """
    context: Dict[str, ContextFeatureDescription]
    sequence: Dict[str, SequenceFeatureDescription]

    context_handlers: Dict[str, FeatureCreator]
    sequence_handlers: Dict[str, FeatureCreator]

    def parse_example(self, example_tensor) -> FeaturesDict:
        """Parses a serialized @{tf.train.Example} returning a dictionary of
        tensors
        """
        context_tensors = tf.io.parse_single_example(
            example_tensor, features=self.context)
        return parse_features(context_tensors, self.context_handlers)

    def parse_sequence_example(
            self,
            example_tensor: tf.Tensor,
    ) -> Tuple[FeaturesDict, FeaturesDict]:
        """Parses a serialized @{tf.train.SequenceExample} returning two
        dictionaries with context and sequence tensors.
        """
        context_tensors, sequence_tensors = tf.io.parse_single_sequence_example(
            example_tensor,
            context_features=self.context,
            sequence_features=self.sequence)

        return (
            parse_features(context_tensors, self.context_handlers),
            parse_features(sequence_tensors, self.sequence_handlers)
        )

    def with_prefix(
            self,
            prefix: str,
            prefix_produced_features: bool = True
    ) -> 'TfFeatureParsers':

        feature_prefix = prefix + '/' if prefix_produced_features else ''
        context_handlers = {
            f'{feature_prefix}{feature_name}': feature_creator._replace(
                keys=[f'{prefix}/{name}' for name in feature_creator.keys])
            for feature_name, feature_creator in self.context_handlers.items()
        }
        sequence_handlers = {
            f'{feature_prefix}{feature_name}': feature_creator._replace(
                keys=[f'{prefix}/{name}' for name in feature_creator.keys])
            for feature_name, feature_creator in self.sequence_handlers.items()
        }

        return TfFeatureParsers(
            context={
                f'{prefix}/{name}': val
                for name, val in self.context.items()
            },
            sequence={
                f'{prefix}/{name}': val
                for name, val in self.sequence.items()
            },
            context_handlers=context_handlers,
            sequence_handlers=sequence_handlers)

    def drop(self,
             context_features: List[str],
             sequence_features: List[str]) -> 'TfFeatureParsers':
        """Drops context and sequence features parsers"""
        ctx_features = set(context_features)
        seq_features = set(sequence_features)

        new_ctx_handlers = dict_utils.filter_dict(
            self.context_handlers, lambda k, v: k not in ctx_features)
        new_seq_handlers = dict_utils.filter_dict(
            self.sequence_handlers, lambda k, v: k not in seq_features)

        used_ctx_descs = {
            key
            for creator in new_ctx_handlers.values()
            for key in creator.keys
        }
        used_seq_descs = {
            key
            for creator in new_seq_handlers.values()
            for key in creator.keys
        }

        new_context = dict_utils.filter_dict(
            self.context, lambda k, v: k in used_ctx_descs)
        new_sequence = dict_utils.filter_dict(
            self.sequence, lambda k, v: k in used_seq_descs)

        return TfFeatureParsers(
            context=new_context,
            sequence=new_sequence,
            context_handlers=new_ctx_handlers,
            sequence_handlers=new_seq_handlers)

    def select(self,
               context_features: Optional[List[str]] = None,
               sequence_features: Optional[List[str]] = None
               ) -> 'TfFeatureParsers':
        ctx_features = set(self.context_handlers) \
            if context_features is None \
            else set(context_features)
        seq_features = set(self.sequence_handlers) \
            if sequence_features is None \
            else set(sequence_features)
        ctx_to_drop = [
            name for name in self.context_handlers if name not in ctx_features
        ]
        seq_to_drop = [
            name for name in self.sequence_handlers if name not in seq_features
        ]
        return self.drop(ctx_to_drop, seq_to_drop)


def _dtype_to_proto_and_dest_type(dt: np.dtype) -> tf.DType:
    if dt.kind == 'u':
        if dt in (np.uint32, np.uint64):
            logging.warning(
                'Casting between unsigned 32 or 64-bit tensors is not supported. '
                'Leaving tf.int64 as destination type')
            return tf.int64, tf.int64
        else:
            return tf.int64, tf.as_dtype(dt)
    if dt.kind in ('i', 'b'):
        return tf.int64, tf.as_dtype(dt)
    elif dt.kind == 'f':
        return tf.float32, tf.as_dtype(dt)
    elif dt.kind in ('U', 'S', 'a'):
        return tf.string, tf.string
    elif dt.kind == 'M':
        # datetime is serialized as int64 nanoseconds
        return tf.int64, tf.int64
    else:
        raise TypeError(f'Unsupported dtype kind {dt.kind}')


class _ParsersAndProducers(NamedTuple):
    context: Dict[str, ContextFeatureDescription]
    sequence: Dict[str, SequenceFeatureDescription]
    context_producers: Dict[str, FeatureProducerFn]
    sequence_producers: Dict[str, FeatureProducerFn]


def _recursive_nested_type_to_feature_parsers(
        type_t: Any,
        parse_tensor_protos: bool = False,
        ignore_sequence_features: bool = False,
        in_sequence_mode: bool = False
) -> _ParsersAndProducers:

    def build_feature_parsers(parsers, producers) -> _ParsersAndProducers:
        if in_sequence_mode:
            return _ParsersAndProducers(
                context={},
                sequence=parsers,
                context_producers={},
                sequence_producers=producers)
        else:
            return _ParsersAndProducers(
                context=parsers,
                sequence={},
                context_producers=producers,
                sequence_producers={})

    if hasattr(type_t, '_field_types'):
        # type_t is a NamedTuple
        parsers_dict = {}
        for field_name, field_t in type_t._field_types.items():
            if typing_inspect.get_origin(field_t) in (list, List):
                while typing_inspect.get_origin(field_t.__args__[0]) in (list, List):
                    field_t = field_t.__args__[0]
                inner_type = field_t.__args__[0]  # List[D] -> D
                parsers_dict[field_name] = _recursive_nested_type_to_feature_parsers(
                    inner_type,
                    parse_tensor_protos=False,
                    ignore_sequence_features=ignore_sequence_features,
                    in_sequence_mode=True)
            else:
                parsers_dict[field_name] = _recursive_nested_type_to_feature_parsers(
                    field_t,
                    parse_tensor_protos=parse_tensor_protos,
                    ignore_sequence_features=ignore_sequence_features,
                    in_sequence_mode=in_sequence_mode)
        context, sequence, context_producers, sequence_producers = \
            dict_utils.unzip_dict(parsers_dict)
        return _ParsersAndProducers(
            context=dict_utils.flatten_nested_dict(context),
            sequence=dict_utils.flatten_nested_dict(sequence),
            context_producers=dict_utils.flatten_nested_dict(context_producers),
            sequence_producers=dict_utils.flatten_nested_dict(sequence_producers))

    elif type_t == np.ndarray:
        # type_t should not be a regular np.ndarray
        raise TypeError(
            'type_t cannot be equal to np.ndarray - no dtype info. '
            'Please use NDarray[D]')
    elif np_utils.is_ndarray_type(type_t):
        # type_t is NDarray[D]
        und_type = np_utils.get_underlying_dtype(type_t)
        if und_type.kind == 'O':
            raise TypeError(
                f'numpy arrays of dtype: {type_t} are not supported')
        proto_type, dest_type = _dtype_to_proto_and_dest_type(und_type)
        if in_sequence_mode:
            return build_feature_parsers(
                tf.io.VarLenFeature(dtype=proto_type),
                cast_tensor_producer(dest_type))
        else:
            if parse_tensor_protos:
                return build_feature_parsers(
                    tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    tensor_proto_parser_producer(dest_type))
            else:
                return build_feature_parsers(
                    tf.io.VarLenFeature(dtype=proto_type),
                    cast_to_dense_tensor_producer(dest_type)
                )
    elif np.dtype(type_t).kind == 'O':
        # type_t is something else but not scalar
        raise TypeError(f'type_t = {type_t} is not supported')
    else:
        # type_t is a scalar value
        proto_dtype, dest_type = _dtype_to_proto_and_dest_type(
            np.dtype(type_t))
        if in_sequence_mode:
            return build_feature_parsers(
                tf.io.FixedLenSequenceFeature(
                    shape=(), dtype=proto_dtype, allow_missing=True),
                cast_tensor_producer(dest_type))
        else:
            if parse_tensor_protos:
                return build_feature_parsers(
                    tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    tensor_proto_parser_producer(dest_type))
            else:
                return build_feature_parsers(
                    tf.io.FixedLenFeature(shape=(), dtype=proto_dtype),
                    cast_tensor_producer(dest_type))


def nested_type_to_feature_parsers(
        tuple_t: Tuple[NamedTuple],
        parse_tensor_protos: bool = False,
        ignore_sequence_features: bool = False
) -> TfFeatureParsers:
    """Returns `TfFeatureParsers` generated from the type like `NamedTuple`,
    which has `_field_types` attribute.

    All the List[...] fields of `tuple_t` will produce sequence feature
    descriptions. Other types of fields will result in context features
    descriptions.

    Example:
    ```
    class InnerType(NamedTuple):
       float_field: float

    class SomeType(NamedTuple):
      int_field: int
      inner_type: InnerType
      inner_types: List[InnerType]
      arrays: List[NDarray[np.float]]

    nested_type_to_feature_parsers(SomeType).context == {
      'int_field': <context feature desc>
      'inner_type/float_field': <context feature desc>
    }

    nested_type_to_feature_parsers(SomeType).sequence == {
      'inner_types/float_field': <sequence feature desc>
      'arrays': <sequence feature desc>
    }
    ```

    Args:
        tuple_t: is a well types NamedTuple with List fields or fields
            adhering to the same requirements as in
            `nested_type_to_feature_parsers` function
        ignore_sequence_features: when True, then resulting
            sequence features dictionary will be empty
        parse_tensor_protos: if true then all serialized **context** features will be
            treated as bytes lists with TensorProtos
    Returns:
        feature_parsers: mapping paths to feature descriptions
    Raises:
        TypeError: when `tuple_t` contains unsupported types
        TypeError: when `type_t` or one of its nested fields are not scalar value,
            `NDArray`'s or `NamedTuple`'s
        TypeError: when detected nested lists - lists which contain structers with
            other lists
    """
    if not hasattr(tuple_t, '_field_types'):
        raise TypeError('tuple_t does not represent a typed NamedTuple')

    parsers_and_producers = _recursive_nested_type_to_feature_parsers(
        tuple_t,
        parse_tensor_protos=parse_tensor_protos,
        ignore_sequence_features=ignore_sequence_features,
        in_sequence_mode=False)

    context, sequence, context_producers, sequence_producers = \
        parsers_and_producers

    context_handlers = {
        key: FeatureCreator(keys=[key], feature_producer=producer)
        for key, producer in context_producers.items()
    }
    sequence_handlers = {
        key: FeatureCreator(keys=[key], feature_producer=producer)
        for key, producer in sequence_producers.items()
    }
    return TfFeatureParsers(context=context,
                            sequence=sequence,
                            context_handlers=context_handlers,
                            sequence_handlers=sequence_handlers)
