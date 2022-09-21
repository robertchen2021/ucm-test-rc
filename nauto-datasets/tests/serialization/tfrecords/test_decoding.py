from typing import NamedTuple, List
from datetime import datetime

import numpy as np
import tensorflow as tf

from nauto_datasets.serialization.tfrecords import (decoding as tf_decoding,
    encoding as tf_encoding)
from nauto_datasets.utils.numpy import NDArray


class InnerTuple(NamedTuple):
    int8_val: np.int8
    int32_val: np.int32
    float_array: NDArray[np.float32]


class OuterTuple(NamedTuple):
    float_val: np.float64
    str_val: np.str
    time_val: np.datetime64
    inner_tuple: InnerTuple
    inner_tuples: List[InnerTuple]
    arrays: List[NDArray[np.uint64]]


class TfFeaturesParsersTest(tf.test.TestCase):

    def test_drop(self):
        parsers = tf_decoding.TfFeatureParsers(
            context={
                'int/feature/a': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'int/feature/b': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'bytes/feature': tf.io.VarLenFeature(dtype=tf.string),
                'float/feature': tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
            },
            sequence={
                'int/features/a': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64),
                'int/features/b': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64),
                'bytes/features': tf.io.VarLenFeature(dtype=tf.string),
                'float/features': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
            },
            context_handlers={
                'int_feature': tf_decoding.FeatureCreator(
                    keys=['int/feature/a', 'int/feature/b'],
                    feature_producer=None),
                'bytes_feature': tf_decoding.FeatureCreator(
                    keys=['bytes/feature'],
                    feature_producer=None),
                'float_feature': tf_decoding.FeatureCreator(
                    keys=['int/feature/a', 'float/feature'],
                    feature_producer=None)
            },
            sequence_handlers={
                'int_features': tf_decoding.FeatureCreator(
                    keys=['int/features/a'],
                    feature_producer=None),
                'bytes_features': tf_decoding.FeatureCreator(
                    keys=['bytes/features'],
                    feature_producer=None),
                'float_features': tf_decoding.FeatureCreator(
                    keys=['float/features'],
                    feature_producer=None)
            })

        new_parsers = parsers.drop(
            context_features=['int_feature'],
            sequence_features=['bytes_features', 'float_features'])

        self.assertDictEqual(
            new_parsers.context_handlers,
            {
                'bytes_feature': parsers.context_handlers['bytes_feature'],
                'float_feature': parsers.context_handlers['float_feature'],
            })
        self.assertDictEqual(
            new_parsers.sequence_handlers,
            {
                'int_features': parsers.sequence_handlers['int_features']
            })

        self.assertDictEqual(
            new_parsers.context,
            {
                'int/feature/a': parsers.context['int/feature/a'],
                'bytes/feature': parsers.context['bytes/feature'],
                'float/feature': parsers.context['float/feature']
            })
        self.assertDictEqual(
            new_parsers.sequence,
            {
                'int/features/a': parsers.sequence['int/features/a']
            })

    def test_select(self):
        parsers = tf_decoding.TfFeatureParsers(
            context={
                'int/feature/a': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'int/feature/b': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                'bytes/feature': tf.io.VarLenFeature(dtype=tf.string),
                'float/feature': tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
            },
            sequence={
                'int/features/a': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64),
                'int/features/b': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64),
                'bytes/features': tf.io.VarLenFeature(dtype=tf.string),
                'float/features': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
            },
            context_handlers={
                'int_feature': tf_decoding.FeatureCreator(
                    keys=['int/feature/a', 'int/feature/b'],
                    feature_producer=None),
                'bytes_feature': tf_decoding.FeatureCreator(
                    keys=['bytes/feature'],
                    feature_producer=None),
                'float_feature': tf_decoding.FeatureCreator(
                    keys=['int/feature/a', 'float/feature'],
                    feature_producer=None)
            },
            sequence_handlers={
                'int_features': tf_decoding.FeatureCreator(
                    keys=['int/features/a'],
                    feature_producer=None),
                'bytes_features': tf_decoding.FeatureCreator(
                    keys=['bytes/features'],
                    feature_producer=None),
                'float_features': tf_decoding.FeatureCreator(
                    keys=['float/features'],
                    feature_producer=None)
            })

        select_all = parsers.select(None, None)
        self.assertDictEqual(
            select_all.context_handlers, parsers.context_handlers)
        self.assertDictEqual(
            select_all.sequence_handlers, parsers.sequence_handlers)
        self.assertDictEqual(
            select_all.context, parsers.context)

        seq = parsers.sequence.copy()
        del seq['int/features/b']
        self.assertDictEqual(
            select_all.sequence, seq)

        select_empty = parsers.select([], [])
        self.assertEqual(len(select_empty.context_handlers), 0)
        self.assertEqual(len(select_empty.sequence_handlers), 0)
        self.assertEqual(len(select_empty.context), 0)
        self.assertEqual(len(select_empty.sequence), 0)

        new_parsers = parsers.select(
            context_features=['int_feature'],
            sequence_features=['bytes_features', 'float_features'])

        self.assertDictEqual(
            new_parsers.context_handlers,
            {
                'int_feature': parsers.context_handlers['int_feature']
            })
        self.assertDictEqual(
            new_parsers.sequence_handlers,
            {
                'bytes_features': parsers.sequence_handlers['bytes_features'],
                'float_features': parsers.sequence_handlers['float_features']
            })

        self.assertDictEqual(
            new_parsers.context,
            {
                'int/feature/a': parsers.context['int/feature/a'],
                'int/feature/b': parsers.context['int/feature/b']
            })
        self.assertDictEqual(
            new_parsers.sequence,
            {
                'bytes/features': parsers.sequence['bytes/features'],
                'float/features': parsers.sequence['float/features']
            })


class TestDecoding(tf.test.TestCase):

    def test_nested_type_to_context_parsers_single_values(self):
        class Int32(NamedTuple):
            val: np.int32

        parsers = tf_decoding.nested_type_to_feature_parsers(
            Int32, parse_tensor_protos=False)
        self.assertEqual(len(parsers.sequence), 0)
        self.assertEqual(len(parsers.sequence_handlers), 0)

        int32_desc = parsers.context['val']
        self.assertIsInstance(int32_desc, tf.io.FixedLenFeature)
        self.assertEqual(int32_desc.shape, ())
        self.assertEqual(int32_desc.dtype, tf.int64)

        # number serialized as TensorProto
        parsers = tf_decoding.nested_type_to_feature_parsers(
            Int32, parse_tensor_protos=True)
        int32_desc = parsers.context['val']
        self.assertIsInstance(int32_desc, tf.io.FixedLenFeature)
        self.assertEqual(int32_desc.shape, ())
        self.assertEqual(int32_desc.dtype, tf.string)

        class Datetime64(NamedTuple):
            val: np.datetime64

        parsers = tf_decoding.nested_type_to_feature_parsers(
            Datetime64, parse_tensor_protos=False)
        int64_desc = parsers.context['val']
        self.assertIsInstance(int64_desc, tf.io.FixedLenFeature)
        self.assertEqual(int64_desc.shape, ())
        self.assertEqual(int64_desc.dtype, tf.int64)

        class Uint16Arr(NamedTuple):
            val: NDArray[np.uint16]

        # regular features
        parsers = tf_decoding.nested_type_to_feature_parsers(
            Uint16Arr, parse_tensor_protos=False)
        np_array_desc = parsers.context['val']
        self.assertIsInstance(np_array_desc, tf.io.VarLenFeature)
        self.assertEqual(np_array_desc.dtype, tf.int64)

        # tensor protos
        parsers = tf_decoding.nested_type_to_feature_parsers(
            Uint16Arr, parse_tensor_protos=True)
        np_array_desc = parsers.context['val']
        self.assertIsInstance(np_array_desc, tf.io.FixedLenFeature)
        self.assertEqual(np_array_desc.dtype, tf.as_dtype(tf.string))

        test_array = np.arange(20, dtype=np.uint16)
        test_array_proto = tf.io.serialize_tensor(test_array).numpy()
        np_array_producer = parsers.context_handlers['val'].feature_producer
        self.assertEqual(parsers.context_handlers['val'].keys, ['val'])
        array_tensor = np_array_producer(test_array_proto)
        self.assertEqual(array_tensor.dtype, tf.uint16)


    def test_nested_type_to_context_parsers_invalid_types(self):
        with self.assertRaises(TypeError):
            tf_decoding.nested_type_to_feature_parsers(np.ndarray)

        with self.assertRaises(TypeError):
            tf_decoding.nested_type_to_feature_parsers(dict)

    def test_nested_type_to_feature_parsers_flat_named_tuple(self):
        parsers = tf_decoding.nested_type_to_feature_parsers(
            InnerTuple,
            parse_tensor_protos=False)
        self.assertSetEqual(
            set(parsers.context.keys()),
            set(('int8_val', 'int32_val', 'float_array')))
        self.assertSetEqual(
            set(parsers.context_handlers.keys()),
            set(('int8_val', 'int32_val', 'float_array')))
        inner_tuple_descs = parsers.context

        self.assertIsInstance(inner_tuple_descs['int8_val'], tf.io.FixedLenFeature)
        self.assertEqual(inner_tuple_descs['int8_val'].shape, ())
        self.assertEqual(inner_tuple_descs['int8_val'].dtype, tf.int64)

        self.assertIsInstance(inner_tuple_descs['int32_val'], tf.io.FixedLenFeature)
        self.assertEqual(inner_tuple_descs['int32_val'].shape, ())
        self.assertEqual(inner_tuple_descs['int32_val'].dtype, tf.int64)

        self.assertIsInstance(
            inner_tuple_descs['float_array'],
            tf.io.VarLenFeature)
        self.assertEqual(
            inner_tuple_descs['float_array'].dtype, tf.float32)

    def test_nested_type_to_context_parsers_flat_named_tuple_as_protos(self):
        parsers = \
            tf_decoding.nested_type_to_feature_parsers(
                InnerTuple,
                parse_tensor_protos=True)
        inner_tuple_descs = parsers.context
        inner_tuple_producers = parsers.context_handlers

        self.assertSetEqual(
            set(inner_tuple_descs.keys()),
            set(('int8_val', 'int32_val', 'float_array')))
        self.assertSetEqual(
            set(inner_tuple_producers.keys()),
            set(('int8_val', 'int32_val', 'float_array')))

        self.assertIsInstance(inner_tuple_descs['int8_val'], tf.io.FixedLenFeature)
        self.assertEqual(inner_tuple_descs['int8_val'].shape, ())
        self.assertEqual(inner_tuple_descs['int8_val'].dtype, tf.string)

        self.assertIsInstance(inner_tuple_descs['int32_val'], tf.io.FixedLenFeature)
        self.assertEqual(inner_tuple_descs['int32_val'].shape, ())
        self.assertEqual(inner_tuple_descs['int32_val'].dtype, tf.string)

        self.assertIsInstance(
            inner_tuple_descs['float_array'],
            tf.io.FixedLenFeature)
        self.assertEqual(
            inner_tuple_descs['float_array'].dtype, tf.string)

    def test_nested_type_to_parsers_single_list_values(self):

        class Int32List(NamedTuple):
            val_list: List[np.int32]

        parsers = tf_decoding.nested_type_to_feature_parsers(
            Int32List,
            parse_tensor_protos=False)
        self.assertEqual(len(parsers.context), 0)
        self.assertEqual(len(parsers.context_handlers), 0)

        int32_desc = parsers.sequence['val_list']
        self.assertIsInstance(int32_desc, tf.io.FixedLenSequenceFeature)
        self.assertEqual(int32_desc.shape, ())
        self.assertEqual(int32_desc.dtype, np.int64)

        class Uint64ArrList(NamedTuple):
            val_list: List[NDArray[np.uint64]]

        parsers = tf_decoding.nested_type_to_feature_parsers(Uint64ArrList)
        self.assertEqual(len(parsers.context), 0)
        self.assertEqual(len(parsers.context_handlers), 0)
        np_array_desc = parsers.sequence['val_list']
        self.assertIsInstance(np_array_desc, tf.io.VarLenFeature)
        self.assertEqual(np_array_desc.dtype, tf.int64)

    def test_nested_type_to_sequence_parsers_nested_lists(self):
        class InnerList(NamedTuple):
            val: List[int]

        class OuterList(NamedTuple):
            val: List[InnerList]

        tf_decoding.nested_type_to_feature_parsers(OuterList)

    def test_nested_named_tuple_with_lists_to_feature_parsers(self):
        outer_tuple_parsers = tf_decoding.nested_type_to_feature_parsers(
            OuterTuple,
            parse_tensor_protos=False)

        # context
        self.assertSetEqual(
            set(outer_tuple_parsers.context.keys()),
            set(('float_val',
                 'str_val',
                 'time_val',
                 'inner_tuple/int8_val',
                 'inner_tuple/int32_val',
                 'inner_tuple/float_array')))
        self.assertSetEqual(
            set(outer_tuple_parsers.context_handlers.keys()),
            set(('float_val',
                 'str_val',
                 'time_val',
                 'inner_tuple/int8_val',
                 'inner_tuple/int32_val',
                 'inner_tuple/float_array')))
        self.assertIsInstance(
            outer_tuple_parsers.context['float_val'], tf.io.FixedLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['float_val'].shape, ())
        self.assertEqual(
            outer_tuple_parsers.context['float_val'].dtype, tf.float32)

        self.assertIsInstance(
            outer_tuple_parsers.context['str_val'],
            tf.io.FixedLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['str_val'].dtype, tf.string)

        self.assertIsInstance(
            outer_tuple_parsers.context['time_val'], tf.io.FixedLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['time_val'].shape, ())
        self.assertEqual(
            outer_tuple_parsers.context['time_val'].dtype, tf.int64)

        self.assertIsInstance(
            outer_tuple_parsers.context['inner_tuple/int8_val'],
            tf.io.FixedLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['inner_tuple/int8_val'].shape,
            ())
        self.assertEqual(
            outer_tuple_parsers.context['inner_tuple/int8_val'].dtype,
            tf.int64)

        self.assertIsInstance(
            outer_tuple_parsers.context['inner_tuple/int32_val'],
            tf.io.FixedLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['inner_tuple/int32_val'].shape,
            ())
        self.assertEqual(
            outer_tuple_parsers.context['inner_tuple/int32_val'].dtype,
            tf.int64)

        self.assertIsInstance(
            outer_tuple_parsers.context['inner_tuple/float_array'],
            tf.io.VarLenFeature)
        self.assertEqual(
            outer_tuple_parsers.context['inner_tuple/float_array'].dtype,
            tf.float32)

        # sequence
        self.assertSetEqual(
            set(outer_tuple_parsers.sequence.keys()),
            set(('inner_tuples/int8_val',
                 'inner_tuples/int32_val',
                 'inner_tuples/float_array',
                 'arrays')))
        self.assertSetEqual(
            set(outer_tuple_parsers.sequence_handlers.keys()),
            set(('inner_tuples/int8_val',
                 'inner_tuples/int32_val',
                 'inner_tuples/float_array',
                 'arrays')))

        self.assertIsInstance(
            outer_tuple_parsers.sequence['inner_tuples/int8_val'],
            tf.io.FixedLenSequenceFeature)
        self.assertEqual(
            outer_tuple_parsers.sequence['inner_tuples/int8_val'].shape,
            ())
        self.assertEqual(
            outer_tuple_parsers.sequence['inner_tuples/int8_val'].dtype,
            tf.int64)

        self.assertIsInstance(
            outer_tuple_parsers.sequence['inner_tuples/int32_val'],
            tf.io.FixedLenSequenceFeature)
        self.assertEqual(
            outer_tuple_parsers.sequence['inner_tuples/int32_val'].shape,
            ())
        self.assertEqual(
            outer_tuple_parsers.sequence['inner_tuples/int32_val'].dtype,
            tf.int64)

        self.assertIsInstance(
            outer_tuple_parsers.sequence['inner_tuples/float_array'],
            tf.io.VarLenFeature)
        self.assertEqual(
            outer_tuple_parsers.sequence['inner_tuples/float_array'].dtype,
            tf.float32)

        self.assertIsInstance(
            outer_tuple_parsers.sequence['arrays'],
            tf.io.VarLenFeature)
        self.assertEqual(
            outer_tuple_parsers.sequence['arrays'].dtype, tf.int64)
