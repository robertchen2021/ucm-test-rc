from datetime import datetime
from typing import Any, Dict, List, NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto

import nauto_datasets.serialization.tfrecords.encoding as tf_encoding
from nauto_datasets.utils.numpy import NDArray, datetime64_to_nano_seconds

tf.compat.v1.enable_eager_execution()


class TfFeaturesTest(tf.test.TestCase):

    @staticmethod
    def get_test_tf_features():
        context = {
            'int_scalar': tf_encoding.int64_scalar_feature(44),
            'int_array': tf_encoding.int64_array_feature(np.arange(5))
        }
        feature_lists = {
            'names': [
                tf_encoding.bytes_scalar_feature(name)
                for name in [b'Janusz', b'Andrzej']
            ],
            'ratings': [
                tf_encoding.int64_array_feature(rates)
                for rates in (
                        np.array([3, 9, 1]),
                        np.array([2, 5]))
            ]
        }
        return tf_encoding.TfFeatures(
            context=context, feature_lists=feature_lists)

    def test_with_prefix(self):
        test_features = self.get_test_tf_features()
        prefixed_features = test_features.with_prefix('prefix_name')
        self.assertSetEqual(
            set(prefixed_features.context.keys()),
            set(('prefix_name/int_scalar', 'prefix_name/int_array')))

        self.assertSetEqual(
            set(prefixed_features.feature_lists.keys()),
            set(('prefix_name/names', 'prefix_name/ratings')))

    def test_to_example(self):
        test_features = self.get_test_tf_features()
        example = test_features.to_example()
        self.assertSetEqual(
            set(example.features.feature.keys()),
            set(('int_scalar', 'int_array')))

        self.assertAllEqual(
            example.features.feature['int_scalar'].int64_list.value,
            [44])
        self.assertAllEqual(
            example.features.feature['int_array'].int64_list.value,
            np.arange(5))

    def test_to_sequence_example(self):
        test_features = self.get_test_tf_features()
        example = test_features.to_sequence_example()
        self.assertSetEqual(
            set(example.context.feature.keys()),
            set(('int_scalar', 'int_array')))

        self.assertAllEqual(
            example.context.feature['int_scalar'].int64_list.value,
            [44])
        self.assertAllEqual(
            example.context.feature['int_array'].int64_list.value,
            np.arange(5))

        self.assertSetEqual(
            set(example.feature_lists.feature_list.keys()),
            set(('names', 'ratings')))

        self.assertAllEqual(
            example.feature_lists.feature_list['names'].feature[0].bytes_list.value,
            [b'Janusz'])
        self.assertAllEqual(
            example.feature_lists.feature_list['names'].feature[1].bytes_list.value,
            [b'Andrzej'])

        self.assertAllEqual(
            example.feature_lists.feature_list['ratings'].feature[0].int64_list.value,
            [3, 9, 1])
        self.assertAllEqual(
            example.feature_lists.feature_list['ratings'].feature[1].int64_list.value,
            [2, 5])


class ExTuple(NamedTuple):
    int_field: int
    float_field: float
    bytes_field: bytes
    dict_field: Dict[str, Any]
    arr_field: NDArray[np.float64]


class InnerTuple(NamedTuple):
    time: np.datetime64
    title: np.str
    vals: np.ndarray


class NestedTuple(NamedTuple):
    int_field: int
    uint32_field: np.uint32
    inner_tuple: InnerTuple
    inner_tuple_list: List[InnerTuple]
    arrays_list: List[NDArray[np.int64]]


def _bytes_to_ndarray(serialized: bytes) -> np.ndarray:
    return tf.make_ndarray(TensorProto.FromString(serialized))


class TfEncodingFunctionTests(tf.test.TestCase):
    def test_array_to_feature(self):
        int32_array = np.arange(10, dtype=np.int32)
        int64_array = np.arange(10, dtype=np.int64)

        for int_array in (int32_array, int64_array):
            feature = tf_encoding.array_to_feature(int_array)
            self.assertEqual(feature.WhichOneof('kind'), 'int64_list')
            self.assertAllEqual(feature.int64_list.value, int_array)

        float_array = np.random.randn(10)
        feature = tf_encoding.array_to_feature(float_array)
        self.assertEqual(feature.WhichOneof('kind'), 'float_list')
        self.assertAllClose(feature.float_list.value, float_array)

        str_array = np.array(['ala', 'ma', 'kota'])
        bytes_array = np.array([val.encode('ascii') for val in str_array])

        for arr in (str_array, bytes_array):
            feature = tf_encoding.array_to_feature(arr)
            self.assertEqual(feature.WhichOneof('kind'), 'bytes_list')
            self.assertAllEqual(feature.bytes_list.value, arr.astype(dtype=np.bytes_))

        start = np.datetime64(datetime.now())
        date_array = np.array([start - np.timedelta64(t, 'D') for t in range(5)])
        feature = tf_encoding.array_to_feature(date_array)
        self.assertEqual(feature.WhichOneof('kind'), 'int64_list')
        self.assertAllEqual(
            feature.int64_list.value,
            datetime64_to_nano_seconds(date_array))

    def test_array_to_feature_invalid_type(self):
        obj_array = np.array([{}, {}])
        with self.assertRaises(TypeError):
            tf_encoding.array_to_feature(obj_array)

    def test_array_to_feature_invalid_shape(self):
        arr = np.arange(12).reshape(3, 4)
        with self.assertRaisesRegex(ValueError,
                                    'array should be one-dimensional'):
            tf_encoding.array_to_feature(arr)

    def test_structure_to_features_single_value(self):
        tffeatures = tf_encoding.structure_to_features(
            dict(val=4), use_tensor_protos=False)
        self.assertIsInstance(tffeatures.context['val'], tf.train.Feature)
        self.assertEqual(tffeatures.context['val'].int64_list.value, [4])

        tffeatures = tf_encoding.structure_to_features(
            dict(val=4), use_tensor_protos=True)
        self.assertIsInstance(tffeatures.context['val'], tf.train.Feature)
        self.assertEqual(_bytes_to_ndarray(tffeatures.context['val'].bytes_list.value[0]), 4)

    def test_structure_to_nested_features_nested_list_error(self):
        structure = dict(a=[dict(a=[1, 2])])
        with self.assertRaisesRegexp(TypeError, 'Detected nested lists'):
            tf_encoding.structure_to_features(structure)

    def test_structure_to_nested_features_flat_value_error(self):
        with self.assertRaises(TypeError):
            tf_encoding.structure_to_features(np.arange(10))

    def test_structure_to_nested_features_invalid_type(self):
        arr_1 = np.arange(10)
        arr_2 = np.random.randn(3)
        test_struct = ExTuple(
            int_field=5,
            float_field=3.3,
            bytes_field=b'Bytie McBytesface',
            dict_field={
                'field_arr': arr_1,
                'field_int': [[42], [41]],  # nested list!
                'field_dict': {
                    'inner_float': 33.1
                }
            },
            arr_field=arr_2)

        with self.assertRaisesRegex(
                TypeError, 'value type: .* is not supported'):
            tf_encoding.structure_to_features(
                test_struct, use_tensor_protos=False)

    def test_encode_named_tuple(self):
        start_time = np.datetime64(datetime.now())
        time_inner = start_time - np.timedelta64(3, 'M').astype('timedelta64[us]')
        time_list_1 = start_time - np.timedelta64(4, 'D').astype('timedelta64[us]')
        time_list_2 = start_time - np.timedelta64(40, 's').astype('timedelta64[us]')

        named_tuple = NestedTuple(
            int_field=3,
            uint32_field=np.uint32(99),
            inner_tuple=InnerTuple(
                time=time_inner,
                title=np.str('no_title'),
                vals=np.arange(10)),
            inner_tuple_list=[
                InnerTuple(
                    time=time_list_1,
                    title=np.str('title_1'),
                    vals=np.arange(-5, 4)),
                InnerTuple(
                    time=time_list_2,
                    title=np.str('title_2'),
                    vals=np.arange(5, 8)),
            ],
            arrays_list=[np.arange(3), np.arange(4), np.arange(5)]
        )

        def test_regular():
            # context features check
            tuple_features = tf_encoding.structure_to_features(
                named_tuple, use_tensor_protos=False)
            self.assertSetEqual(
                set(tuple_features.context.keys()),
                set(('int_field',
                     'uint32_field',
                     'inner_tuple/time',
                     'inner_tuple/title',
                     'inner_tuple/vals')))

            self.assertAllEqual(tuple_features.context['int_field'].int64_list.value,
                                [named_tuple.int_field])
            self.assertAllEqual(tuple_features.context['uint32_field'].int64_list.value,
                                [named_tuple.uint32_field])
            self.assertAllEqual(tuple_features.context['inner_tuple/time'].int64_list.value,
                                [datetime64_to_nano_seconds(named_tuple.inner_tuple.time)])
            self.assertAllEqual(tuple_features.context['inner_tuple/title'].bytes_list.value,
                                [named_tuple.inner_tuple.title.encode()])
            self.assertAllEqual(tuple_features.context['inner_tuple/vals'].int64_list.value,
                                named_tuple.inner_tuple.vals)

            # sequence features check
            self.assertSetEqual(
                set(tuple_features.feature_lists.keys()),
                set(('inner_tuple_list/time',
                     'inner_tuple_list/title',
                     'inner_tuple_list/vals',
                     'arrays_list')))

            for i in range(2):
                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/time'][i].int64_list.value,
                                    [datetime64_to_nano_seconds(named_tuple.inner_tuple_list[i].time)])

                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/title'][i].bytes_list.value,
                                    [named_tuple.inner_tuple_list[i].title.encode()])

                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/vals'][i].int64_list.value,
                                    named_tuple.inner_tuple_list[i].vals)

            for i in range(3):
                self.assertAllEqual(tuple_features.feature_lists['arrays_list'][i].int64_list.value,
                                    named_tuple.arrays_list[i])

        def test_tensor_protos():
            # context features check
            tuple_features = tf_encoding.structure_to_features(
                named_tuple, use_tensor_protos=True)
            self.assertSetEqual(
                set(tuple_features.context.keys()),
                set(('int_field',
                     'uint32_field',
                     'inner_tuple/time',
                     'inner_tuple/title',
                     'inner_tuple/vals')))

            self.assertAllEqual(
                _bytes_to_ndarray(
                    tuple_features.context['int_field'].bytes_list.value[0]),
                named_tuple.int_field)
            # problem with unsigned 32 and 64 integers - should be cast to int64
            uint32_val = tf.io.parse_tensor(
                tuple_features.context['uint32_field'].bytes_list.value[0],
                out_type=tf.int64)
            self.assertEqual(uint32_val.dtype, tf.int64)
            self.assertEqual(uint32_val.numpy(), named_tuple.uint32_field)

            self.assertAllEqual(
                _bytes_to_ndarray(
                    tuple_features.context['inner_tuple/time'].bytes_list.value[0]),
                datetime64_to_nano_seconds(named_tuple.inner_tuple.time))
            self.assertAllEqual(
                _bytes_to_ndarray(
                    tuple_features.context['inner_tuple/title'].bytes_list.value[0]),
                named_tuple.inner_tuple.title.encode())
            self.assertAllEqual(
                _bytes_to_ndarray(
                    tuple_features.context['inner_tuple/vals'].bytes_list.value[0]),
                named_tuple.inner_tuple.vals)

            # sequence features check - like in regular example
            self.assertSetEqual(
                set(tuple_features.feature_lists.keys()),
                set(('inner_tuple_list/time',
                     'inner_tuple_list/title',
                     'inner_tuple_list/vals',
                     'arrays_list')))

            for i in range(2):
                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/time'][i].int64_list.value,
                                    [datetime64_to_nano_seconds(named_tuple.inner_tuple_list[i].time)])

                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/title'][i].bytes_list.value,
                                    [named_tuple.inner_tuple_list[i].title.encode()])

                self.assertAllEqual(tuple_features.feature_lists['inner_tuple_list/vals'][i].int64_list.value,
                                    named_tuple.inner_tuple_list[i].vals)

            for i in range(3):
                self.assertAllEqual(tuple_features.feature_lists['arrays_list'][i].int64_list.value,
                                    named_tuple.arrays_list[i])

        test_regular()
        test_tensor_protos()

    def test_encoding_dictionary_with_many_lists(self):

        dictionary = dict(
            a=3,
            b=['ala', 'ma', 'kota'],
            c=[
                dict(d=1), dict(d=2), dict(d=3), dict(e=4)
            ],
            f=dict(g=[1, 2], h=5)
        )

        def test_regular():
            tffeatures = tf_encoding.structure_to_features(
                dictionary,
                use_tensor_protos=False,
                ignore_sequence_features=False)

            self.assertSetEqual(
                set(tffeatures.context.keys()),
                set(('a', 'f/h')))

            self.assertSetEqual(
                set(tffeatures.feature_lists.keys()),
                set(('b', 'c/d', 'c/e', 'f/g')))

            self.assertAllEqual(tffeatures.context['a'].int64_list.value,
                                [dictionary['a']])
            self.assertAllEqual(tffeatures.context['f/h'].int64_list.value,
                                [dictionary['f']['h']])

            for i in range(3):
                self.assertAllEqual(tffeatures.feature_lists['b'][i].bytes_list.value,
                                    [dictionary['b'][i].encode()])

            self.assertEqual(len(tffeatures.feature_lists['c/d']), 3)
            for i in range(3):
                self.assertAllEqual(tffeatures.feature_lists['c/d'][i].int64_list.value,
                                    [dictionary['c'][i]['d']])
            self.assertAllEqual(tffeatures.feature_lists['c/e'][0].int64_list.value,
                                [dictionary['c'][3]['e']])

            self.assertEqual(len(tffeatures.feature_lists['f/g']), 2)
            for i in range(2):
                self.assertAllEqual(tffeatures.feature_lists['f/g'][i].int64_list.value,
                                    [dictionary['f']['g'][i]])

        def test_tensor_protos():
            tffeatures = tf_encoding.structure_to_features(
                dictionary,
                use_tensor_protos=True,
                ignore_sequence_features=False)

            self.assertSetEqual(
                set(tffeatures.context.keys()),
                set(('a', 'f/h')))

            self.assertSetEqual(
                set(tffeatures.feature_lists.keys()),
                set(('b', 'c/d', 'c/e', 'f/g')))

        test_regular()
        test_tensor_protos()
