import numpy as np
import tensorflow as tf
from typing import NamedTuple

from nauto_datasets.utils import dicts as dict_utils


class TestDictUtils(tf.test.TestCase):
    def test_flatten_nested_features_dict(self):
        arr_1 = np.arange(10)
        arr_2 = np.random.randn(3)
        test_dict = dict(
            int_field=5,
            float_field=3.3,
            bytes_field=b'Bytie McBytesface',
            dict_field={
                'field_arr': arr_1,
                'field_int': 42,
                'field_dict': {
                    'inner_float': 33.1
                }
            },
            arr_field=arr_2)

        flattened_vals = dict_utils.flatten_nested_dict(test_dict)

        self.assertSetEqual(
            set(flattened_vals.keys()),
            set(('int_field',
                 'float_field',
                 'bytes_field',
                 'dict_field/field_arr',
                 'dict_field/field_int',
                 'dict_field/field_dict/inner_float',
                 'arr_field')))

        # some check
        self.assertEqual(
            flattened_vals['dict_field/field_dict/inner_float'], 33.1)

        flattened_vals = dict_utils.flatten_nested_dict(
            test_dict, key_prefix='some_prefix', sep='#')

        self.assertSetEqual(
            set(flattened_vals.keys()),
            set(('some_prefix#int_field',
                 'some_prefix#float_field',
                 'some_prefix#bytes_field',
                 'some_prefix#dict_field#field_arr',
                 'some_prefix#dict_field#field_int',
                 'some_prefix#dict_field#field_dict#inner_float',
                 'some_prefix#arr_field')))

        # some check
        self.assertAllEqual(
            flattened_vals['some_prefix#dict_field#field_arr'], arr_1)

    def test_concat_dicts(self):

        dict_1 = dict(a=3, b='ala', c=[1])
        dict_2 = dict(a=2, b='ola')
        dict_3 = dict(b='ula', c=[2])
        dict_4 = dict()

        dicts = [dict_1, dict_2, dict_3, dict_4]

        c_dict = dict_utils.concat_dicts(dicts)

        self.assertDictEqual(
            c_dict,
            dict(
                a=[3, 2],
                b=['ala', 'ola', 'ula'],
                c=[[1], [2]]
            ))

    def test_unzip_dict(self):
        in_dict = dict(a=[1, 2, 3], b=['A', 'B', 'C', 'D'])
        out_dicts = dict_utils.unzip_dict(in_dict)

        exp_out_dicts = [
            dict(a=1, b='A'),
            dict(a=2, b='B'),
            dict(a=3, b='C'),
            dict(b='D')
        ]

        self.assertAllEqual(out_dicts, exp_out_dicts)

        class NT(NamedTuple):
            field_1: int
            field_2: str

        in_dict = dict(a=NT(1, 'A'), b=NT(2, 'B'))
        out_dicts = dict_utils.unzip_dict(in_dict)
        exp_out_dicts = [dict(a=1, b=2), dict(a='A', b='B')]
