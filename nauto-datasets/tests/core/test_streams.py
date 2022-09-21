import numpy as np
import tensorflow as tf
from typing import List, NamedTuple

from nauto_datasets.core import streams
from nauto_datasets.utils.tuples import NamedTupleMetaEx
from nauto_datasets.utils.numpy import NDArray


class BoxInt(NamedTuple):
    val: int


class ExStream(streams.StreamMixin, metaclass=NamedTupleMetaEx):
    x: NDArray[np.int32]  # x is an indexing field
    y: NDArray[np.float32]
    k: List[BoxInt]


class StreamMixinTest(tf.test.TestCase):

    def test_creation_valid(self):
        ints = list(map(BoxInt, np.arange(10)))
        stream = ExStream(np.arange(10), np.zeros(10), ints)
        self.assertAllEqual(stream.x, np.arange(10))
        self.assertAllEqual(stream.y, np.zeros(10))
        self.assertAllEqual(stream.k, ints)

    def test_creation_with_empty_field(self):
        stream = ExStream(np.arange(10), np.array([], dtype=np.float32), [])
        self.assertAllEqual(stream.x, np.arange(10))
        self.assertAllEqual(stream.y, [])
        self.assertAllEqual(stream.k, [])

    def test_creation_invalid(self):
        with self.assertRaisesRegex(
                ValueError, 'Stream arrays have to be empty or of equal size'):
            ExStream(np.arange(10), np.arange(9), [BoxInt(1)])

        with self.assertRaisesRegex(
                ValueError, 'Stream arrays have to be empty or of equal size'):
            ExStream(np.array([], dtype=np.int32), np.array([], dtype=np.float32), [BoxInt(1)])

    def test_size(self):
        ints = list(map(BoxInt, np.arange(10)))
        stream = ExStream(np.arange(10), np.arange(10), ints)
        self.assertEqual(stream._size(), 10)
        stream = ExStream(np.arange(10), np.array([], dtype=np.float32), [])
        self.assertEqual(stream._size(), 10)

    def test_is_empty(self):
        self.assertFalse(ExStream(np.arange(1), np.arange(1), [])._is_empty())
        self.assertTrue(ExStream(np.arange(0), np.arange(0), [])._is_empty())
        self.assertFalse(ExStream(np.arange(1), np.arange(0), [BoxInt(1)])._is_empty())

    def test_to_and_from_df(self):
        x = np.arange(5)
        y = np.random.randn(5).astype(np.float32)
        k = list(map(BoxInt, np.arange(5)))
        stream = ExStream(x, y, k)
        df = stream._to_df()
        stream_2 = ExStream.from_df(df)
        self.assertAllEqual(stream.x, stream_2.x)
        self.assertAllClose(stream.y, stream_2.y)
        self.assertAllEqual(stream.k, stream_2.k)

    def test_to_and_from_df_with_empty_field(self):
        x = np.arange(5)
        y = np.random.randn(0).astype(np.float32)
        k = list(map(BoxInt, x))
        stream = ExStream(x, y, k)
        df = stream._to_df()
        stream_2 = ExStream.from_df(df)
        self.assertAllEqual(stream.x, stream_2.x)
        self.assertAllClose(stream.y, stream_2.y)
        self.assertAllEqual(stream.k, stream_2.k)

    def test_empty_creation(self):
        empty_stream = ExStream.empty()
        self.assertEqual(len(empty_stream.x), 0)
        self.assertEqual(len(empty_stream.y), 0)
        self.assertEqual(len(empty_stream.k), 0)

    def test_range(self):
        x = np.arange(5)
        y = np.random.randn(5)
        k = list(map(BoxInt, x))
        stream = ExStream(x, y, k)
        self.assertEqual(stream._range('x'), (x[0], x[-1]))
        self.assertEqual(stream._range('y'), (y[0], y[-1]))
        self.assertEqual(stream._range('k'), (k[0], k[-1]))
        self.assertIsNone(ExStream.empty()._range('x'))

    def test_concat(self):
        x_1 = np.arange(5)
        x_2 = np.arange(-5, 0)
        x_3 = np.arange(5, 10)
        y_1 = np.random.randn(5)
        y_2 = np.random.randn(5)
        y_3 = np.random.randn(5)
        k_1 = list(map(BoxInt, x_1))
        k_2 = list(map(BoxInt, x_2))
        k_3 = list(map(BoxInt, x_3))

        s_1 = ExStream(x_1, y_1, k_1)
        s_2 = ExStream(x_2, y_2, k_2)
        s_3 = ExStream(x_3, y_3, k_3)

        s_c = ExStream.concat([s_1, s_2, s_3])
        self.assertAllEqual(s_c.x, np.concatenate([x_1, x_2, x_3]))
        self.assertAllClose(s_c.y, np.concatenate([y_1, y_2, y_3]))
        self.assertAllEqual(s_c.k, list(np.concatenate([k_1, k_2, k_3])))


class AdditionaBaseClass:
    pass


ComExStream = streams.create_combined_stream_type(
    'ComExStream', ExStream, [AdditionaBaseClass])


class NotAStream(metaclass=NamedTupleMetaEx):
    x: int


class CombinedStreamMixinTest(tf.test.TestCase):

    def test_subtyping(self):
        self.assertTrue(
            issubclass(ComExStream,
                       (CombinedStreamMixinTest, AdditionaBaseClass)))

    def test_create_combined_stream_from_invalid_type(self):
        with self.assertRaisesRegex(
                ValueError,
                'base_stream_t should be a StreamMixin'):
            streams.create_combined_stream_type('ComNotAStream', NotAStream)

    def test_by_hand_creation_valid(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        k = list(map(BoxInt, x))
        com_stream = ComExStream(
            stream=ExStream(x, y, k),
            lengths=[len(x)])
        self.assertAllEqual(com_stream.stream.x, x)
        self.assertAllEqual(com_stream.stream.y, y)
        self.assertAllEqual(com_stream.stream.k, k)
        self.assertAllEqual(com_stream.lengths, [len(x)])

        # should not fail
        ComExStream(
            stream=ExStream(x, y, k),
            lengths=[len(x)-1, 1])

    def test_by_hand_creation_invalid(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        k = []
        with self.assertRaisesRegex(
                ValueError,
                'Stream length is not equal to the sum of lengths'):
            ComExStream(stream=ExStream(x, y, k), lengths=[len(x)-1])

    def test_from_substreams(self):
        x_1 = np.arange(5)
        x_2 = np.arange(-5, 0)
        y_1 = np.arange(5, 10)
        y_2 = np.arange(10, 15)
        k_1 = list(map(BoxInt, x_1))
        k_2 = list(map(BoxInt, x_2))
        s_1 = ExStream(x_1, y_1, k_1)
        s_2 = ExStream(x_2, y_2, k_2)

        s_c = ComExStream.from_substreams([s_1, s_2])
        self.assertAllEqual(s_c.stream.x, np.concatenate([x_1, x_2]))
        self.assertAllEqual(s_c.stream.y, np.concatenate([y_1, y_2]))
        self.assertAllEqual(s_c.stream.k, list(np.concatenate([k_1, k_2])))
        self.assertAllEqual(s_c.lengths, [len(x_1), len(x_2)])

    def test_from_substreams_invalid(self):
        x_1 = np.arange(5)
        x_2 = np.arange(-5, 0)
        y_1 = np.arange(5, 10)
        # second substream has an empty field
        # for all substreams, this field has to be either empty or non-empty
        # at once
        y_2 = np.arange(0)
        s_1 = ExStream(x_1, y_1, [])
        s_2 = ExStream(x_2, y_2, [])

        with self.assertRaisesRegex(
                ValueError, 'Stream arrays have to be empty or of equal size'):
            ComExStream.from_substreams([s_1, s_2])

    def test_count(self):
        stream = ComExStream(
            stream=ExStream(np.arange(10), np.arange(10), list(map(BoxInt, np.arange(10)))),
            lengths=[5, 5])
        self.assertEqual(stream._substreams_count(), 2)

    def test_offsets(self):
        stream = ComExStream(
            stream=ExStream(np.arange(10), np.arange(10), []),
            lengths=[3, 7])
        self.assertAllEqual(stream._substreams_offsets(), [3, 10])

    def test_ith_substream(self):
        stream = ComExStream(
            stream=ExStream(np.arange(10), np.random.randn(10), list(map(BoxInt, np.arange(10)))),
            lengths=[3, 7])
        str_0 = stream._ith_substream(0)
        str_1 = stream._ith_substream(1)

        self.assertAllEqual(str_0.x, stream.stream.x[:3])
        self.assertAllEqual(str_0.y, stream.stream.y[:3])
        self.assertAllEqual(str_0.k, stream.stream.k[:3])
        self.assertAllEqual(str_1.x, stream.stream.x[3:])
        self.assertAllEqual(str_1.y, stream.stream.y[3:])
        self.assertAllEqual(str_1.k, stream.stream.k[3:])

    def test_substreams(self):
        stream = ComExStream(
            stream=ExStream(np.arange(10), np.random.randn(10), list(map(BoxInt, np.arange(10)))),
            lengths=[3, 7])
        substreams = stream._substreams()
        self.assertEqual(len(substreams), 2)
        str_0, str_1 = substreams
        self.assertAllEqual(str_0.x, stream.stream.x[:3])
        self.assertAllEqual(str_0.y, stream.stream.y[:3])
        self.assertAllEqual(str_0.k, stream.stream.k[:3])
        self.assertAllEqual(str_1.x, stream.stream.x[3:])
        self.assertAllEqual(str_1.y, stream.stream.y[3:])
        self.assertAllEqual(str_1.k, stream.stream.k[3:])
