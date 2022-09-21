import numpy as np
import tensorflow as tf

from nauto_datasets.utils import transformations as nauto_trans


class ToUtcTimeTests(tf.test.TestCase):

    def test_positive_offset(self):
        time_ns = np.arange(11, 101, 10).astype(np.uint64)
        boot_time = np.iinfo(np.uint64).max >> np.uint64(1)
        offset = np.int64(40)

        utc_time_ns = nauto_trans.to_utc_time(
            time_ns, boot_time, offset, in_place=False)
        self.assertAllEqual(utc_time_ns, time_ns + boot_time + np.uint64(offset))

        nauto_trans.to_utc_time(time_ns, boot_time, offset, in_place=True)
        self.assertAllEqual(utc_time_ns, time_ns)

    def test_negative_offset(self):
        time_ns = np.arange(11, 101, 10).astype(np.uint64)
        boot_time = np.iinfo(np.uint64).max >> np.uint64(2)
        offset = np.int64(-42342)

        utc_time_ns = nauto_trans.to_utc_time(
            time_ns, boot_time, offset, in_place=False)
        self.assertAllEqual(utc_time_ns, time_ns + boot_time - np.uint64(-offset))

        nauto_trans.to_utc_time(time_ns, boot_time, offset, in_place=True)
        self.assertAllEqual(utc_time_ns, time_ns)
