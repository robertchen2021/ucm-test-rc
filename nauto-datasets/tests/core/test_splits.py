import numpy as np
import pandas as pd
import unittest
from pyspark.sql import DataFrame

from nauto_datasets.core import splits
from nauto_datasets.utils.tests import SparkTestCase


class TestRandomSplitByColumn(SparkTestCase):

    def _get_data(self) -> DataFrame:
        ids = np.arange(300)
        features = np.random.rand(300)

        label_1s = np.repeat(False, 300)
        label_2s = np.repeat(False, 300)
        label_1s[np.random.choice(300, size=30, replace=False)] = True
        label_2s[np.random.choice(300, size=100, replace=False)] = True
        pdf = pd.DataFrame(
            dict(id=ids,
                 feature=features,
                 label_1=label_1s,
                 label_2=label_2s))
        pdf.set_index('id')

        return self.spark.createDataFrame(pdf)

    def test_random_split_into_5(self):

        df = self._get_data()

        split_confs = [
            splits.SplitConfig('s1', 1, True),
            splits.SplitConfig('s2', 1, True),
            splits.SplitConfig('s3', 1, True),
            splits.SplitConfig('s4', 1, True),
            splits.SplitConfig('s5', 1, True)
        ]

        splitter = splits.RandomSplitByColumn(
            split_confs,
            column='label_2')

        data_splits = splitter.split(df)
        self.assertEqual(len(data_splits), len(split_confs))
        self.assertSetEqual(
            set(data_splits.keys()),
            set(['s1', 's2', 's3', 's4', 's5']))

        for ds in data_splits.values():
            self.assertEqual(df.schema, ds.schema)

        s1_true_count = data_splits['s1'].where('label_2').count()
        s2_true_count = data_splits['s2'].where('label_2').count()
        s3_true_count = data_splits['s3'].where('label_2').count()
        s4_true_count = data_splits['s4'].where('label_2').count()
        s5_true_count = data_splits['s5'].where('label_2').count()
        s1_count = data_splits['s1'].count()
        s2_count = data_splits['s2'].count()
        s3_count = data_splits['s3'].count()
        s4_count = data_splits['s4'].count()
        s5_count = data_splits['s5'].count()

        self.assertEqual(
            np.sum(
                [s1_true_count,
                 s2_true_count,
                 s3_true_count,
                 s4_true_count,
                 s5_true_count]),
            df.where('label_2').count())
        self.assertEqual(
            np.sum(
                [s1_count,
                 s2_count,
                 s3_count,
                 s4_count,
                 s5_count]),
            df.count())
        self.assertAlmostEqual(s1_true_count, 20, delta=15)
        self.assertAlmostEqual(s2_true_count, 20, delta=15)
        self.assertAlmostEqual(s3_true_count, 20, delta=15)
        self.assertAlmostEqual(s4_true_count, 20, delta=15)
        self.assertAlmostEqual(s5_true_count, 20, delta=15)

    def test_random_split_with_max_size(self):

        df = self._get_data()

        split_confs = [
            splits.SplitConfig('s1', 2, True),
            splits.SplitConfig('s2', 1, True),
            splits.SplitConfig('s3', 1, True),
        ]

        splitter = splits.RandomSplitByColumn(
            split_confs,
            column='label_2',
            max_size=200
        )

        data_splits = splitter.split(df)
        self.assertEqual(len(data_splits), len(split_confs))
        self.assertSetEqual(
            set(data_splits.keys()),
            set(['s1', 's2', 's3']))

        for ds in data_splits.values():
            self.assertEqual(df.schema, ds.schema)

        s1_true_count = data_splits['s1'].where('label_2').count()
        s2_true_count = data_splits['s2'].where('label_2').count()
        s3_true_count = data_splits['s3'].where('label_2').count()

        s1_count = data_splits['s1'].count()
        s2_count = data_splits['s2'].count()
        s3_count = data_splits['s3'].count()

        # it is not guaranteed that the sampled number will be exact
        self.assertAlmostEqual(s1_count + s2_count + s3_count, 200, delta=30)

        # check dataset proportions - again - not guaranteed exact number
        self.assertAlmostEqual(s1_count, s2_count + s3_count, delta=20)
        self.assertAlmostEqual(
            s1_true_count, s2_true_count + s3_true_count,
            delta=20)

    def test_split_with_max_neg_fraction(self):
        df = self._get_data()

        split_confs = [
            splits.SplitConfig('s1', 2, True),
            splits.SplitConfig('s2', 1, True),
            splits.SplitConfig('s3', 1, True),
        ]

        splitter = splits.RandomSplitByColumn(
            split_confs,
            column='label_1',
            max_negatives_fraction=0.7)

        data_splits = splitter.split(df)

        data_splits = splitter.split(df)
        self.assertEqual(len(data_splits), len(split_confs))
        self.assertSetEqual(
            set(data_splits.keys()),
            set(['s1', 's2', 's3']))

        for ds in data_splits.values():
            self.assertEqual(df.schema, ds.schema)

        s1_true_count = data_splits['s1'].where('label_1').count()
        s2_true_count = data_splits['s2'].where('label_1').count()
        s3_true_count = data_splits['s3'].where('label_1').count()

        s1_count = data_splits['s1'].count()
        s2_count = data_splits['s2'].count()
        s3_count = data_splits['s3'].count()

        self.assertEqual(
            s1_true_count + s2_true_count + s3_true_count,
            df.where('label_1').count())

        # delta everywhere - everythin is approxmiate
        self.assertAlmostEqual(
            s1_count + s2_count + s3_count,
            100,
            delta=20)

        self.assertAlmostEqual(s1_count, s2_count + s3_count, delta=30)
        self.assertAlmostEqual(
            s1_true_count,
            s2_true_count + s3_true_count,
            delta=15)
        self.assertAlmostEqual(s2_count, s3_count, delta=15)
        self.assertAlmostEqual(s2_true_count, s3_true_count, delta=15)


class TestSplitStrategy(SparkTestCase):

    def test_proto_serialization(self):

        split_confs = [
            splits.SplitConfig('s1', 3, True),
            splits.SplitConfig('s2', 5, False),
            splits.SplitConfig('s3', 2, True),
            splits.SplitConfig('s4', 4, False),
            splits.SplitConfig('s5', 1, True)
        ]

        splitter = splits.RandomSplitByColumn(
            split_confs,
            column='some_column',
            max_negatives_fraction=0.9,
            max_size=None)

        split_strategy_pb = splits.SplitStrategy.instance_to_pb(
            splitter)
        des_splitter = splits.SplitStrategy.instance_from_pb(
            split_strategy_pb)

        self.assertEqual(des_splitter._split_confs, splitter._split_confs)
        self.assertEqual(des_splitter._column, splitter._column)
        self.assertAlmostEqual(des_splitter._max_negatives_fraction,
                               splitter._max_negatives_fraction)
        self.assertEqual(des_splitter._max_size, splitter._max_size)
        self.assertEqual(des_splitter, splitter)

    def test_failing_deserialization(self):
        class SomeOtherStrategy(splits.SplitStrategy):
            pass

        with self.assertRaises(ValueError):
            splits.SplitStrategy.instance_to_pb(SomeOtherStrategy)

    def test_base_merge(self):
        def produce(count: int, begin: int):
            ids = np.arange(begin, begin + count)
            features = np.random.rand(count)

            label_1s = np.repeat(False, count)
            label_2s = np.repeat(False, count)
            label_1s[np.random.choice(count, size=30, replace=False)] = True
            label_2s[np.random.choice(count, size=count - 30, replace=False)] = True
            pdf = pd.DataFrame(
                dict(id=ids,
                     feature=features,
                     label_1=label_1s,
                     label_2=label_2s))
            pdf.set_index('id')

            return self.spark.createDataFrame(pdf)

        old_train = produce(100, 0)
        old_test = produce(50, 100)

        new_train = produce(200, 200)
        new_test = produce(100, 450)

        merged_splits = splits.SplitStrategy.merge(
            dict(train=old_train, test=old_test),
            dict(train=new_train, test=new_test))

        self.assertSetEqual(set(merged_splits), {'train', 'test'})
        self.assertEqual(merged_splits['train'].count(),
                         old_train.count() + new_train.count())
        self.assertEqual(merged_splits['test'].count(),
                         old_test.count() + new_test.count())
