import abc
from typing import Dict, List, NamedTuple, Optional, Tuple, Any

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from nauto_datasets.core import spark
from nauto_datasets.protos import splits_pb2

DataSplits = Dict[str, DataFrame]


class SplitStrategy(metaclass=abc.ABCMeta):
    """An interface defining a dataset splitter"""

    @abc.abstractmethod
    def split(self, data: DataFrame) -> DataSplits:
        """Performes splitting of the `data` into disjoin subsets"""

    @staticmethod
    def instance_to_pb(
            split_strategy: 'SplitStrategy') -> splits_pb2.SplitStrategy:
        """Serializes a particular instance of `SplitStrategy` as one of the
        variants of `splits_pb2.SplitStrategy`

        The following instances are accepted:
        - @{nauto_datasets.cores.splits.RandomSplitByColumn}

        Args:
            splti_strategy: an instance of `SplitStrategy`

        Returns:
            `SplitStrategy` proto `Message` with the chosen instance as a
            variant.
        """
        if isinstance(split_strategy, RandomSplitByColumn):
            return splits_pb2.SplitStrategy(
                random_split_by_column=split_strategy.to_pb())
        else:
            raise ValueError(
                f'Unsupported split_strategy type {split_strategy}')

    @staticmethod
    def instance_from_pb(
            split_strategy_pb: splits_pb2.SplitStrategy) -> 'SplitStrategy':
        """Deserializes protobuf `Message` with a particular instance
        of `SplitStrategy`

        This class handles the following instances:
        - @{nauto_datasets.cores.splits.RandomSplitByColumn}

        Args:
            split_strategy_pb: a `splits_pb2.SplitStrategy` probotuf `Message`

        Returns:
            `SplitStrategy` instance
        """
        # introducing circural dependency for variant handling
        variant_name = split_strategy_pb.WhichOneof('split_strategy')
        if variant_name == 'random_split_by_column':
            return RandomSplitByColumn.from_pb(
                split_strategy_pb.random_split_by_column)
        else:
            raise ValueError(
                f'Unsupported split split strategy variant {variant_name}')

    @staticmethod
    def merge(
            old_splits: DataSplits,
            new_splits: DataSplits,
            reshuffle: bool = True
    ) -> DataSplits:
        """Merges two datasets.

        Given datasets should have splits with the same names and columns
        for the operation to suceed.

        Provide alternative implementation in the subclass if your split
        strategy stipulates requirements, which prohibit simple concatenation
        of relevant splits.

        Returns:
             data_splits with the same names as in `old_splits` and `new_splits`,
             where each split is a simple concatenation of corresponding
             old and new splits.
        """

        if set(old_splits) != set(new_splits):
            raise ValueError('Splits should have the same names')

        def concat(old_df: DataFrame, new_df: DataFrame) -> DataFrame:
            # align the columns of the second dataframe with the columns of the first
            # otherwise the union might fail or produce invalid results
            return old_df.union(new_df.select(old_df.columns))

        def maybe_reshuffle(df: DataFrame) -> DataFrame:
            if reshuffle:
                return spark.shuffle(df)
            else:
                return df

        return {
            split_name: maybe_reshuffle(
                concat(old_splits[split_name], new_splits[split_name]))
            for split_name in old_splits
        }


class SplitConfig(NamedTuple):
    """Configuration of a single data split.
    Attributes:
        name: the name of the split - subset of the data
        fraction: the number describing the relative fraction
            of examples, which should be a part of this split.
            This number has meaning only in the context of all other
            split configs, as it should be normalized by the sum of
            all other split's fractions to get the actual percentage.
        shuffle: whether the rows of these split should be in random
            order.
    """
    name: str
    fraction: float
    shuffle: bool

    def to_pb(self) -> splits_pb2.SplitConfig:
        """Serializes config as proto `Message`"""
        return splits_pb2.SplitConfig(**self._asdict())

    @staticmethod
    def from_pb(split_conf_pb: splits_pb2.SplitConfig) -> 'SplitConfig':
        """Deserializes config from proto `Message`"""
        return SplitConfig(name=split_conf_pb.name,
                           fraction=split_conf_pb.fraction,
                           shuffle=split_conf_pb.shuffle)

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return splits_pb2.SplitConfig


class RandomSplitByColumn(SplitStrategy):
    """`SplitStrategy` variant performing stratified sampling on
    the chosen column.
    """

    def __init__(
            self,
            split_confs: List[SplitConfig],
            column: str,
            max_negatives_fraction: Optional[float] = None,
            max_size: Optional[int] = None
    ) -> None:
        """Creates a `RandomSplitByColumn` strategy

        Args:
            split_confs: a list of configurations of each split
            column: the binary column chosen for the splitting
            max_negatives_fraction: the maximum allowed fraction
                of the negative examples in each split
            max_size: the maximum number of the elements in total
                in all the splits.
        """
        self._split_confs = split_confs
        self._column = column
        self._max_negatives_fraction = max_negatives_fraction
        self._max_size = max_size

    def __eq__(self, other: Any) -> bool:
        def compare_optional_floats(
                f1: Optional[float], f2: Optional[float]) -> bool:
            return (
                (f1, f2) == (None, None) or
                (f1 is not None and f2 is not None and
                    abs(f1 - f2) < 1e-3))

        if isinstance(other, RandomSplitByColumn):
            return (
                self._split_confs == other._split_confs and
                self._column == other._column and
                compare_optional_floats(
                    self._max_negatives_fraction,
                    other._max_negatives_fraction),
                self._max_size == other._max_size)
        else:
            return False

    def to_pb(self) -> splits_pb2.RandomSplitByColumn:
        """Serializes `RandomSplitByColumn` as proto `Message`"""
        return splits_pb2.RandomSplitByColumn(
            split_confs=[conf.to_pb() for conf in self._split_confs],
            column=self._column,
            max_negatives_fraction=self._max_negatives_fraction,
            max_size=self._max_size)

    @staticmethod
    def from_pb(
            rs_pb: splits_pb2.RandomSplitByColumn) -> 'RandomSplitByColumn':
        """Deserializes `RandomSplitByColumn` from proto `Message`"""
        if rs_pb.max_negatives_fraction == 0:
            mnf = None
        else:
            mnf = rs_pb.max_negatives_fraction
        if rs_pb.max_size == 0:
            ms = None
        else:
            ms = rs_pb.max_size

        return RandomSplitByColumn(
            split_confs=[
                SplitConfig.from_pb(conf_pb) for conf_pb in rs_pb.split_confs
            ],
            column=rs_pb.column,
            max_negatives_fraction=mnf,
            max_size=ms)

    @staticmethod
    def pb_message_type() -> type:
        """Returns the type of the associate protobuf `Message`"""
        return splits_pb2.RandomSplitByColumn

    def _balance_data(
            self,
            pos_df: DataFrame,
            neg_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        pos_count = pos_df.count()
        neg_count = neg_df.count()

        if pos_count + neg_count == 0:
            return pos_df, neg_df

        if (self._max_negatives_fraction is not None
                and neg_count / (pos_count + neg_count) > self._max_negatives_fraction):
            sub_neg_count = (
                (self._max_negatives_fraction * pos_count) /
                (1.0 - self._max_negatives_fraction))

            neg_df = neg_df.sample(
                withReplacement=False,
                fraction=sub_neg_count / neg_count)
            # it is actually an approximation
            neg_count = sub_neg_count

        if self._max_size is not None:
            pos_fraction = pos_count / (pos_count + neg_count)
            pos_to_sample = pos_fraction * self._max_size
            neg_to_sample = self._max_size - pos_to_sample

            pos_df = pos_df.sample(withReplacement=False,
                                   fraction=pos_to_sample / pos_count)
            neg_df = neg_df.sample(withReplacement=False,
                                   fraction=neg_to_sample / neg_count)

        return pos_df, neg_df

    def split(self, data_df: DataFrame) -> DataSplits:
        """Returns splits after performing stratified sampling procedure.

        Each split should contain approximately the same fractions
        of positive and negative examples dessignated by the split column.
        """
        # normalize fractions
        fractions = [sc.fraction for sc in self._split_confs]
        fractions = np.array(fractions) / np.sum(fractions)
        positives_df = data_df.where(F.col(self._column))
        negatives_df = data_df.where(~F.col(self._column))

        if (self._max_negatives_fraction is not None
                or self._max_size is not None):
            positives_df, negatives_df = self._balance_data(
                positives_df, negatives_df)

        positive_splits_df = positives_df.randomSplit(weights=fractions)
        negative_splits_df = negatives_df.randomSplit(weights=fractions)

        splits = [
            p_df.union(n_df) for p_df, n_df
            in zip(positive_splits_df, negative_splits_df)
        ]
        splits = [
            spark.shuffle(part_df) if sc.shuffle else part_df
            for part_df, sc in zip(splits, self._split_confs)
        ]

        return {
            sc.name: part_df for sc, part_df
            in zip(self._split_confs, splits)
        }
