from typing import List, NamedTuple, Dict
from datetime import date
import itertools


property_cache = {}  # named tuples cannot have additional fields, so storing cache here


def cached_property(f):
    def _hash(instance, method):
        return str(id(instance)) + "_" + str(id(method))

    def get(self):
        try:
            return property_cache[_hash(self, f)]
        except (AttributeError, KeyError):
            x = property_cache[_hash(self, f)] = f(self)
            return x
    return property(get)


class ModelBooks(NamedTuple):
    model_name: str
    versions: List[str]
    statistics: 'ModelStatistics'

    @property
    def identifier(self):
        return f"Model: {self.model_name}"


class ModelVersionBooks(NamedTuple):
    model_name: str
    version: str
    statistics: 'ModelStatistics'

    @property
    def identifier(self):
        return f"Model: {self.model_name}\nModel version: {self.version}"


class ModelStatistics(NamedTuple):
    # todo add examples for each class
    # todo hard examples mining
    time_range: 'TimeRange'
    pr_data: 'PRData'
    pr_curve: 'PRCurve'
    judged_cases: int
    total_cases: int

    @cached_property
    def coverage(self) -> float:
        return self.judged_cases / self.total_cases

    @staticmethod
    def aggregate(versions: List['ModelStatistics']) -> 'ModelStatistics':
        return ModelStatistics(
            time_range=TimeRange.aggregate([v.time_range for v in versions]),
            pr_data=PRData.aggregate([v.pr_data for v in versions]),
            pr_curve=PRCurve(
                true_labels=list(itertools.chain.from_iterable([v.pr_curve.true_labels for v in versions])),
                predicted_scores=list(itertools.chain.from_iterable([v.pr_curve.predicted_scores for v in versions])),
            ),
            judged_cases=sum([v.judged_cases for v in versions]),
            total_cases=sum([v.total_cases for v in versions]),
        )


class PRCurve(NamedTuple):
    true_labels: List[bool]
    predicted_scores: List[bool]

    @cached_property
    def precision_values(self) -> List[float]:
        return [i.data.precision for i in self.items]

    @cached_property
    def recall_values(self) -> List[float]:
        return [i.data.recall for i in self.items]

    @cached_property
    def threshold_values(self) -> List[float]:
        return [i.threshold for i in self.items]

    @cached_property
    def f1_values(self) -> List[float]:
        return [i.data.f1 for i in self.items]

    @cached_property
    def best_f1(self) -> float:
        return max([i.data.f1 for i in self.items])

    @cached_property
    def best_threshold(self) -> float:
        return [i.threshold for i in self.items if i.data.f1 == self.best_f1][0]

    @cached_property
    def best_threshold_index(self) -> int:
        return [i.threshold for i in self.items].index(self.best_threshold)

    @cached_property
    def items(self) -> List['PRCurveItem']:
        items = []
        for threshold in [t / 100 for t in range(0, 101)]:
            items.append(PRCurveItem(
                threshold=threshold,
                data=PRData.from_labels(
                    true_labels=self.true_labels,
                    predicted_labels=[score > threshold for score in self.predicted_scores]
                )
            ))
        return items


class PRCurveItem(NamedTuple):
    threshold: float
    data: 'PRData'


class PRData(NamedTuple):
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @cached_property
    def num_cases(self) -> int:
        return self.false_negatives + self.false_positives + self.true_negatives + self.true_positives

    @cached_property
    def f1(self) -> float:
        if self.precision or self.recall:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        return 0.

    @cached_property
    def precision(self) -> float:
        if self.true_positives or self.false_positives:
            return self.true_positives / (self.true_positives + self.false_positives)
        return 1.

    @cached_property
    def recall(self) -> float:
        if self.true_positives or self.false_negatives:
            return self.true_positives / (self.true_positives + self.false_negatives)
        return 0.

    @cached_property
    def accuracy(self) -> float:
        if self.num_cases:
            return (self.true_positives + self.true_negatives) / self.num_cases
        return 0.

    @staticmethod
    def from_labels(true_labels: List[bool], predicted_labels: List[bool]) -> 'PRData':
        return PRData(
            true_positives=sum([1 for index, label in enumerate(predicted_labels) if label and true_labels[index]]),
            true_negatives=sum([1 for index, label in enumerate(predicted_labels) if not label and not true_labels[index]]),
            false_positives=sum([1 for index, label in enumerate(predicted_labels) if label and not true_labels[index]]),
            false_negatives=sum([1 for index, label in enumerate(predicted_labels) if not label and true_labels[index]]),
        )

    @staticmethod
    def aggregate(prs: List['PRData']) -> 'PRData':
        return PRData(
            true_positives=sum([pr.true_positives for pr in prs]),
            true_negatives=sum([pr.true_negatives for pr in prs]),
            false_positives=sum([pr.false_positives for pr in prs]),
            false_negatives=sum([pr.false_negatives for pr in prs]),
        )


class TimeRange(NamedTuple):
    min_time: date
    max_time: date
    distribution: 'TimeDistribution'

    @staticmethod
    def from_dates(dates: List[date]) -> 'TimeRange':
        return TimeRange(
            min_time=min(dates),
            max_time=max(dates),
            distribution=TimeDistribution.from_dates(dates)
        )

    @staticmethod
    def aggregate(ranges: List['TimeRange']) -> 'TimeRange':
        return TimeRange(
            min_time=min([r.min_time for r in ranges]),
            max_time=max([r.max_time for r in ranges]),
            distribution=TimeDistribution.aggregate([r.distribution for r in ranges])
        )


class TimeDistribution(NamedTuple):
    # todo sort and fill the gaps
    daily_distribution: Dict[date, int]

    @cached_property
    def key_name(self):
        return "day"  # todo

    @cached_property
    def weekly_distribution(self) -> Dict[str, int]:
        pass  # todo

    @cached_property
    def monthly_distribution(self) -> Dict[str, int]:
        pass  # todo

    @cached_property
    def recommended_distribution(self) -> Dict[str, int]:
        # todo real implementation
        return {k.isoformat(): v for k, v in self.daily_distribution.items()}

        if len(self.daily_distribution.keys()) < 30:
            return {k.isoformat(): v for k, v in self.daily_distribution.items()}

        if len(self.daily_distribution) > 200:
            return self.monthly_distribution

        return self.weekly_distribution

    @staticmethod
    def from_dates(dates: List[date]) -> 'TimeDistribution':
        return TimeDistribution(daily_distribution={
            k: len(list(v))
            for k, v in itertools.groupby(sorted(dates))
        })

    @staticmethod
    def aggregate(distributions: List['TimeDistribution']) -> 'TimeDistribution':
        aggregated = {}
        for distribution in distributions:
            for current_date, value in distribution.daily_distribution.items():
                if date not in aggregated:
                    aggregated[current_date] = 0
                aggregated[current_date] += value
        return TimeDistribution(daily_distribution=aggregated)
