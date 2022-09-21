from nauto_zoo.interfaces import ModelResponse
from typing import Dict, List, Optional


class ModelResponseTimeline(ModelResponse):
    def __init__(self, timeline: 'Timeline', **kwargs):
        if "score" not in kwargs:
            kwargs["score"] = 0.
        self.timeline = timeline
        super().__init__(**kwargs)

    def validate(self):
        super().validate()
        if not isinstance(self.timeline, Timeline):
            raise RuntimeError("Timeline is not a Timeline instance")
        if not self.timeline.is_valid():
            raise RuntimeError("Timeline is invalid")


class Timeline(object):
    def __init__(self, review_type: str, offset_ns: int = None):
        self._review_type = review_type
        self._offset_ns = offset_ns
        self._elements: List[TimelineElement] = []

    def __repr__(self):
        if self.is_empty():
            return f"Empty timeline `{self._review_type}`"
        return f"Timeline `{self._review_type}`. Offset: `{self._offset_ns}`, " \
               f"elements: {[[e.start_ns, e.end_ns] for e in self._elements]}"

    def add_element(self, element: 'TimelineElement'):
        assert isinstance(element, TimelineElement)
        self._elements.append(element)

    def is_empty(self) -> bool:
        return len(self._elements) == 0

    def is_valid(self) -> bool:
        return self._offset_ns is not None

    def set_offset_ns(self, offset_ns: int):
        self._offset_ns = offset_ns

    def __len__(self):
        return len(self._elements)

    def drt_format(self):
        return {
            'offset_ns': self._offset_ns,
            'elements': [e.drt_format(self._review_type) for e in self._elements]
        }

    @property
    def elements(self) -> "() -> Iterator[TimelineElement]":
        return self._elements.__iter__

    @property
    def review_type(self):
        return self._review_type


class TimelineElement(object):
    def __init__(self, start_ns: int, end_ns: int, element_type: str, confidence: int = 100, extra_fields: Optional[Dict]=None):
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.element_type = element_type
        self.confidence = confidence
        if extra_fields is not None:
            self.extra_fields = extra_fields
        else:
            self.extra_fields = {}
        self.validate()

    def validate(self):
        assert type(self.start_ns) is int
        assert type(self.end_ns) is int
        assert type(self.element_type) is str
        assert type(self.confidence) is int
        assert type(self.extra_fields) is dict
        if self.start_ns > self.end_ns:
            raise RuntimeError(f"Impossible timeline element - it starts at {self.start_ns} ns and ends at "
                               f"{self.end_ns} ns")

    def drt_format(self, review_type: str):
        label = {
            "element_type": self.element_type,
            "range": [self.start_ns, self.end_ns],
            "confidence": self.confidence,
        }
        return {
            "review_type": review_type,
            "review_source": "model",
            "label": {**label, **self.extra_fields}
        }
