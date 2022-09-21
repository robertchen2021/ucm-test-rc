import inspect
from typing import Any, Dict, Optional, NamedTuple
import numpy as np


class ModelInput(object):
    video_in: Any  # usually np.array of frames
    video_out: Any  # usually np.array of frames
    snapshot_in: Any # usually np.array of image
    snapshot_out: Any # usually np.array of image
    sensor: Any  # usually nauto_datasets.core.sensors.<something>
    video_in_ts: Any
    video_out_ts: Any
    tfrecords: Any = None
    metadata: Dict = {}

    def __init__(self, metadata: Optional[Dict] = None):
        if metadata is not None:
            self.metadata = metadata

    def as_dict(self) -> Dict:
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        return {a[0]: a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))}

    def get(self, media_type: str) -> Any:
        return getattr(self, media_type.replace("-", "_"))

    def set(self, media_type: str, value: Any):
        setattr(self, media_type.replace("-", "_"), value)


class ModelResponse(object):
    def __init__(self, summary: str, score: float, confidence: int = 100, raw_output: Any = None):
        self.summary = summary  # for binary classification possible values are "TRUE" and "FALSE"
        self.score = score  # float 0 .. 1
        self.confidence = confidence
        self.raw_output = raw_output
        self.validate()

    def validate(self):
        if type(self.summary) is not str:
            raise RuntimeError(f"Summary should be string, {type(self.summary)} given")
        if type(self.confidence) is not int:
            raise RuntimeError(f"Confidence should be int, {type(self.confidence)} given")
        if type(self.score) is not float:
            raise RuntimeError(f"Score should be float, {type(self.score)} given")
        if not 1. >= self.score >= 0.:
            raise RuntimeError(f"Score should be between 0 and 1, {self.score} given")
        if not 100 >= self.confidence >= 0:
            raise RuntimeError(f"Confidence should be between 0 and 100, {self.confidence} given")


class ModelResponseCollection(object):
    class ModelResponseDecorator(NamedTuple):
        model_response: ModelResponse
        judgement_type: str
        judgement_subtype: Optional[str] = None

        def __getattr__(self, name):
            return self.model_response.__getattribute__(name)

        def is_timeline(self):
            from .timeline import ModelResponseTimeline
            return isinstance(self.model_response, ModelResponseTimeline)

    def __init__(self):
        self._items = []

    def add(self, model_response: ModelResponse, judgement_type: str, judgement_subtype: Optional[str] = None):
        self._items.append(self.ModelResponseDecorator(
            model_response=model_response,
            judgement_type=judgement_type,
            judgement_subtype=judgement_subtype
        ))

    def __iter__(self):
        return self._items.__iter__()

    def __len__(self):
        return len(self._items)
