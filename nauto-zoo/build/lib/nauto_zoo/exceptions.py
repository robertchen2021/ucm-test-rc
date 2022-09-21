from typing import List


class MalformedModelInputError(Exception):
    pass


class MalformedSensorGZipError(MalformedModelInputError):
    pass


class IncompleteInputMediaError(MalformedModelInputError):
    pass


class MissingSensorStreamError(IncompleteInputMediaError):
    def __init__(self, message: str, missing_streams: List[str]):
        super(MissingSensorStreamError, self).__init__(message)
        if type(missing_streams) is str:
            missing_streams = [missing_streams]
        self.missing_streams = missing_streams


class TooShortSensorStreamError(MalformedModelInputError):
    def __init__(self, too_short_stream_name: str, too_short_stream_length: int, required_length: int):
        message = f"Stream `{too_short_stream_name}` is too short - should have length at least {required_length}" \
                  f" but has length {too_short_stream_length}"
        super(TooShortSensorStreamError, self).__init__(message)
        self.too_short_stream_name = too_short_stream_name
        self.too_short_stream_length = too_short_stream_length
        self.required_length = required_length


class DoNotWantToProduceJudgementError(Exception):
    # identifies that input was understood but model does not want to produce a judgement for it
    pass
