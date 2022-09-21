from enum import Enum

from typing import Dict, List, NamedTuple

from nauto_datasets.protos import drt_pb2


class EventType(Enum):
    CALIBRATION = 'calibration'
    DISTRACTION = 'distraction'
    SEVERE_G_EVENT = 'severe-g-event'
    TAILGATING = 'tailgating'
    OBSTRUCTED_CAMERA = 'obstructed-camera'
    MARK = 'mark'
    DISTRACTION_RESEARCH = 'distraction-research'
    BRAKING_HARD = 'braking-hard'
    CRASHNET = 'crashnet'
    CUSTOM_COLLISION = 'custom-collision'

    def to_pb(self) -> int:
        """Serializes `EventType` as protobuf enum"""
        to_int = dict(zip(EventType, range(len(EventType))))
        return to_int[self]

    @staticmethod
    def from_pb(et_pb: int) -> 'EventType':
        """Reads `FileSource` from protobuf enum"""
        return list(EventType)[et_pb]


class JudgmentType(Enum):
    DISTRACTION = 'distraction'
    TAILGATING = 'tailgating'
    COLLISION = 'collision'
    QUESTIONAIRE = 'questionaire'
    CALIBRATION = 'calibration'
    LOOSE_DEVICE = 'loose-device'
    BAD_INSTALL = 'bad-install'
    OBSTRUCTED_CAMERA = 'obstructed-camera'
    WINDSHIELD_WIPERS = 'windshield-wipers'
    IR_CAM_ISSUE = 'ir-cam-issue'
    NATURAL_LANGUAGE_DESCRIPTION = 'natural-language-description'
    TIMELINE = 'timeline'
    COLLISION_TIMELINE = 'collision-timeline'

    def to_pb(self) -> int:
        """Serializes `JudgmentType` as protobuf enum"""
        to_int = dict(zip(JudgmentType, range(len(JudgmentType))))
        return to_int[self]

    @staticmethod
    def from_pb(jt_pb: int) -> 'JudgmentType':
        """Reads `FileSource` from protobuf enum"""
        return list(JudgmentType)[jt_pb]


class MediaType(Enum):
    SENSOR = 'sensor'
    SNAPSHOT_IN = 'snapshot-in'
    SNAPSHOT_OUT = 'snapshot-out'
    VIDEO_IN = 'video-in'
    VIDEO_OUT = 'video-out'
    MJPEG_IN = 'mjpeg-in'

    def to_pb(self) -> int:
        """Serializes `MediaType` as protobuf enum"""
        to_int = dict(zip(MediaType, range(len(MediaType))))
        return to_int[self]

    @staticmethod
    def from_pb(mt_pb: int) -> 'MediaType':
        """Reads `MediaType` from protobuf enum"""
        return list(MediaType)[mt_pb]
