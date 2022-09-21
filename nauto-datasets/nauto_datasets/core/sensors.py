import abc
import inspect
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Type, Dict

import numpy as np
from google.protobuf.message import Message as PbMessage

from nauto_datasets.core import streams
from nauto_datasets.core.streams import CombinedStreamMixin, StreamMixin
from nauto_datasets.utils import transformations as nauto_trans
from nauto_datasets.utils.numpy import NDArray, get_underlying_dtype
from nauto_datasets.utils.tuples import NamedTupleMetaEx
from sensor import sensor_pb2


def _get_named_tuple_from_pb(named_tuple_t: Type[NamedTuple],
                             pb2_instance: PbMessage,
                             **default_values: Dict[str, Any]) -> Any:
    def parse_value(value: Any, type_t: type) -> Any:
        try:
            und_type = get_underlying_dtype(type_t)
            return np.array(value, dtype=und_type)
        except Exception:
            return type_t(value)

    def default_value(name, field_t):
        try:
            return parse_value(getattr(pb2_instance, name), field_t)
        except:
            # this is required for overrides to work properly
            return None

    dct = {
        name: default_values.get(
            name,
            default_value(name, field_t))
        for name, field_t in named_tuple_t._field_types.items()
    }
    return named_tuple_t(**dct)


class PredictionsStream:
    """Interface intended to be implemented by streams
    with model predictions.

    Since such predictions might link to some model id in the `model_definition`
    list of `Recording` and the indices of corresponding models between
    different recordings might change we need to have a way to update
    this mappings making it consistent in the CombinedRecording.
    """

    def remap_model_ids(self, mapping: NDArray[int]) -> 'PredictionsStream':
        """Changes mapping of model predictions from model with id 'ind'
        to model with id 'mapping[ind]'
        """
        raise NotImplementedError()


class UtcTimeConvertible:

    def to_utc_time(
            self,
            utc_boot_time_ns: np.uint64,
            utc_boot_time_offset_ns: np.int64
    ) -> 'UtcTimeConvertible':
        raise NotImplementedError()


class SensorStream(UtcTimeConvertible):
    """SensorStream type used as a mixin to enrich sensor based streams"""

    def __new__(cls, *args, **kwargs) -> None:
        instance = super(SensorStream, cls).__new__(cls, *args, **kwargs)
        if not issubclass(cls, StreamMixin):
            raise ValueError(f'{cls} is not a StreamMixin')
        if 'sensor_ns' != cls._fields[0]:
            raise ValueError('{cls} is not indexed by sensor_ns')
        return instance

    def to_utc_time(
            self,
            utc_boot_time_ns: np.uint64,
            utc_boot_time_offset_ns: np.int64
    ) -> StreamMixin:
        """Returns a new `SensorStream` with sensor_ns tranformed to utc"""
        return self._replace(
            sensor_ns=nauto_trans.to_utc_time(
                self.sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False))


class CombinedUtcTimeConvertible:
    """CombinedUtcTimeConvertible type used as a mixin to enrich combined streams
    with utc time conversions
    """

    def __new__(cls, *args, **kwargs) -> None:
        instance = super(CombinedUtcTimeConvertible, cls).__new__(cls, *args, **kwargs)
        if not issubclass(cls, CombinedStreamMixin):
            raise ValueError(f'{cls} is not a CombinedStreamMixin')
        return instance

    def to_utc_time(
            self,
            utc_boot_time_ns: List[np.uint64],
            utc_boot_time_offset_ns: List[np.int64]
    ) -> 'CombinedUtcTimeConvertible':
        """Returns a new `CombinedStreamMixin` with times tranformed to utc
        Args:
            utc_boot_time_ns: an array of utc boot time starts for each
                substream belonging to a different recording. Should be of the
                same length as the number of substreams
            utc_boot_time_offset_ns: an array of additional nanosecond offsets
                for each substream belonging to a different recording. Should
                be of the same length as the number of substreams
        Returns:
            a new `CombinedStreamMixin` with all time based fields translated to utc time
        """
        if not issubclass(self.stream.__class__, UtcTimeConvertible):
            raise ValueError('Included stream is not UtcTimeConvertible')
        if not (len(utc_boot_time_ns)
                == len(utc_boot_time_offset_ns)
                == self._substreams_count()):
            raise ValueError(
                'Lengths of offsets are not the same as the number of substreams')

        return self.from_substreams([
            subs.to_utc_time(boot_time, boot_time_offset)
            for subs, boot_time, boot_time_offset
            in zip(self._substreams(), utc_boot_time_ns, utc_boot_time_offset_ns)
        ])


class FrameStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    score: NDArray[np.float]

    @staticmethod
    def from_pb(fs_pb: sensor_pb2.FrameStream) -> 'FrameStream':
        return _get_named_tuple_from_pb(FrameStream, fs_pb)


class AQBitStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    heading: NDArray[np.float]
    dt: NDArray[np.uint64]
    dp: NDArray[np.float]
    gyro_z: NDArray[np.float]

    @staticmethod
    def from_pb(aqb_pb: sensor_pb2.AQBitStream) -> 'AQBitStream':
        return _get_named_tuple_from_pb(AQBitStream, aqb_pb)


class XYZStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    x: NDArray[np.float]
    y: NDArray[np.float]
    z: NDArray[np.float]
    w: NDArray[np.float]

    @staticmethod
    def from_pb(xyz_pb: sensor_pb2.XYZStream) -> 'XYZStream':
        # Handle pre-2.4 messages with st_ms rather than sensor_ns
        if len(xyz_pb.sensor_ns) == 0:
            sensor_ns = nauto_trans.delta_decompress(
                np.array(xyz_pb.st_ms, dtype=np.uint64) * 1e6
            ).astype(np.uint64)
        else:
            sensor_ns = nauto_trans.delta_decompress(
                np.array(xyz_pb.sensor_ns, dtype=np.uint64))

        return XYZStream(
            sensor_ns=sensor_ns,
            system_ms=nauto_trans.delta_decompress(
                np.array(xyz_pb.system_ms, dtype=np.uint64)),
            x=nauto_trans.delta_decompress(
                np.array(xyz_pb.x), xyz_pb.scale).astype(np.float),
            y=nauto_trans.delta_decompress(
                np.array(xyz_pb.y), xyz_pb.scale).astype(np.float),
            z=nauto_trans.delta_decompress(
                np.array(xyz_pb.z), xyz_pb.scale).astype(np.float),
            w=nauto_trans.delta_decompress(
                np.array(xyz_pb.w), xyz_pb.scale).astype(np.float))

    def clip_x(self, max_value: float) -> 'XYZStream':
        return XYZStream(
            x=np.clip(self.x, a_min=-max_value, a_max=max_value),
            y=self.y,
            z=self.z,
            w=self.w,
            sensor_ns=self.sensor_ns,
            system_ms=self.system_ms
        )

    def clip_y(self, max_value: float) -> 'XYZStream':
        return XYZStream(
            x=self.x,
            y=np.clip(self.y, a_min=-max_value, a_max=max_value),
            z=self.z,
            w=self.w,
            sensor_ns=self.sensor_ns,
            system_ms=self.system_ms
        )

    def clip_z(self, max_value: float) -> 'XYZStream':
        return XYZStream(
            x=self.x,
            y=self.y,
            z=np.clip(self.z, a_min=-max_value, a_max=max_value),
            w=self.w,
            sensor_ns=self.sensor_ns,
            system_ms=self.system_ms
        )

    def clip(self, max_value: float) -> 'XYZStream':
        stream = self.clip_x(max_value)
        stream = stream.clip_y(max_value)
        return stream.clip_z(max_value)

    def cut(self, max_length: int, cut_beginning: bool = False) -> 'XYZStream':
        if cut_beginning:
            start = max(0, self.x.shape[0] - max_length)
            return XYZStream(
                x=self.x[start:],
                y=self.y[start:],
                z=self.z[start:],
                w=self.w[start:],
                sensor_ns=self.sensor_ns[start:],
                system_ms=self.system_ms[start:]
            )
        else:
            return XYZStream(
                x=self.x[:max_length],
                y=self.y[:max_length],
                z=self.z[:max_length],
                w=self.w[:max_length],
                sensor_ns=self.sensor_ns[:max_length],
                system_ms=self.system_ms[:max_length]
            )

    def find_peak(self) -> int:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2).argmax()


class GPSStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    latitude: NDArray[np.float]
    longitude: NDArray[np.float]
    accuracy: NDArray[np.float]
    speed: NDArray[np.float]
    gps_ms: NDArray[np.uint64]
    bearing: NDArray[np.float]
    sv_count: NDArray[np.int32]
    pdop: NDArray[np.float]
    hdop: NDArray[np.float]
    vdop: NDArray[np.float]
    altitude: NDArray[np.float]

    @staticmethod
    def from_pb(gps_pb: sensor_pb2.GPSStream) -> 'GPSStream':
        return _get_named_tuple_from_pb(GPSStream, gps_pb)


class SpeedSource(Enum):
    UNKNOWN = 0
    GPS = 1
    OBD = 2


class SpeedStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    speed: NDArray[np.float]

    speed_source: NDArray[np.int]

    @staticmethod
    def from_pb(gps_pb: sensor_pb2.SpeedStream) -> 'SpeedStream':
        return _get_named_tuple_from_pb(SpeedStream, gps_pb)


class PowerConsumptionStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    voltage_v: NDArray[np.float]
    current_ma: NDArray[np.uint64]
    power_mw: NDArray[np.float]

    @staticmethod
    def from_pb(gps_pb: sensor_pb2.PowerConsumptionStream) -> 'PowerConsumptionStream':
        return _get_named_tuple_from_pb(PowerConsumptionStream, gps_pb)


class BrakingDistanceStream(StreamMixin, metaclass=NamedTupleMetaEx):
    system_ms: NDArray[np.uint64]
    speed_sensor_ns: NDArray[np.uint64]
    acc_sensor_ns: NDArray[np.uint64]

    predicted_braking_distance_m: NDArray[np.float]
    predicted_distracted_braking_distance_m: NDArray[np.float]

    @staticmethod
    def from_pb(braking_pb: sensor_pb2.BrakingDistanceStream) -> 'BrakingDistanceStream':
        return BrakingDistanceStream(
            system_ms=nauto_trans.delta_decompress(
                np.array(braking_pb.system_ms, dtype=np.uint64)),
            speed_sensor_ns=nauto_trans.delta_decompress(
                np.array(braking_pb.speed_sensor_ns, dtype=np.uint64)),
            acc_sensor_ns=nauto_trans.delta_decompress(
                np.array(braking_pb.acc_sensor_ns, dtype=np.uint64)),
            predicted_braking_distance_m=nauto_trans.delta_decompress(
                np.array(braking_pb.predicted_braking_distance_m), braking_pb.scale).astype(np.float),
            predicted_distracted_braking_distance_m=nauto_trans.delta_decompress(np.array(
                braking_pb.predicted_distracted_braking_distance_m), braking_pb.scale).astype(np.float),
        )

    def to_utc_time(
            self,
            utc_boot_time_ns: np.uint64,
            utc_boot_time_offset_ns: np.int64
    ) -> StreamMixin:
        """Returns a new `SensorStream` with sensor_ns tranformed to utc"""
        return self._replace(
            speed_sensor_ns=nauto_trans.to_utc_time(
                self.speed_sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False),
            acc_sensor_ns=nauto_trans.to_utc_time(
                self.acc_sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False))


class FCWStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    ttc: NDArray[np.float]
    distance_estimate: NDArray[np.float]

    @staticmethod
    def from_pb(fcw_pb: sensor_pb2.FCWStream) -> 'FCWStream':
        return _get_named_tuple_from_pb(FCWStream, fcw_pb)


class EKFConfig(metaclass=NamedTupleMetaEx):
    rot_angle_x: np.float
    rot_angle_y: np.float
    rot_count: np.uint32
    sigma_ax: np.float

    config_ts: np.uint64
    state_vector_length: np.uint32
    config_x: NDArray[np.float]
    config_p: NDArray[np.float]
    config_r: NDArray[np.float]

    @staticmethod
    def from_pb(ekf_pb: sensor_pb2.EKFStream) -> 'EKFConfig':
        return _get_named_tuple_from_pb(EKFConfig, ekf_pb)


class EKFStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    acc_x: NDArray[np.float]
    acc_y: NDArray[np.float]
    acc_z: NDArray[np.float]
    gyr_x: NDArray[np.float]
    gyr_y: NDArray[np.float]
    gyr_z: NDArray[np.float]
    mag_x: NDArray[np.float]
    mag_y: NDArray[np.float]
    mag_z: NDArray[np.float]
    grv_x: NDArray[np.float]
    grv_y: NDArray[np.float]
    grv_z: NDArray[np.float]
    latitude: NDArray[np.float]
    longitude: NDArray[np.float]
    speed: NDArray[np.float]
    heading: NDArray[np.float]
    accuracy: NDArray[np.float]
    rsv1: NDArray[np.float]
    rsv2: NDArray[np.float]
    rsv3: NDArray[np.float]
    rsv4: NDArray[np.float]
    rsv5: NDArray[np.float]

    @staticmethod
    def from_pb(ekf_pb: sensor_pb2.EKFStream) -> 'EKFStream':
        return _get_named_tuple_from_pb(EKFStream, ekf_pb)


class ObdStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]

    brick_code: NDArray[np.str]
    value: NDArray[np.float]
    brick_ns: NDArray[np.uint64]

    @staticmethod
    def from_pb(obd_pb: sensor_pb2.ObdStream) -> 'ObdStream':
        return _get_named_tuple_from_pb(ObdStream, obd_pb)


class DeviceOrientationStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    name: NDArray[np.str]
    n_samples: NDArray[np.uint64]
    pitch: NDArray[np.float]
    roll: NDArray[np.float]
    yaw: NDArray[np.float]
    converged: NDArray[np.bool]
    shadow_mode: NDArray[np.bool]
    is_accel_algo: NDArray[np.bool]
    accel_x: NDArray[np.float]
    accel_y: NDArray[np.float]
    accel_z: NDArray[np.float]
    theta_x: NDArray[np.float]
    theta_y: NDArray[np.float]
    theta_z: NDArray[np.float]

    @staticmethod
    def from_pb(
            do_pb: sensor_pb2.DeviceOrientationStream
    ) -> 'DeviceOrientationStream':
        return _get_named_tuple_from_pb(DeviceOrientationStream, do_pb)


class LooseDeviceStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    heuristic_score: NDArray[np.float]

    @staticmethod
    def from_pb(ld_pb: sensor_pb2.LooseDeviceStream) -> 'LooseDeviceStream':
        return _get_named_tuple_from_pb(LooseDeviceStream, ld_pb)


class DistractionStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    score_looking_down: NDArray[np.float]
    score_looking_up: NDArray[np.float]
    score_looking_left: NDArray[np.float]
    score_looking_right: NDArray[np.float]
    score_cell_phone: NDArray[np.float]
    score_smoking: NDArray[np.float]
    score_holding_object: NDArray[np.float]
    score_eyes_closed: NDArray[np.float]
    score_no_face: NDArray[np.float]
    score_no_seat_belt: NDArray[np.float]

    @staticmethod
    def from_pb(dist_pb: sensor_pb2.DistractionStream) -> 'DistractionStream':
        return _get_named_tuple_from_pb(DistractionStream, dist_pb)


class DrowsinessConfig(metaclass=NamedTupleMetaEx):
    threshold: np.float

    @staticmethod
    def from_pb(drowsy_pb: sensor_pb2.DrowsinessStream) -> 'DrowsinessConfig':
        return _get_named_tuple_from_pb(DrowsinessConfig, drowsy_pb)


class DrowsinessStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    score: NDArray[np.float]
    isDrowsy: NDArray[np.bool]

    @staticmethod
    def from_pb(drowsy_pb: sensor_pb2.DrowsinessStream) -> 'DrowsinessStream':
        return _get_named_tuple_from_pb(DrowsinessStream, drowsy_pb)


class AppliedOrientationStream(
        StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    yaw: NDArray[np.float]
    roll: NDArray[np.float]
    pitch: NDArray[np.float]

    @staticmethod
    def from_pb(
            ao_pb: sensor_pb2.AppliedOrientation
    ) -> 'AppliedOrientationStream':
        return _get_named_tuple_from_pb(AppliedOrientationStream, ao_pb)


class TailgatingStream(
        StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    score: NDArray[np.float32]
    distance_estimate: NDArray[np.float32]
    front_box_index: NDArray[np.int32]

    @staticmethod
    def from_pb(
            t_pb: sensor_pb2.TailgatingStream
    ) -> 'TailgatingStream':
        return _get_named_tuple_from_pb(TailgatingStream, t_pb)


class BoundingBox(StreamMixin, metaclass=NamedTupleMetaEx):
    left: NDArray[np.float32]
    top: NDArray[np.float32]
    right: NDArray[np.float32]
    bottom: NDArray[np.float32]

    objectType: NDArray[np.int32]
    score: NDArray[np.float32]

    def get_bbox_array(self) -> NDArray[np.float32]:
        """Returns an array of shape [num_boxes, 4] with values
        [left, top, right, bottom]"""
        return np.stack([self.left, self.top, self.right, self.bottom],
                        axis=1)

    def get_bbox_sizes(self) -> NDArray[np.float32]:
        """Returns an array of shape [num_boxes, 2] with values [width, height]"""
        return np.stack([self.right - self.left, self.bottom - self.top],
                        axis=1)

    def get_bbox_areas(self) -> NDArray[np.float32]:
        """Returns an array of shape [num_boxes] with bounding box areas"""
        return np.prod(self.get_bbox_sizes(), axis=1)

    def to_normalized_coordinates(
            self,
            image_width: int,
            image_height: int
    ) -> 'BoundingBox':
        """Normalizes bbox coordinates to [0,1] range with respect to the image size"""
        if image_width * image_height == 0:
            raise ValueError('Image should have positive size')
        return self._replace(
            left=self.left / image_width,
            top=self.top / image_height,
            right=self.right / image_width,
            bottom=self.bottom / image_height)

    def to_absolute_coordinates(
            self,
            image_width: int,
            image_height: int
    ) -> 'BoundingBox':
        """Transforms bbox coordinates from [0,1] range to pixel coordinates"""
        if image_width * image_height == 0:
            raise ValueError('Image should have positive size')
        return self._replace(
            left=self.left * image_width,
            top=self.top * image_height,
            right=self.right * image_width,
            bottom=self.bottom * image_height)

    @staticmethod
    def from_pb(bb_pb: sensor_pb2.BoundingBox) -> 'BoundingBox':
        return _get_named_tuple_from_pb(BoundingBox, bb_pb)


class BoundingBoxStream(
        StreamMixin,
        SensorStream,
        PredictionsStream,
        metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]

    bounding_box: List[BoundingBox]
    model_id: NDArray[np.int32]

    @abc.abstractmethod
    def remap_model_ids(self, mapping: NDArray[np.int32]) -> 'PredictionsStream':
        return self._replace(model_id=mapping[self.model_id])

    @staticmethod
    def from_pb(bb_s_pb: sensor_pb2.BoundingBox) -> 'BoundingBoxStream':
        bounding_boxes = [
            BoundingBox.from_pb(bb_pb) for bb_pb in bb_s_pb.bounding_box
        ]
        return _get_named_tuple_from_pb(BoundingBoxStream,
                                        bb_s_pb,
                                        bounding_box=bounding_boxes)


class ImuStatisticsStream(StreamMixin,
                          UtcTimeConvertible,
                          metaclass=NamedTupleMetaEx):
    accel_sensor_ns: NDArray[np.uint64]
    accel_system_ms: NDArray[np.uint64]
    accel_sample_count: NDArray[np.int32]
    ax_first_moment: NDArray[np.float]
    ax_second_moment: NDArray[np.float]
    ax_third_moment: NDArray[np.float]
    ax_fourth_moment: NDArray[np.float]
    ay_first_moment: NDArray[np.float]
    ay_second_moment: NDArray[np.float]
    ay_third_moment: NDArray[np.float]
    ay_fourth_moment: NDArray[np.float]
    az_first_moment: NDArray[np.float]
    az_second_moment: NDArray[np.float]
    az_third_moment: NDArray[np.float]
    az_fourth_moment: NDArray[np.float]
    gyro_sensor_ns: NDArray[np.uint64]
    gyro_system_ms: NDArray[np.uint64]
    gyro_sample_count: NDArray[np.int32]
    gx_first_moment: NDArray[np.float]
    gx_second_moment: NDArray[np.float]
    gx_third_moment: NDArray[np.float]
    gx_fourth_moment: NDArray[np.float]
    gy_first_moment: NDArray[np.float]
    gy_second_moment: NDArray[np.float]
    gy_third_moment: NDArray[np.float]
    gy_fourth_moment: NDArray[np.float]
    gz_first_moment: NDArray[np.float]
    gz_second_moment: NDArray[np.float]
    gz_third_moment: NDArray[np.float]
    gz_fourth_moment: NDArray[np.float]

    def to_utc_time(
            self,
            utc_boot_time_ns: np.uint64,
            utc_boot_time_offset_ns: np.int64
    ) -> StreamMixin:
        """Returns a new `SensorStream` with sensor_ns tranformed to utc"""
        return self._replace(
            accel_sensor_ns=nauto_trans.to_utc_time(
                self.accel_sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False),
            gyro_sensor_ns=nauto_trans.to_utc_time(
                self.gyro_sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False))

    @staticmethod
    def from_pb(ims_pb: sensor_pb2.ImuStatisticsStream) -> 'ImuStatisticsStream':
        return _get_named_tuple_from_pb(ImuStatisticsStream, ims_pb)


class Severity(Enum):
    UNSET = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Status(Enum):
    UNKNOWN = 0
    STARTED = 1
    ONGOING = 2
    FINISHED = 3


class RiskAssessmentTriggeringDetail(metaclass=NamedTupleMetaEx):
    distraction_start_sensor_ns: np.uint64
    ttc_start_sensor_ns: np.uint64

    speed_sensor_ns: np.uint64

    @staticmethod
    def from_pb(ratd_pb: sensor_pb2.RiskAssessmentTriggeringDetail) -> 'RiskAssessmentTriggeringDetail':
        return _get_named_tuple_from_pb(RiskAssessmentTriggeringDetail, ratd_pb)


class RiskAssessmentSuppressingDetail(metaclass=NamedTupleMetaEx):
    deliberate_braking_start_sensor_ns: np.uint64

    @staticmethod
    def from_pb(rasd_pb: sensor_pb2.RiskAssessmentSuppressingDetail) -> 'RiskAssessmentSuppressingDetail':
        return _get_named_tuple_from_pb(RiskAssessmentSuppressingDetail, rasd_pb)


class RiskType(Enum):
    FCW = 0
    FCW_DISTRACTION = 1


class RiskAssessmentStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]
    type: NDArray[np.int]
    status: NDArray[np.int]
    start_timestamp_ns: NDArray[np.uint64]
    end_timestamp_ns: NDArray[np.uint64]
    risk_severity: NDArray[np.int]
    should_play_rta: NDArray[np.bool]
    suppressed: NDArray[np.bool]

    @staticmethod
    def from_pb(ra_pb: sensor_pb2.RiskAssessmentStream) -> 'RiskAssessmentStream':
        return _get_named_tuple_from_pb(RiskAssessmentStream, ra_pb)


class RiskRtaStream(StreamMixin, metaclass=NamedTupleMetaEx):
    risk_start_sensor_ns: NDArray[np.uint64]
    risk_start_system_ms: NDArray[np.uint64]
    risk_type: NDArray[np.int]
    rta_requested_sensor_ns: NDArray[np.uint64]

    @staticmethod
    def from_pb(rr_pb: sensor_pb2.RiskRtaStream) -> 'RiskRtaStream':
        return _get_named_tuple_from_pb(RiskRtaStream, rr_pb)

    def to_utc_time(
            self,
            utc_boot_time_ns: np.uint64,
            utc_boot_time_offset_ns: np.int64
    ) -> StreamMixin:
        """Returns a new `SensorStream` with sensor_ns tranformed to utc"""
        return self._replace(
            risk_start_sensor_ns=nauto_trans.to_utc_time(
                self.risk_start_sensor_ns,
                utc_boot_time_ns,
                utc_boot_time_offset_ns,
                in_place=False))


class State(Enum):
    UNKNOWN = 0
    MOVING = 1
    STOPPED = 2
    PARKED = 3
    LEFT = 4
    RIGHT = 5


class MessageType(Enum):
    NONE = 0
    FILTERED = 1
    SEVERE_G = 2
    LOWERED_G = 3
    ACCELERATION = 4
    BRAKING = 5
    CORNER_LEFT = 6
    CORNER_RIGHT = 7
    LEFT_RIGHT_TURN = 8
    MOVING_STOPPED = 9
    DELIBERATE_BRAKING = 10
    DEVICE_ORIENTATION = 11


class ManeuverStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    sensor_ns: NDArray[np.uint64]
    system_ms: NDArray[np.uint64]
    message_type: NDArray[np.int]
    algorithm_version: NDArray[np.str]
    threshold_file_info: NDArray[np.str]
    event_status: NDArray[np.int]
    start_timestamp_ns: NDArray[np.uint64]
    end_timestamp_ns: NDArray[np.uint64]
    duration_ns: NDArray[np.int64]
    start_speed_mps: NDArray[np.float]
    end_speed_mps: NDArray[np.float]
    peak_timestamp_ns: NDArray[np.uint64]
    peak_speed_mps: NDArray[np.float]
    peak_acceleration_mps2: NDArray[np.float]
    peak_rotation_rad_s: NDArray[np.float]

    moving_state: NDArray[np.int]
    direction: NDArray[np.int]
    severity: NDArray[np.int]
    original_message_type: NDArray[np.int]
    filtered_reason: NDArray[np.str]
    applied_orientation_x_rad: NDArray[np.float]
    applied_orientation_y_rad: NDArray[np.float]
    applied_orientation_z_rad: NDArray[np.float]

    @staticmethod
    def from_pb(m_pb: sensor_pb2.ManeuverStream) -> 'ManeuverStream':
        return _get_named_tuple_from_pb(ManeuverStream, m_pb)


class TemperatureStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    system_ms: NDArray[np.uint64]

    cpu_temperature_celsius: NDArray[np.float]
    gpu_temperature_celsius: NDArray[np.float]
    computation_capacity: NDArray[np.float]
    sensor_ns: NDArray[np.uint64]

    @staticmethod
    def from_pb(temperature_pb: sensor_pb2.TemperatureStream) -> 'TemperatureStream':
        return _get_named_tuple_from_pb(TemperatureStream, temperature_pb)


class OmniFusionReportTag(Enum):
    UNDEFINED = 0
    GPS_RECEIVED = 1
    GPS_PROCESSED = 2
    IMU_RECEIVED = 3
    IMU_PROCESSED = 4
    GPS_RAW = 5


class DeltaCompressedSint64:
    scale: np.uint32
    delte_compresed_data: NDArray[np.int64]


class OmniFusionVehicleDynamicsStream(
        StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    batch_sensor_ns: NDArray[np.uint64]
    batch_system_ms: NDArray[np.uint64]
    sample_done_sensor_ns: NDArray[np.uint64]
    sensor_ns: NDArray[np.uint64]
    initialized: NDArray[np.bool]
    report_tag: NDArray[np.int]
    xfilt_accel: NDArray[np.float]
    xfilt_brake: NDArray[np.float]
    xfilt_lcorn: NDArray[np.float]
    xfilt_rcorn: NDArray[np.float]

    @staticmethod
    def from_pb(ofvd_pb: sensor_pb2.OmniFusionVehicleDynamicsStream
                ) -> 'OmniFusionVehicleDynamicsStream':
        return OmniFusionVehicleDynamicsStream(
            sensor_ns=nauto_trans.delta_decompress(
                np.array(ofvd_pb.sensor_ns, dtype=np.uint64)),
            batch_sensor_ns=nauto_trans.delta_decompress(
                np.array(ofvd_pb.batch_sensor_ns, dtype=np.uint64)),
            batch_system_ms=nauto_trans.delta_decompress(
                np.array(ofvd_pb.batch_system_ms, dtype=np.uint64)),
            sample_done_sensor_ns=nauto_trans.delta_decompress(
                np.array(ofvd_pb.sample_done_sensor_ns, dtype=np.uint64)),
            initialized=np.array(ofvd_pb.initialized, dtype=np.bool),
            report_tag=np.array(ofvd_pb.report_tag, dtype=np.int),
            xfilt_accel=nauto_trans.delta_decompress(
                np.array(ofvd_pb.xfilt_accel.delte_compresed_data),
                ofvd_pb.xfilt_accel.scale).astype(np.float),
            xfilt_brake=nauto_trans.delta_decompress(
                np.array(ofvd_pb.xfilt_brake.delte_compresed_data),
                ofvd_pb.xfilt_brake.scale).astype(np.float),
            xfilt_lcorn=nauto_trans.delta_decompress(
                np.array(ofvd_pb.xfilt_lcorn.delte_compresed_data),
                ofvd_pb.xfilt_lcorn.scale).astype(np.float),
            xfilt_rcorn=nauto_trans.delta_decompress(
                np.array(ofvd_pb.xfilt_rcorn.delte_compresed_data),
                ofvd_pb.xfilt_rcorn.scale).astype(np.float)
        )


class OmniFusionLocationStream(StreamMixin, SensorStream, metaclass=NamedTupleMetaEx):
    batch_sensor_ns: NDArray[np.uint64]
    batch_system_ms: NDArray[np.uint64]
    sample_done_sensor_ns: NDArray[np.uint64]
    sensor_ns: NDArray[np.uint64]
    initialized: NDArray[np.bool]
    report_tag: NDArray[np.int]
    latitude_deg: NDArray[np.float]
    latitude_accuracy_deg: NDArray[np.float]
    longitude_deg: NDArray[np.float]
    longitude_accuracy_deg: NDArray[np.float]
    altitude_m: NDArray[np.float]
    altitude_accuracy_m: NDArray[np.float]
    speed_mps: NDArray[np.float]
    speed_accuracy_mps: NDArray[np.float]
    bearing_deg: NDArray[np.float]
    bearing_accuracy_deg: NDArray[np.float]
    longitudinal_velocity_mps: NDArray[np.float]
    longitudinal_velocity_accuracy_mps: NDArray[np.float]
    north_velocity_mps: NDArray[np.float]
    north_velocity_accuracy_mps: NDArray[np.float]
    east_velocity_mps: NDArray[np.float]
    east_velocity_accuracy_mps: NDArray[np.float]
    down_velocity_mps: NDArray[np.float]
    down_velocity_accuracy_mps: NDArray[np.float]

    @staticmethod
    def from_pb(ofls_pb: sensor_pb2.OmniFusionLocationStream
                ) -> 'OmniFusionLocationStream':
        return OmniFusionLocationStream(
            sensor_ns=nauto_trans.delta_decompress(
                np.array(ofls_pb.sensor_ns, dtype=np.uint64)),
            batch_sensor_ns=nauto_trans.delta_decompress(
                np.array(ofls_pb.batch_sensor_ns, dtype=np.uint64)),
            batch_system_ms=nauto_trans.delta_decompress(
                np.array(ofls_pb.batch_system_ms, dtype=np.uint64)),
            sample_done_sensor_ns=nauto_trans.delta_decompress(
                np.array(ofls_pb.sample_done_sensor_ns, dtype=np.uint64)),
            initialized=np.array(ofls_pb.initialized, dtype=np.bool),
            report_tag=np.array(ofls_pb.report_tag, dtype=np.int),
            latitude_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.latitude_deg.delte_compresed_data),
                ofls_pb.latitude_deg.scale).astype(np.float),
            latitude_accuracy_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.latitude_accuracy_deg.delte_compresed_data),
                ofls_pb.latitude_accuracy_deg.scale).astype(np.float),
            longitude_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.longitude_deg.delte_compresed_data),
                ofls_pb.longitude_deg.scale).astype(np.float),
            longitude_accuracy_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.longitude_accuracy_deg.delte_compresed_data),
                ofls_pb.longitude_accuracy_deg.scale).astype(np.float),
            altitude_m=nauto_trans.delta_decompress(
                np.array(ofls_pb.altitude_m.delte_compresed_data),
                ofls_pb.altitude_m.scale).astype(np.float),
            altitude_accuracy_m=nauto_trans.delta_decompress(
                np.array(ofls_pb.altitude_accuracy_m.delte_compresed_data),
                ofls_pb.altitude_accuracy_m.scale).astype(np.float),
            speed_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.speed_mps.delte_compresed_data),
                ofls_pb.speed_mps.scale).astype(np.float),
            speed_accuracy_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.speed_accuracy_mps.delte_compresed_data),
                ofls_pb.speed_accuracy_mps.scale).astype(np.float),
            bearing_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.bearing_deg.delte_compresed_data),
                ofls_pb.bearing_deg.scale).astype(np.float),
            bearing_accuracy_deg=nauto_trans.delta_decompress(
                np.array(ofls_pb.bearing_accuracy_deg.delte_compresed_data),
                ofls_pb.bearing_accuracy_deg.scale).astype(np.float),
            longitudinal_velocity_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.longitudinal_velocity_mps.delte_compresed_data),
                ofls_pb.longitudinal_velocity_mps.scale).astype(np.float),
            longitudinal_velocity_accuracy_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.longitudinal_velocity_accuracy_mps.delte_compresed_data),
                ofls_pb.longitudinal_velocity_accuracy_mps.scale).astype(np.float),
            north_velocity_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.north_velocity_mps.delte_compresed_data),
                ofls_pb.north_velocity_mps.scale).astype(np.float),
            north_velocity_accuracy_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.north_velocity_accuracy_mps.delte_compresed_data),
                ofls_pb.north_velocity_accuracy_mps.scale).astype(np.float),
            east_velocity_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.east_velocity_mps.delte_compresed_data),
                ofls_pb.east_velocity_mps.scale).astype(np.float),
            east_velocity_accuracy_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.east_velocity_accuracy_mps.delte_compresed_data),
                ofls_pb.east_velocity_accuracy_mps.scale).astype(np.float),
            down_velocity_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.down_velocity_mps.delte_compresed_data),
                ofls_pb.down_velocity_mps.scale).astype(np.float),
            down_velocity_accuracy_mps=nauto_trans.delta_decompress(
                np.array(ofls_pb.down_velocity_accuracy_mps.delte_compresed_data),
                ofls_pb.down_velocity_accuracy_mps.scale).astype(np.float)
        )


class Vertex(metaclass=NamedTupleMetaEx):
    x_px: np.uint32
    y_px: np.uint32
    x_scaled: np.float32
    y_scaled: np.float32

    @staticmethod
    def from_pb(vertex_pb: sensor_pb2.Vertex) -> 'Vertex':
        return _get_named_tuple_from_pb(Vertex, vertex_pb)


class Polygon(metaclass=NamedTupleMetaEx):
    vertices: List[Vertex]

    @staticmethod
    def from_pb(polygon_pb: sensor_pb2.Polygon) -> 'Polygon':
        vertices = [Vertex.from_pb(pb) for pb in polygon_pb.vertices]
        return _get_named_tuple_from_pb(Polygon,
                                        polygon_pb,
                                        vertices=vertices
                                        )


class CenterlinePolygonStream(StreamMixin, metaclass=NamedTupleMetaEx):
    polygon: List[Polygon]
    frame_image_system_ms: NDArray[np.uint64]
    frame_image_sensor_ns: NDArray[np.uint64]
    frame_mcod_system_ms: NDArray[np.uint64]
    frame_mcod_sensor_ns: NDArray[np.uint64]

    @staticmethod
    def from_pb(centerline_pb: sensor_pb2.CenterlinePolygonStream
                ) -> 'CenterlinePolygonStream':
        polygon = [Polygon.from_pb(pb) for pb in centerline_pb.polygon]
        return _get_named_tuple_from_pb(CenterlinePolygonStream,
                                        centerline_pb,
                                        polygon=polygon)


class DistanceEstimationParameters(metaclass=NamedTupleMetaEx):
    slope: np.float
    intercept: np.float

    @staticmethod
    def from_pb(de_pb: sensor_pb2.DistanceEstimationParameters
                ) -> 'DistanceEstimationParameters':
        return _get_named_tuple_from_pb(DistanceEstimationParameters, de_pb)

class AutoCalibrationStream(StreamMixin, metaclass=NamedTupleMetaEx):
    crop: List[BoundingBox]
    calibration_triangle: List[Polygon]
    distance_estimation_parameters: List[DistanceEstimationParameters]
    system_ms: NDArray[np.uint64]

    @staticmethod
    def from_pb(ac_pb: sensor_pb2.AutoCalibrationStream
                ) -> 'AutoCalibrationStream':
        crop = [BoundingBox.from_pb(bb_pb) for bb_pb in ac_pb.crop]
        dep = [DistanceEstimationParameters.from_pb(de_pb) for de_pb in ac_pb.distance_estimation_parameters]
        calibration_triangle = [Polygon.from_pb(pb) for pb in ac_pb.calibration_triangle]
        return _get_named_tuple_from_pb(AutoCalibrationStream,
                                        ac_pb,
                                        crop=crop,
                                        distance_estimation_parameters=dep,
                                        calibration_triangle=calibration_triangle
                                        )


class ModelDefinition(metaclass=NamedTupleMetaEx):
    name: str
    version: str

    @staticmethod
    def from_pb(md_pb: sensor_pb2.ModelDefinition) -> 'ModelDefinition':
        return _get_named_tuple_from_pb(ModelDefinition, md_pb)


def _maybe_empty_stream(
        stream_t: Type[StreamMixin], pb: Optional[PbMessage]
) -> Type[StreamMixin]:
    if pb is None:
        return stream_t.empty()
    else:
        return stream_t.from_pb(pb)


class RecordingMetadata(metaclass=NamedTupleMetaEx):
    capture_start_system_ms: np.uint64
    utc_boot_time_ns: np.uint64
    version: np.str
    app_session: np.uint32
    boot_session: np.uint32
    utc_boot_time_offset_ns: np.int64

    @staticmethod
    def from_pb(r_pb: sensor_pb2.Recording) -> 'RecordingMetadata':
        return _get_named_tuple_from_pb(RecordingMetadata, r_pb)


class Recording(metaclass=NamedTupleMetaEx):
    acc: XYZStream
    gyro: XYZStream
    grv: XYZStream
    mag: XYZStream
    lin: XYZStream
    gps: GPSStream
    speed: SpeedStream
    dist: FrameStream
    tail: FrameStream
    aqbit: AQBitStream
    ekf: EKFStream
    ekf_config: EKFConfig
    obd: ObdStream
    device_orientation: DeviceOrientationStream
    dist_multilabel: DistractionStream
    drowsiness: DrowsinessStream
    drowsiness_config: DrowsinessConfig
    loose_device: LooseDeviceStream
    applied_orientation: AppliedOrientationStream
    oriented_acc: XYZStream
    oriented_gyro: XYZStream
    bounding_boxes_external: BoundingBoxStream
    mcod_bounding_boxes: BoundingBoxStream
    tailgating: TailgatingStream
    imu_statistics: ImuStatisticsStream
    rt_acc: XYZStream
    rt_gyro: XYZStream
    power_consumption: PowerConsumptionStream
    rt_oriented_acc: XYZStream
    rt_oriented_gyro: XYZStream
    rt_device_orientation: DeviceOrientationStream
    fcw: FCWStream
    risk_assessment: RiskAssessmentStream
    risk_rta: RiskRtaStream
    maneuver: ManeuverStream
    temperature: TemperatureStream
    braking_distance: BrakingDistanceStream
    omnifusion_vehicle_dynamics: OmniFusionVehicleDynamicsStream
    omnifusion_location: OmniFusionLocationStream
    auto_calibration: AutoCalibrationStream
    model_definition: List[ModelDefinition]
    centerline_polygon: CenterlinePolygonStream
    metadata: RecordingMetadata

    @staticmethod
    def from_pb(r_pb: sensor_pb2.Recording) -> 'Recording':
        return Recording(
            acc=_maybe_empty_stream(XYZStream, r_pb.acc),
            gyro=_maybe_empty_stream(XYZStream, r_pb.gyro),
            grv=_maybe_empty_stream(XYZStream, r_pb.grv),
            mag=_maybe_empty_stream(XYZStream, r_pb.mag),
            lin=_maybe_empty_stream(XYZStream, r_pb.lin),
            gps=_maybe_empty_stream(GPSStream, r_pb.gps),
            speed=_maybe_empty_stream(SpeedStream, r_pb.speed),
            dist=_maybe_empty_stream(FrameStream, r_pb.dist),
            tail=_maybe_empty_stream(FrameStream, r_pb.tail),
            aqbit=_maybe_empty_stream(AQBitStream, r_pb.aqbit),
            ekf=_maybe_empty_stream(EKFStream, r_pb.ekf),
            ekf_config=(
                EKFConfig.from_pb(r_pb.ekf) if r_pb.ekf is not None else None),
            obd=_maybe_empty_stream(ObdStream, r_pb.obd),
            device_orientation=_maybe_empty_stream(
                DeviceOrientationStream, r_pb.device_orientation),
            dist_multilabel=_maybe_empty_stream(
                DistractionStream, r_pb.dist_multilabel),
            drowsiness=_maybe_empty_stream(
                DrowsinessStream, r_pb.drowsiness),
            drowsiness_config=(
                DrowsinessConfig.from_pb(r_pb.drowsiness)
                if r_pb.drowsiness is not None else None),
            loose_device=_maybe_empty_stream(
                LooseDeviceStream, r_pb.loose_device),
            applied_orientation=_maybe_empty_stream(
                AppliedOrientationStream, r_pb.applied_orientation),
            oriented_acc=_maybe_empty_stream(
                XYZStream, r_pb.oriented_acc),
            oriented_gyro=_maybe_empty_stream(
                XYZStream, r_pb.oriented_gyro),
            bounding_boxes_external=_maybe_empty_stream(
                BoundingBoxStream, r_pb.bounding_boxes_external),
            mcod_bounding_boxes=_maybe_empty_stream(
                BoundingBoxStream, r_pb.mcod_bounding_boxes),
            tailgating=_maybe_empty_stream(
                TailgatingStream, r_pb.tailgating),
            imu_statistics=_maybe_empty_stream(
                ImuStatisticsStream, r_pb.imu_statistics),
            rt_acc=_maybe_empty_stream(XYZStream, r_pb.rt_acc),
            rt_gyro=_maybe_empty_stream(XYZStream, r_pb.rt_gyro),
            power_consumption=_maybe_empty_stream(PowerConsumptionStream,
                                                  r_pb.power_consumption),
            rt_oriented_acc=_maybe_empty_stream(XYZStream, r_pb.rt_oriented_acc),
            rt_oriented_gyro=_maybe_empty_stream(XYZStream, r_pb.rt_oriented_gyro),
            rt_device_orientation=_maybe_empty_stream(
                DeviceOrientationStream, r_pb.rt_device_orientation),
            fcw=_maybe_empty_stream(FCWStream, r_pb.fcw),
            risk_assessment=_maybe_empty_stream(
                RiskAssessmentStream, r_pb.risk_assessment),
            risk_rta=_maybe_empty_stream(RiskRtaStream, r_pb.risk_rta),
            maneuver=_maybe_empty_stream(ManeuverStream, r_pb.maneuver),
            temperature=_maybe_empty_stream(TemperatureStream, r_pb.temperature),
            braking_distance=_maybe_empty_stream(
                BrakingDistanceStream, r_pb.braking_distance),
            centerline_polygon=_maybe_empty_stream(
                CenterlinePolygonStream, r_pb.centerline_polygon),
            omnifusion_vehicle_dynamics=_maybe_empty_stream(OmniFusionVehicleDynamicsStream,
                                                            r_pb.omnifusion_vehicle_dynamics),
            omnifusion_location=_maybe_empty_stream(OmniFusionLocationStream,
                                                    r_pb.omnifusion_location),
            auto_calibration=_maybe_empty_stream(AutoCalibrationStream,
                                                 r_pb.auto_calibration),
            model_definition=[
                ModelDefinition.from_pb(def_pb) for def_pb
                in r_pb.model_definition
            ],
            metadata=RecordingMetadata.from_pb(r_pb))

    def to_utc_time(self) -> 'Recording':
        transformed_sensor_streams = {
            name: getattr(self, name).to_utc_time(
                self.metadata.utc_boot_time_ns,
                self.metadata.utc_boot_time_offset_ns)
            for name, field_type in self._field_types.items()
            if inspect.isclass(field_type) and issubclass(field_type, UtcTimeConvertible)
        }
        return self._replace(**transformed_sensor_streams)


ComFrameStream = streams.create_combined_stream_type(
    'ComFrameStream',
    FrameStream,
    [CombinedUtcTimeConvertible])
ComAQBitStream = streams.create_combined_stream_type(
    'ComAQBitStream',
    AQBitStream,
    [CombinedUtcTimeConvertible])
ComFCWStream = streams.create_combined_stream_type(
    'ComFCWStream',
    FCWStream,
    [CombinedUtcTimeConvertible])
ComXYZStream = streams.create_combined_stream_type(
    'ComXYZStream',
    XYZStream,
    [CombinedUtcTimeConvertible])
ComGPSStream = streams.create_combined_stream_type(
    'ComGPSStream',
    GPSStream,
    [CombinedUtcTimeConvertible])
ComSpeedStream = streams.create_combined_stream_type(
    'SpeedStream',
    SpeedStream,
    [CombinedUtcTimeConvertible])
ComEKFStream = streams.create_combined_stream_type(
    'ComEKFStream',
    EKFStream,
    [CombinedUtcTimeConvertible])
ComObdStream = streams.create_combined_stream_type(
    'ComObdStream',
    ObdStream,
    [CombinedUtcTimeConvertible])
ComDeviceOrientationStream = streams.create_combined_stream_type(
    'ComDeviceOrientationStream',
    DeviceOrientationStream,
    [CombinedUtcTimeConvertible])
ComLooseDeviceStream = streams.create_combined_stream_type(
    'ComLooseDeviceStream',
    LooseDeviceStream,
    [CombinedUtcTimeConvertible])
ComDistractionStream = streams.create_combined_stream_type(
    'ComDistractionStream',
    DistractionStream,
    [CombinedUtcTimeConvertible])
ComDrowsinessStream = streams.create_combined_stream_type(
    'ComDrowsinessStream',
    DrowsinessStream,
    [CombinedUtcTimeConvertible])
ComTailgatingStream = streams.create_combined_stream_type(
    'ComTailgatingStream',
    TailgatingStream,
    [CombinedUtcTimeConvertible])
ComAppliedOrientationStream = streams.create_combined_stream_type(
    'ComAppliedOrientationStream',
    AppliedOrientationStream,
    [CombinedUtcTimeConvertible])
ComBoundingBoxStream = streams.create_combined_stream_type(
    'ComBoundingBoxStream',
    BoundingBoxStream,
    [CombinedUtcTimeConvertible])
ComMCODBoundingBoxStream = streams.create_combined_stream_type(
    'ComMCODBoundingBoxStream',
    BoundingBoxStream,
    [CombinedUtcTimeConvertible])
ComImuStatisticsStream = streams.create_combined_stream_type(
    'ComImuStatisticsStream',
    ImuStatisticsStream,
    [CombinedUtcTimeConvertible])
ComPowerConsumptionStream = streams.create_combined_stream_type(
    'ComPowerConsumptionStream',
    PowerConsumptionStream,
    [CombinedUtcTimeConvertible])
ComRiskAssessmentStream = streams.create_combined_stream_type(
    'ComRiskAssessmentStream',
    RiskAssessmentStream,
    [CombinedUtcTimeConvertible])
ComManeuverStream = streams.create_combined_stream_type(
    'ComManeuverStream',
    ManeuverStream,
    [CombinedUtcTimeConvertible])
ComBrakingDistanceStream = streams.create_combined_stream_type(
    'ComBrakingDistanceStream',
    BrakingDistanceStream)
ComTemperatureStream = streams.create_combined_stream_type(
    'ComTemperatureStream',
    TemperatureStream,
    [CombinedUtcTimeConvertible])
ComAutoCalibrationStream = streams.create_combined_stream_type(
    'ComAutoCalibrationStream',
    AutoCalibrationStream)
ComOmniFusionVehicleDynamicsStream = streams.create_combined_stream_type(
    'ComOmniFusionVehicleDynamicsStream',
    OmniFusionVehicleDynamicsStream,
    [CombinedUtcTimeConvertible])
ComOmniFusionLocationStream = streams.create_combined_stream_type(
    'ComOmniFusionLocationStream',
    OmniFusionLocationStream,
    [CombinedUtcTimeConvertible])
ComCenterlinePolygonStream = streams.create_combined_stream_type(
    'ComCenterlinePolygonStream',
    CenterlinePolygonStream)


class CombinedRecording(metaclass=NamedTupleMetaEx):
    acc: ComXYZStream
    gyro: ComXYZStream
    grv: ComXYZStream
    mag: ComXYZStream
    lin: ComXYZStream
    gps: ComGPSStream
    speed: ComSpeedStream
    dist: ComFrameStream
    tail: ComFrameStream
    aqbit: ComAQBitStream
    ekf: ComEKFStream
    ekf_configs: List[EKFConfig]
    obd: ComObdStream
    device_orientation: ComDeviceOrientationStream
    dist_multilabel: ComDistractionStream
    drowsiness: ComDrowsinessStream
    drowsiness_configs: List[DrowsinessConfig]
    loose_device: ComLooseDeviceStream
    applied_orientation: ComAppliedOrientationStream
    oriented_acc: ComXYZStream
    oriented_gyro: ComXYZStream
    bounding_boxes_external: ComBoundingBoxStream
    mcod_bounding_boxes: ComMCODBoundingBoxStream
    tailgating: ComTailgatingStream
    imu_statistics: ComImuStatisticsStream
    rt_acc: ComXYZStream
    rt_gyro: ComXYZStream
    power_consumption: ComPowerConsumptionStream
    rt_oriented_acc: ComXYZStream
    rt_oriented_gyro: ComXYZStream
    rt_device_orientation: ComDeviceOrientationStream
    fcw: ComFCWStream
    risk_assessment: ComRiskAssessmentStream
    maneuver: ComManeuverStream
    temperature: ComTemperatureStream
    braking_distance: ComBrakingDistanceStream
    centerline_polygon: ComCenterlinePolygonStream
    model_definition: List[ModelDefinition]
    omnifusion_vehicle_dynamics: ComOmniFusionVehicleDynamicsStream
    omnifusion_location: ComOmniFusionLocationStream
    auto_calibration: ComAutoCalibrationStream
    metadatas: List[RecordingMetadata]

    def recordings_count(self) -> int:
        return len(self.metadatas)

    def to_utc_time(self) -> 'CombinedRecording':
        utc_boot_times = [
            m.utc_boot_time_ns for m in self.metadatas
        ]
        utc_boot_offsets = [
            m.utc_boot_time_offset_ns for m in self.metadatas
        ]
        transformed_sensor_streams = {
            name: getattr(self, name).to_utc_time(
                utc_boot_times,
                utc_boot_offsets)
            for name, field_type in self._field_types.items()
            if inspect.isclass(field_type) and issubclass(field_type,
                                                          CombinedUtcTimeConvertible)
        }
        return self._replace(**transformed_sensor_streams)

    @staticmethod
    def from_recordings(recordings: List[Recording]) -> 'CombinedRecording':
        if not recordings:
            raise ValueError('No recordings to combine')

        # making model mappings concistent after combining substreams
        # with potentially different orders of models
        model_ids = {}
        model_mappings = []
        model_definitions = []
        for rec in recordings:
            mapping = []
            for m_def in rec.model_definition:
                if m_def in model_ids:
                    ind = model_ids[m_def]
                else:
                    ind = len(model_ids)
                    model_ids[m_def] = ind
                    model_definitions.append(m_def)
                mapping.append(ind)

            model_mappings.append(np.array(mapping, dtype=np.int32))

        return CombinedRecording(
            acc=ComXYZStream.from_substreams(
                [rec.acc for rec in recordings]),
            gyro=ComXYZStream.from_substreams(
                [rec.gyro for rec in recordings]),
            grv=ComXYZStream.from_substreams(
                [rec.grv for rec in recordings]),
            mag=ComXYZStream.from_substreams(
                [rec.mag for rec in recordings]),
            lin=ComXYZStream.from_substreams(
                [rec.lin for rec in recordings]),
            gps=ComGPSStream.from_substreams(
                [rec.gps for rec in recordings]),
            speed=ComSpeedStream.from_substreams(
                [rec.speed for rec in recordings]),
            dist=ComFrameStream.from_substreams(
                [rec.dist for rec in recordings]),
            tail=ComFrameStream.from_substreams(
                [rec.tail for rec in recordings]),
            aqbit=ComAQBitStream.from_substreams(
                [rec.aqbit for rec in recordings]),
            ekf=ComEKFStream.from_substreams(
                [rec.ekf for rec in recordings]),
            ekf_configs=[rec.ekf_config for rec in recordings],
            obd=ComObdStream.from_substreams(
                [rec.obd for rec in recordings]),
            device_orientation=ComDeviceOrientationStream.from_substreams(
                [rec.device_orientation for rec in recordings]),
            dist_multilabel=ComDistractionStream.from_substreams(
                [rec.dist_multilabel for rec in recordings]),
            drowsiness=ComDrowsinessStream.from_substreams(
                [rec.drowsiness for rec in recordings]),
            drowsiness_configs=[rec.drowsiness_config for rec in recordings],
            loose_device=ComLooseDeviceStream.from_substreams(
                [rec.loose_device for rec in recordings]),
            applied_orientation=ComAppliedOrientationStream.from_substreams(
                [rec.applied_orientation for rec in recordings]),
            oriented_acc=ComXYZStream.from_substreams(
                [rec.oriented_acc for rec in recordings]),
            oriented_gyro=ComXYZStream.from_substreams(
                [rec.oriented_gyro for rec in recordings]),
            bounding_boxes_external=ComBoundingBoxStream.from_substreams(
                [
                    rec.bounding_boxes_external.remap_model_ids(mapping)
                    for rec, mapping in zip(recordings, model_mappings)
                ]),
            mcod_bounding_boxes=ComMCODBoundingBoxStream.from_substreams(
                [
                    rec.mcod_bounding_boxes.remap_model_ids(mapping)
                    for rec, mapping in zip(recordings, model_mappings)
                ]),
            tailgating=ComTailgatingStream.from_substreams(
                [rec.tailgating for rec in recordings]),
            imu_statistics=ComImuStatisticsStream.from_substreams(
                [rec.imu_statistics for rec in recordings]),
            rt_acc=ComXYZStream.from_substreams(
                [rec.rt_acc for rec in recordings]),
            rt_gyro=ComXYZStream.from_substreams(
                [rec.rt_gyro for rec in recordings]),
            power_consumption=ComPowerConsumptionStream.from_substreams(
                [rec.power_consumption for rec in recordings]),
            rt_oriented_acc=ComXYZStream.from_substreams(
                [rec.rt_oriented_acc for rec in recordings]),
            rt_oriented_gyro=ComXYZStream.from_substreams(
                [rec.rt_oriented_gyro for rec in recordings]),
            rt_device_orientation=ComDeviceOrientationStream.from_substreams(
                [rec.rt_device_orientation for rec in recordings]),
            fcw=ComFCWStream.from_substreams(
                [rec.fcw for rec in recordings]),
            risk_assessment=ComRiskAssessmentStream.from_substreams(
                [rec.risk_assessment for rec in recordings]),
            maneuver=ComManeuverStream.from_substreams(
                [rec.maneuver for rec in recordings]),
            temperature=ComTemperatureStream.from_substreams(
                [rec.temperature for rec in recordings]),
            braking_distance=ComBrakingDistanceStream.from_substreams(
                [rec.braking_distance for rec in recordings]),
            centerline_polygon=ComCenterlinePolygonStream.from_substreams(
                [rec.centerline_polygon for rec in recordings]),
            omnifusion_vehicle_dynamics=ComOmniFusionVehicleDynamicsStream.from_substreams(
                [rec.omnifusion_vehicle_dynamics for rec in recordings]),
            omnifusion_location=ComOmniFusionLocationStream.from_substreams(
                [rec.omnifusion_location for rec in recordings]),
            auto_calibration=ComAutoCalibrationStream.from_substreams(
                [rec.auto_calibration for rec in recordings]),
            model_definition=model_definitions,
            metadatas=[rec.metadata for rec in recordings])


class ImuStream(NamedTuple):
    acc: XYZStream
    gyro: XYZStream

    @staticmethod
    def from_recording(recording: CombinedRecording) -> "ImuStream":
        if recording.acc.stream._is_empty() or recording.gyro.stream._is_empty():
            raise ValueError("Streams are empty!")

        gyro = recording.gyro.stream
        version = recording.metadatas[0].version
        if (
                version is not None
                and len(version) > 0
                and float(".".join(version.split(".")[:2])) < 2.4
        ):
            # there is supposedly a bug in the old versions of sensor data.
            gyro = gyro._replace(x=gyro.x * 4.0, y=gyro.y * 4.0, z=gyro.z * 4.0)

        return ImuStream(acc=recording.acc.stream, gyro=gyro)
