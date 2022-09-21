from .interfaces import AbstractSensorPreprocessor
from typing import List, Dict
from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2
from nauto_zoo import MalformedSensorGZipError
from nauto_datasets.imu_utils import convert_oriented_to_raw, convert_raw_to_oriented

class SensorPreprocessorCombined(AbstractSensorPreprocessor):
    def preprocess_sensor_files(self, sensor_files: List[str], metadata: Dict = None) -> CombinedRecording:
        try:
            sensor_data = [open(file_name, "rb").read() for file_name in sensor_files]
            com_rec = CombinedRecording.from_recordings([
                Recording.from_pb(protobuf.parse_message_from_gzipped_bytes(sensor_pb2.Recording, recording_bytes))
                for recording_bytes in sensor_data
            ])

            if getattr(com_rec, 'rt_acc').stream._is_empty() and getattr(com_rec, 'rt_oriented_acc').stream._is_empty():
                if not getattr(com_rec, 'acc').stream._is_empty():
                    return com_rec
                else:
                    raise ValueError("Both rt and rt_oriented streams are empty!")
            elif getattr(com_rec, 'rt_acc').stream._is_empty():
                computed_raw_acc, computed_raw_gyro = convert_oriented_to_raw(com_rec)
                return com_rec._replace(**{'rt_acc': computed_raw_acc, 'rt_gyro': computed_raw_gyro})
            elif getattr(com_rec, 'rt_oriented_acc').stream._is_empty():
                computed_oriented_acc, computed_oriented_gyro = convert_raw_to_oriented(com_rec)
                return com_rec._replace(**{'rt_oriented_acc': computed_oriented_acc, 'rt_oriented_gyro': computed_oriented_gyro})
            else:
                return com_rec

        except EOFError:
            raise MalformedSensorGZipError()

        except OSError:
            raise MalformedSensorGZipError()
