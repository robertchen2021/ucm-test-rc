from pathlib import Path
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

import logging
import gzip


from nauto_datasets.utils.boto import BotoS3Client
from nauto_datasets.core.sensors import CombinedRecording, Recording
from nauto_datasets.serialization.jsons import sensors as sensor_serialization
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2


def get_combined_recording(paths: List[Path],
                           max_pool_connection: int = 100
                           ) -> CombinedRecording:
    """
    Usage: get_combined_path(['nauto1-prod-upload/.../.../8214f2412wd1'])
    Grabs the specified S3 File Object and deserializes the sensor
    data.

    Input: Must be a list of Path Objects (from pathlib).
    """
    if None in paths:
        logging.error('No sensor in sensor paths')
        return None

    if isinstance(paths[0], str):
        paths = list(map(Path, paths))

    try:
        client = BotoS3Client(max_pool_connections=max_pool_connection)
        data_chunks = client.read_file_list(paths)
        recordings = []
        for msg_bytes in data_chunks:
            try:
                rec_pb = protobuf.parse_message_from_gzipped_bytes(
                    sensor_pb2.Recording, msg_bytes)
                recordings.append(Recording.from_pb(rec_pb))

            except:
                continue

        return CombinedRecording.from_recordings(recordings)

    except ClientError:
        gzip_files = [Path(p) for p in paths]
        recordings = []
        for file_path in gzip_files:
            rec_pb = protobuf.parse_message_from_gzipped_file(
                sensor_pb2.Recording, file_path)
            recordings.append(Recording.from_pb(rec_pb))
        return CombinedRecording.from_recordings(recordings)

    except Exception:
        logging.exception('Could not read sensor data')
        return None


def get_recording_json(paths: List[Path],
                       max_pool_connection: int = 100
                       ) -> Optional[Dict[str, Any]]:
    """Converts Sensor Data to JSON"""
    return sensor_serialization \
        .combined_recording_to_json(
            get_combined_recording(paths=paths,
                                   max_pool_connection=max_pool_connection)
        )
