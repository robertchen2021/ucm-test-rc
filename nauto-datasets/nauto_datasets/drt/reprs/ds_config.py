from enum import Enum
from typing import NamedTuple, List, Dict


class JoinVideos(Enum):
    BOTH = 'both'  # to output joined and separate
    TRUE = 'true'  # to join interior/exterior videos
    FALSE = 'false'  # to output interior/exterior separately


class JoinVideosType(Enum):
    HORIZONTAL_EXT_INT = 'horizontal-ext-int'
    HORIZONTAL_INT_EXT = 'horizontal-int-ext'
    VERTICAL_EXT_INT = 'vertical-ext-int'
    VERTICAL_INT_EXT = 'vertical-int-ext'


class FailureCases(Enum):
    # Handle and attempt to fix specific failure cases.
    #   'failed_video_duration_valid': sensor_start time is incorrect from video cutoff bug,
    #      so reconstruct start_time using sensor_end - video_trans_duration (output may not be well synced)
    #   'failed_video_sensor_time': `is_sensor_time=0` in message_params, meaning firmware callback is not used
    #      to assign sensor_time (output may not be well synced)
    #   'failed_media_downloaded': Some videos or sensor files did not download. Attempt to stitch with gaps.

    VIDEO_SENSOR_TIME = 'failed_video_sensor_time'
    VIDEO_DURATION_VALID = 'failed_video_duration_valid'
    MEDIA_DOWNLOADED = 'failed_media_downloaded'


class RequestConfig(NamedTuple):
    prepare_new_request: bool
    request_description: str
    requested_by: str
    prepared_by: str
    dataset_name: str
    rights: str = 'Nauto Confidential. For research and development purposes only'
    join_videos: JoinVideos = JoinVideos.BOTH
    join_videos_type: JoinVideosType = JoinVideosType.HORIZONTAL_EXT_INT
    extract_audio: bool = False
    create_sensor_json: bool = True
    plot_sensor_data: bool = True
    dry_run: bool = False
    metadata_cols: List[str] = ['fleet_id', 'device_id', 'message_id', 'message_type', 'message_ts',
                                'vehicle_id', 'vin', 'make', 'model', 'year', 'vehicle_type', 'utc_basetime']
    handle_failure_cases: List[FailureCases] = [FailureCases.VIDEO_SENSOR_TIME, FailureCases.VIDEO_DURATION_VALID]

    def to_dict(self) -> Dict:
        dict_repr = self._asdict()
        dict_repr['join_videos'] = self.join_videos.value
        dict_repr['join_videos_type'] = self.join_videos_type.value
        dict_repr['handle_failure_cases'] = [h.value for h in self.handle_failure_cases]

        return dict_repr
