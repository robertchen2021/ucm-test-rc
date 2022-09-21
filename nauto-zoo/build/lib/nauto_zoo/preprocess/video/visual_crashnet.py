from typing import List, Any, Optional
import numpy as np
import cv2
from .video import AbstractVideoPreprocessor
from .video import VideoPreprocessorTS


def crop_with_ar(img, ar=768/720, crop_h=720, crop_w=768):
    h = img.shape[0]
    w = img.shape[1]

    cropped_w = int(h * ar)
    cropped_img = img[:,-cropped_w:,:]

    if cropped_img.shape[0] != crop_h or cropped_img.shape[1] != crop_w:
        cropped_img = cv2.resize(cropped_img, dsize=(crop_w, crop_h))

    return cropped_img

class VisualCrashnetPreprocessor(VideoPreprocessorTS):
    DEFAULT_HALF_LENGTH = 5
    DEFAULT_FPS = 5
    def __init__(self,
                 half_length: float = DEFAULT_HALF_LENGTH,
                 fps: float = DEFAULT_FPS,
                 **kwargs):
        super().__init__(**kwargs)
        self._frame_intervals_s_from_peak = np.arange(-half_length, half_length + (1 / fps), (1 / fps)).tolist()

    def preprocess_video_files(self, video_files: List[str], metadata=None, crop_w_ar=True) -> List[np.array]:
        frames, frames_ts = super().preprocess_video_files(video_files, metadata)

        if metadata["type"] == "crashnet":
            peak_sensor_ns = int(metadata['params']['crashnet_data']['computed_peak_sensor_ns'])
        elif metadata["type"] == "severe-g-event":
            peak_sensor_ns = int(metadata['params']['maneuver_data']['peak_time_ns'])
        else:
            raise ValueError("Event type not supported for this pre-processor!")

        utc_boot_time = int(metadata['params']['utc_boot_time_ns'])
        boot_offset = int(metadata['params']['utc_boot_time_offset_ns'])
        frame_sensor_ns = np.array([frames_ts]) - utc_boot_time
        frame_indices = np.argmin(np.abs(np.tile((frame_sensor_ns.T-peak_sensor_ns)/1e9, len(self._frame_intervals_s_from_peak)) - np.array([self._frame_intervals_s_from_peak])), axis=0)
        if crop_w_ar:
            return np.array([crop_with_ar(frames[i]) for i in frame_indices])
        else:
            return np.array([frames[i] for i in frame_indices])