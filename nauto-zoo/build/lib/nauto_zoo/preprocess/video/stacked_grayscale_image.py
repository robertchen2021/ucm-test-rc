from typing import List, Any, Optional, Tuple
import numpy as np
import cv2
from .video import VideoPreprocessorTS
from .video import AbstractVideoPreprocessor


class StackedGrayscalePreprocessor(AbstractVideoPreprocessor):
    DEFAULT_FS = 0
    OUTPUT_SHAPE = (320, 180)

    def __init__(self,
                 fs: int = DEFAULT_FS,
                 output_shape: Tuple[int, int] = OUTPUT_SHAPE,
                 **kwargs):
        super().__init__(**kwargs)
        self._fs = fs
        self._output_shape = output_shape

    def extract_frames_from_file(self, video_file):
        sampled_frames = []
        video = cv2.VideoCapture(video_file)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame)
        video.release()
        return sampled_frames

    def normalize_and_stack(self, frames, resize_shape, fs=0):
        gray_frames = [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in frames]
        resized_frames = [cv2.resize(item, resize_shape, interpolation=cv2.INTER_CUBIC) for item in gray_frames]
        normalized_frames = [(item - 127.5) / 127.5 for item in resized_frames]

        stacked_img_array = []
        for i in range(0, len(normalized_frames) - 2):
            start_idx = i
            end_idx = i + 3 + (2 * fs)
            if end_idx > len(normalized_frames):
                break
            stacked_img_array.append(
                np.transpose(np.array(normalized_frames[start_idx:end_idx:fs + 1]), axes=(1, 2, 0)))
        return stacked_img_array

    def preprocess_video_files(self, video_files: List[str], metadata=None) -> List[np.array]:
        frames = []
        for video_file in video_files:
            frames += self.extract_frames_from_file(video_file)

        stacked_img_array = self.normalize_and_stack(frames, self._output_shape, self._fs)

        return stacked_img_array