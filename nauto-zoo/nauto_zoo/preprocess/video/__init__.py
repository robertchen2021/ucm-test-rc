from typing import List, Any, Optional
from moviepy.editor import VideoFileClip
import numpy as np

from .video import AbstractVideoPreprocessor, VideoPreprocessorTS
from .visual_crashnet import VisualCrashnetPreprocessor
from .stacked_grayscale_image import StackedGrayscalePreprocessor

class VideoPreprocessorFrameShare(AbstractVideoPreprocessor):
    def __init__(self, frame_share: float, **kwargs):
        super().__init__(**kwargs)
        self._frame_share = frame_share

    def preprocess_video_files(self, video_files: List[str], metadata=None) -> List[np.array]:
        frames = []
        for video_file in video_files:
            frames += self.extract_frames_from_file(video_file, frame_share=self._frame_share)
        return frames


class VideoPreprocessorFps(AbstractVideoPreprocessor):
    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self._fps = fps

    def preprocess_video_files(self, video_files: List[str], metadata=None) -> List[np.array]:
        frames = []
        for video_file in video_files:
            video_fps = self.get_fps(video_file)
            frames += self.extract_frames_from_file(video_file=video_file, frame_share=self._fps / video_fps)
        return frames

    @staticmethod
    def get_fps(video_file_name: str) -> float:
        # cannot use OpenCV because it incorrectly identifies FPS of TS videos as 18000
        return VideoFileClip(video_file_name).fps


class VideoPreprocessorSaveFiles(AbstractVideoPreprocessor):
    def preprocess_video_files(self, video_files: List[str]) -> List[str]:
        return video_files
