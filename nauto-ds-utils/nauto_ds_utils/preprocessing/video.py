import cv2
from typing import List, Any, Optional
from moviepy.editor import VideoFileClip
import abc
import numpy as np


class AbstractVideoPreprocessor(abc.ABC):
    def __init__(
        self,
        always_sample_first_frame: bool = False,
        grayscale: bool = False,
        max_frames: Optional[int] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ):
        self._always_sample_first_frame = always_sample_first_frame
        self._grayscale = grayscale
        self._max_frames = max_frames
        self._frame_width = frame_width
        self._frame_height = frame_height

    @abc.abstractmethod
    def preprocess_video_files(self, video_files: List[str]) -> Any:
        pass

    def extract_frames_from_file(self, video_file: str, 
        frame_share: float) -> List[np.array]:
        sampled_frames = []
        accumulator = 1 if self._always_sample_first_frame else 0
        video = cv2.VideoCapture(video_file)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            accumulator += frame_share
            if accumulator >= 1:
                if self._max_frames is not None:
                    if len(sampled_frames) > self._max_frames:
                        raise RuntimeError(
                            f'Aborted processing after sampling {self._max_frames} frames and not reaching the end')
                accumulator -= 1
                if self._frame_width is not None and self._frame_height is not None:
                    frame = cv2.resize(
                        frame, (self._frame_width, self._frame_height))
                if self._grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame)
        video.release()
        return sampled_frames


class VideoPreprocessorFrameShare(AbstractVideoPreprocessor):
    def __init__(self, frame_share: float, **kwargs):
        super().__init__(**kwargs)
        self._frame_share = frame_share

    def preprocess_video_files(self, video_files: List[str]) -> List[np.array]:
        frames = []
        for video_file in video_files:
            frames += self.extract_frames_from_file(
                video_file, frame_share=self._frame_share)
        return frames


class VideoPreprocessorFps(AbstractVideoPreprocessor):
    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self._fps = fps

    def preprocess_video_files(self, video_files: List[str]) -> List[np.array]:
        frames = []
        for video_file in video_files:
            video_fps = self.get_fps(video_file)
            frames += self.extract_frames_from_file(
                video_file=video_file, frame_share=self._fps / video_fps)
        return frames

    @staticmethod
    def get_fps(video_file_name: str) -> float:
        # cannot use OpenCV because it incorrectly identifies FPS of TS videos as 18000
        return VideoFileClip(video_file_name).fps


class VideoPreprocessorSaveFiles(AbstractVideoPreprocessor):
    def preprocess_video_files(self, video_files: List[str]) -> List[str]:
        return video_files
