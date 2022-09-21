# from universal_model.config.preprocess import VideoPreprocessingConfig, VIDEO_PREPROCESSING_FRAME_SHARE, VIDEO_PREPROCESSING_FPS
# from universal_model.preprocess.video import preprocess_video, get_fps
from pathlib import Path
import pytest
from nauto_zoo.preprocess.video import VideoPreprocessorFps, VideoPreprocessorFrameShare
import subprocess

def test_should_sample_all_frames():
    sut = VideoPreprocessorFrameShare(frame_share=1.)
    frames = sut.preprocess_video_files([get_test_video_path()])
    assert len(frames) == 75


def test_should_sample_first_frame():
    sut = VideoPreprocessorFrameShare(frame_share=0., always_sample_first_frame=True)
    frames = sut.preprocess_video_files([get_test_video_path()])
    assert len(frames) == 1


def test_should_sample_frame_share():
    sut = VideoPreprocessorFrameShare(frame_share=.3)
    frames = sut.preprocess_video_files([get_test_video_path()])
    assert len(frames) == 22


def test_should_sample_fps():
    sut = VideoPreprocessorFps(fps=6.)
    frames = sut.preprocess_video_files([get_test_video_path()])
    assert len(frames) == 28


def test_should_not_oversample_fps():
    sut = VideoPreprocessorFps(fps=1000.)
    frames = sut.preprocess_video_files([get_test_video_path()])
    assert len(frames) == 75


def test_should_grayscale():
    sut = VideoPreprocessorFrameShare(frame_share=0., always_sample_first_frame=True, grayscale=True)
    frame = sut.preprocess_video_files([get_test_video_path()])[0]
    assert len(frame.shape) == 2


def test_should_not_grayscale():
    sut = VideoPreprocessorFrameShare(frame_share=0., always_sample_first_frame=True, grayscale=False)
    frame = sut.preprocess_video_files([get_test_video_path()])[0]
    assert len(frame.shape) == 3
    assert frame.shape[2] == 3


def test_should_apply_max_frames():
    with pytest.raises(RuntimeError) as excinfo:
        sut = VideoPreprocessorFrameShare(frame_share=1., max_frames=1)
        sut.preprocess_video_files([get_test_video_path()])
    assert str(excinfo.value) == 'Aborted processing after sampling 1 frames and not reaching the end'


def test_should_resize():
    sut = VideoPreprocessorFrameShare(frame_share=0., always_sample_first_frame=True, frame_width=100, frame_height=50)
    frames = sut.preprocess_video_files([get_test_video_path()])
    frame = frames[0]
    assert frame.shape[0] == 50
    assert frame.shape[1] == 100


def test_should_not_resize():
    sut = VideoPreprocessorFrameShare(frame_share=0., always_sample_first_frame=True)
    frames = sut.preprocess_video_files([get_test_video_path()])
    frame = frames[0]
    assert frame.shape[0] == 480
    assert frame.shape[1] == 864


def test_should_fetch_fps():
    assert VideoPreprocessorFps.get_fps(get_test_video_path()) == 15.58


def test_ffprobe():
    print(
        subprocess.Popen(['ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[1].decode("utf-8"))
    assert True

def get_test_video_path() -> str:
    return str(Path('./test_data/video-out1.ts').resolve())

