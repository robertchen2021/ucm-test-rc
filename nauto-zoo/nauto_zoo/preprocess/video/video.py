import cv2
from typing import List, Any, Optional
import os
import pandas as pd
import abc
import numpy as np
import subprocess
import json

def run_probe(filename, cmd='ffprobe'):
    """Run ffprobe on the specified file and return a JSON representation of the output.

    Raises:
        The stderr output can be retrieved by accessing the
        ``stderr`` property of the exception.
    """
    args = [cmd, '-show_format', '-show_streams', '-show_frames', '-of', 'json', filename]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return json.loads(out.decode('utf-8'))

def get_frames(row, frames, directory, event_utc_basetime=0):
    """Get frame info for a video and augment with message data.

    TODO: Only keep probe columns used later.
    """
    fpath = os.path.join(directory, row.fname)
    probe = run_probe(str(fpath))
    frame_info = [s for s in probe['frames']]
    frame_info = pd.DataFrame(frame_info)
    frame_info['frame_num'] = frame_info.assign(n=1).sort_values('pkt_pts').groupby('media_type').n.cumsum() - 1
    frame_info['sensor_ts'] = (frame_info.pkt_pts_time.apply(lambda x: int(float(x) * 1e9)) +
                               row.sensor_start + event_utc_basetime)
    # frame_info['source'] = '{}-{}'.format(row.device_id, row.message_id)
    frame_info['message_type'] = row.message_type
    frame_info['fpath'] = fpath
    frames.append(frame_info)


def get_event_frames(event_media, directory, utc_basetime):
    """Get combined frame info for all videos in event."""
    frames = []
    (event_media
     .loc[event_media.media_type == 'video']
     .apply(get_frames, args=(frames, directory, utc_basetime), axis=1))
    frames = (pd.concat(frames, sort=False)
              .sort_values(['message_type', 'media_type', 'sensor_ts'])
              .reset_index(drop=True))
    frames['frame_ts_diff'] = frames.groupby(['message_type', 'media_type']).sensor_ts.diff().div(1e6)
    return frames


def split_frames_components(frames):
    """Split frames into internal,  external, audio."""
    vframes_int = frames.loc[((frames.message_type == 'video-in-ts') | (frames.message_type == 'video-in')) &
                             (frames.media_type == 'video')].reset_index(drop=True)
    vframes_ext = frames.loc[(frames.message_type == 'video-out-ts') &
                             (frames.media_type == 'video')].reset_index(drop=True)
    aframes = frames.loc[(frames.message_type == 'video-in-ts') &
                         (frames.media_type == 'audio')].reset_index(drop=True)
    return (vframes_int, vframes_ext, aframes)


def get_trims(frames, min_frame_ts, max_frame_ts):
    """Get combined frame info for all videos in event."""
    f_df = (frames.loc[(frames.sensor_ts >= min_frame_ts) &
                       (frames.sensor_ts <= max_frame_ts)])
    trim_df = (f_df
               .assign(f_start_offset=(f_df.sensor_ts - min_frame_ts) / 1e9)
               .assign(f_start=f_df.frame_num)
               .assign(f_end=f_df.frame_num + 1)
               .groupby('fpath')
               .agg({'f_start': 'first', 'f_end': 'last', 'f_start_offset': 'first'})
               .reset_index()
               .values)
    frame_ts_list = f_df.sensor_ts.values
    return (trim_df, frame_ts_list)

def get_component_trims(frames, preserve_frames=False):
    """Get information to trim each media file for syncing."""
    vframes_int, vframes_ext, aframes = split_frames_components(frames)

    min_frame_ts = vframes_int.sensor_ts.min()
    max_frame_ts = vframes_int.sensor_ts.max()
    vint_trim, vint_frames = get_trims(vframes_int, min_frame_ts, max_frame_ts)

    trims = (vint_trim, None, None)
    frame_timestamps = (vint_frames, None, None)
    return trims, frame_timestamps, (min_frame_ts, max_frame_ts)

# def get_component_trims(frames, preserve_frames=False):
#     """Get information to trim each media file for syncing."""
#     vframes_int, vframes_ext, aframes = split_frames_components(frames)
#
#     if preserve_frames is False:
#         # Find trims and offset to match int/ext
#         min_frame_ts = np.max((vframes_int.sensor_ts.min(), vframes_ext.sensor_ts.min()))
#         max_frame_ts = np.min((vframes_int.sensor_ts.max(), vframes_ext.sensor_ts.max()))
#
#     else:
#         # Set trims to include all available video frames
#         min_frame_ts = np.min((vframes_int.sensor_ts.min(), vframes_ext.sensor_ts.min()))
#         max_frame_ts = np.max((vframes_int.sensor_ts.max(), vframes_ext.sensor_ts.max()))
#
#     vint_trim, vint_frames = get_trims(vframes_int, min_frame_ts, max_frame_ts)
#     vext_trim, vext_frames = get_trims(vframes_ext, min_frame_ts, max_frame_ts)
#     audio_trim, audio_frames = get_trims(aframes, min_frame_ts, max_frame_ts)
#
#     # if len(audio_frames) > 0:
#     #     # Multiply audio trims by samples
#     #     audio_nb_samples = aframes.drop_duplicates(['fpath'], keep='first').nb_samples.astype(int).values
#     #     audio_trim[:, 1] *= audio_nb_samples
#     #     audio_trim[:, 2] *= audio_nb_samples
#
#     trims = (vint_trim, vext_trim, audio_trim)
#     frame_timestamps = (vint_frames, vext_frames, audio_frames)
#     return trims, frame_timestamps, (min_frame_ts, max_frame_ts)

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

    def extract_frames_from_file(self, video_file: str, frame_share: float) -> List[np.array]:
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
                    frame = cv2.resize(frame, (self._frame_width, self._frame_height))
                if self._grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame)
        video.release()
        return sampled_frames

class VideoPreprocessorTS(AbstractVideoPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract_frames_from_file_with_ts(self, video_file, metadata):
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

    def preprocess_video_files(self, video_files: List[str], metadata=None) -> List[np.array]:
        event_media_list = []
        for item in metadata['media']:
            media_params = item['params']
            item.pop('params')
            media_dict = item
            media_dict.update(media_params)
            event_media_list.append(media_dict)

        event_media = pd.DataFrame(event_media_list)
        event_media.rename(columns={"type": "message_type"}, inplace=True)
        event_media = event_media[(event_media.message_type == 'video-in')]
        event_media.sort_values('message_id', inplace=True)
        event_media['fname'] = [file.split('/')[-1] for file in video_files]
        event_media['media_type'] = ['video'] * event_media.shape[0]
        event_media['sensor_start'] = event_media['sensor_start'].astype(np.int64)
        event_media['sensor_end'] = event_media['sensor_end'].astype(np.int64)
        frames_info = get_event_frames(event_media, '/tmp', metadata['params']['utc_boot_time_ns'])
        trims, frame_timestamps, frame_ts_lims = get_component_trims(frames_info, preserve_frames=True)
        frames = []
        for video_file in video_files:
            frames += self.extract_frames_from_file_with_ts(video_file, metadata=metadata)
        return frames, frame_timestamps[0]