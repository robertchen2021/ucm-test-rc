"""Package Events for dataset creation."""
import boto3
import ffmpeg
import json
import logging
import os
import pandas as pd
import numpy as np
import subprocess
import gzip
from ast import literal_eval
from io import BytesIO
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import plot

from nauto_datasets.core import sensors
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


###############################################################################
# Common functions
###############################################################################
def get_param(params, key):
    """Extract parameter from message_params."""
    p = json.loads(params)
    if key in p.keys():
        return p[key]


def get_media_type(message_type):
    """Determine if video or sensor from message_type."""
    if 'video' in message_type:
        return 'video'
    elif message_type == 'sensor':
        return 'sensor'


###############################################################################
# Load event list and event media messages
###############################################################################
def load_events(events_fname, s3_client, s3_bucket, s3_key, ts_fields=[]):
    """Get pre-selected list of events to process from S3."""
    obj = s3_client.get_object(Bucket=s3_bucket, Key='{}/{}'.format(s3_key, events_fname))
    events = pd.read_csv(BytesIO(obj['Body'].read()), index_col=0)

    for f in ts_fields:
        if f in events.columns:
            events[f] = pd.to_datetime(events[f])
    events.videos = events.videos.apply(literal_eval)
    events.sensors = events.sensors.apply(literal_eval)
    return events


def load_events_media_messages(videos_fname, sensors_fname, s3_client, s3_bucket, s3_key,
                               datetime_cols=[]):
    """Load event media messages from S3."""
    try:
        obj = s3_client.get_object(Bucket=s3_bucket,
                                   Key=os.path.join(s3_key, videos_fname))
        f = BytesIO(obj['Body'].read())
        videos = pd.read_csv(f, index_col=0)
        obj = s3_client.get_object(Bucket=s3_bucket,
                                   Key=os.path.join(s3_key, sensors_fname))
        f = BytesIO(obj['Body'].read())
        sensors = pd.read_csv(f, index_col=0)
    except:
        logger.error('S3 File not found')
        return None

    for col in datetime_cols:
        videos[col] = pd.to_datetime(videos[col])
        sensors[col] = pd.to_datetime(sensors[col])

    return (videos, sensors)


###############################################################################
# Load Media
###############################################################################
def download_event_media(event_media, s3_client, media_dir, overwrite=False):
    """Download event media files from S3 to local directory."""
    os.makedirs(media_dir, exist_ok=True)
    event_media['fname'] = None
    event_media['downloaded'] = None
    for i, row in event_media.iterrows():
        try:
            source_url = row.upload_data
            source_bucket = source_url.split('/')[0]
            source_key = source_url[len(source_bucket) + 1::]
            if 'snapshot' in row.message_type:
                message_extension = '.jpg'
            elif '-ts' in row.message_type:
                message_extension = '.ts'
            elif 'video' in row.message_type:
                message_extension = '.mp4'
            elif 'sensor' in row.message_type:
                message_extension = '.pb.gz'
            elif 'mjpeg' in row.message_type:
                message_extension = '.mjpeg'
            else:
                message_extension = ''
            fname = '{}-{}-{}{}'.format(row.device_id, row.message_id, row.message_type, message_extension)
            dest_fpath = os.path.join(media_dir, fname)
            if (overwrite is True) | (fname not in os.listdir(media_dir)):
                s3_client.download_file(source_bucket, source_key, dest_fpath)
            event_media.loc[i, 'fname'] = fname
            event_media.loc[i, 'downloaded'] = True
        except:
            event_media.loc[i, 'downloaded'] = False
    return event_media


def get_event_media_messages(event, video_messages, sensor_messages):
    """Get related event media messages from list of video and sensor messages."""
    event_media = (pd.concat((sensor_messages.loc[sensor_messages.message_id.isin(event['sensors'])].copy(),
                              video_messages.loc[video_messages.message_id.isin(event['videos'])].copy()),
                             sort=False)
                   .drop_duplicates(subset=['device_id', 'message_id', 'message_type']))
    event_media = event_media.loc[event_media.device_id == event['device_id']].reset_index(drop=True)
    event_media['event_message_id'] = event['message_id']
    event_media['event_message_ts'] = event['message_ts']
    event_media['event_message_type'] = event['event_message_type']
    return event_media


###############################################################################
# Check Media
###############################################################################
def get_event_media_check_df(event, event_media, check_downloaded=False, media_dir=None, dur_diff_tol=0.05):
    """Check each event media is available and has valid video duration."""
    if event_media is None:
        return None

    # Create dataframe of expected event media from event (filter non-message_id values)
    event_media_check_dict = {'video-in-ts': [v for v in event['videos'] if len(v) == 16],
                              'video-out-ts': [v for v in event['videos'] if len(v) == 16],
                              'sensor': [v for v in event['sensors'] if len(v) == 16]}
    event_media_check = []
    for k, v in event_media_check_dict.items():
        df = pd.DataFrame(v, columns=['message_id'])
        df['message_type'] = k
        df = df[['message_type', 'message_id']]
        event_media_check.append(df)
    event_media_check = pd.concat(event_media_check)
    if len(event_media_check) == 0:
        return None

    # Merge available event media from event_media dataframe
    event_media_check = event_media_check.merge(event_media
                                                .assign(available=True),
                                                how='left',
                                                on=['message_type', 'message_id'])
    event_media_check.available = event_media_check.available.fillna(False)

    # Compare sensor_duration and video_trans_duration to check if timestamps are valid
    event_media_check.loc[:, ['sensor_duration', 'video_trans_duration']] = event_media_check.loc[:, ['sensor_duration', 'video_trans_duration']].fillna(0).astype(float)
    event_media_check['valid_video_duration'] = ((event_media_check.sensor_duration - event_media_check.video_trans_duration).abs() <= dur_diff_tol)

    # Check expected media files were downloaded successfully
    if check_downloaded is True:
        flist = os.listdir(media_dir)
        event_media_check['downloaded'] = event_media_check.fname.apply(lambda x: x in flist)

    return event_media_check


def check_event_media_valid(event_media_check, check_downloaded=False):
    """Check that all event media is valid, or give failure reason."""
    # Check all media is available and downloaded
    if event_media_check is None:
        return "failed_media_not_found"
    if not event_media_check.available.all():
        return "failed_media_available"
    if check_downloaded is True:
        if not event_media_check.downloaded.all():
            return "failed_media_downloaded"

    # Check for presence of any video or sensor files
    if event_media_check.loc[event_media_check.media_type == 'video'].shape[0] == 0:
        return "failed_no_videos_found"
    if event_media_check.loc[event_media_check.media_type == 'sensor'].shape[0] == 0:
        return "failed_no_sensors_found"

    # Check videos for sensor_time and valid_video_duration
    if not event_media_check.loc[event_media_check.media_type == 'video'].is_sensor_time.all():
        return "failed_video_sensor_time"
    if not event_media_check.loc[event_media_check.media_type == 'video'].valid_video_duration.all():
        return "failed_video_duration_valid"

    else:
        return "success"


###############################################################################
# Process Video
###############################################################################
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
    frame_info['source'] = '{}-{}'.format(row.device_id, row.message_id)
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
    vframes_int = frames.loc[(frames.message_type == 'video-in-ts') &
                             (frames.media_type == 'video')].reset_index(drop=True)
    vframes_ext = frames.loc[(frames.message_type == 'video-out-ts') &
                             (frames.media_type == 'video')].reset_index(drop=True)
    aframes = frames.loc[(frames.message_type == 'video-in-ts') &
                         (frames.media_type == 'audio')].reset_index(drop=True)
    return (vframes_int, vframes_ext, aframes)


def get_event_frame_stats(event_media, frames):
    """Get basic statistics on frame counts and frame rate.

    TODO: Check for missing frames, inconsistencies between clips.
    """
    vframes_int, vframes_ext, aframes = split_frames_components(frames)
    vint_cts = vframes_int.source.value_counts()
    vext_cts = vframes_ext.source.value_counts()
    aint_cts = aframes.source.value_counts()
    frame_stats = (pd.DataFrame(vint_cts)
                   .join(vext_cts, rsuffix='_out')
                   .join(aint_cts, rsuffix='_audio')
                   .sort_index()
                   .rename(columns={'source': 'vid_in_frames',
                                    'source_out': 'vid_out_frames',
                                    'source_audio': 'audio_frames'}))

    frame_type_stats = frames.groupby(['message_type', 'media_type']).frame_ts_diff.describe(percentiles=[0.5])
    frame_type_stats['ave_frame_rate'] = frame_type_stats['count'].div(event_media.loc[event_media.media_type == 'video'].groupby('message_type').video_trans_duration.sum())
    return frame_stats, frame_type_stats


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

    if preserve_frames is False:
        # Find trims and offset to match int/ext
        min_frame_ts = np.max((vframes_int.sensor_ts.min(), vframes_ext.sensor_ts.min()))
        max_frame_ts = np.min((vframes_int.sensor_ts.max(), vframes_ext.sensor_ts.max()))

    else:
        # Set trims to include all available video frames
        min_frame_ts = np.min((vframes_int.sensor_ts.min(), vframes_ext.sensor_ts.min()))
        max_frame_ts = np.max((vframes_int.sensor_ts.max(), vframes_ext.sensor_ts.max()))

    vint_trim, vint_frames = get_trims(vframes_int, min_frame_ts, max_frame_ts)
    vext_trim, vext_frames = get_trims(vframes_ext, min_frame_ts, max_frame_ts)
    audio_trim, audio_frames = get_trims(aframes, min_frame_ts, max_frame_ts)

    if len(audio_frames) > 0:
        # Multiply audio trims by samples
        audio_nb_samples = aframes.drop_duplicates(['fpath'], keep='first').nb_samples.astype(int).values
        audio_trim[:, 1] *= audio_nb_samples
        audio_trim[:, 2] *= audio_nb_samples

    trims = (vint_trim, vext_trim, audio_trim)
    frame_timestamps = (vint_frames, vext_frames, audio_frames)
    return trims, frame_timestamps, (min_frame_ts, max_frame_ts)


def create_event_video(trims, output_dir, fname,
                       join_videos=True, join_type='horizontal-int-ext',
                       extract_audio=False, preserve_frames=False):
    """Create concatenated video for event."""
    vint_trim, vext_trim, audio_trim = trims
    try:
        if preserve_frames is False:
            # Get offset to have full matching between internal/external video
            pts_sign = '-'
            pts_offset_int, pts_offset_ext, pts_offset_audio = vext_trim[0][3], vint_trim[0][3], vext_trim[0][3]
        else:
            # Get offset to preserve all video frames
            pts_sign = '+'
            pts_offset_int, pts_offset_ext, pts_offset_audio = vint_trim[0][3], vext_trim[0][3], vint_trim[0][3]

        # Trim and concatenate videos
        vid_int = [ffmpeg.input(x[0])['v'].trim(start_frame=x[1], end_frame=x[2]).setpts('PTS-STARTPTS') for x in vint_trim]
        joined_int = ffmpeg.concat(*vid_int).setpts('PTS{}{}/TB'.format(pts_sign, pts_offset_int))
        vid_ext = [ffmpeg.input(x[0]).trim(start_frame=x[1], end_frame=x[2]).setpts('PTS-STARTPTS') for x in vext_trim]
        joined_ext = ffmpeg.concat(*vid_ext).setpts('PTS{}{}/TB'.format(pts_sign, pts_offset_ext))

        # Trim and concatenate audio if present
        if len(audio_trim) > 0:
            audio_int = [ffmpeg.input(x[0])['a'].filter('atrim', start_sample=x[1], end_sample=x[2]).filter('asetpts', 'PTS-STARTPTS') for x in audio_trim]
            joined_audio = ffmpeg.concat(*audio_int, v=0, a=1).filter('asetpts', 'PTS{}{}/TB'.format(pts_sign, pts_offset_audio))

        # Options for stitching - must be done in correct order to prevent showing blank frames
        vids = [joined_int, joined_ext]

        # Create video
        os.makedirs(output_dir, exist_ok=True)
        if (join_videos is False) | (join_videos == 'both'):
            # Output separate videos, method depends on if audio is present or not
            if len(audio_trim) > 0:
                args = (ffmpeg
                        .output(joined_int, joined_audio, os.path.join(output_dir, '{}-internal.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
            else:
                args = (joined_int
                        .output(os.path.join(output_dir, '{}-internal.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
            args = (joined_ext
                    .output(os.path.join(output_dir, '{}-external.mp4'.format(fname)))
                    .get_args())
            p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
            p.wait()
        if (join_videos == 'internal'):
            # Output internal videos, method depends on if audio is present or not
            if len(audio_trim) > 0:
                args = (ffmpeg
                        .output(joined_int, joined_audio, os.path.join(output_dir, '{}-internal.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
            else:
                args = (joined_int
                        .output(os.path.join(output_dir, '{}-internal.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
        if (join_videos == 'external'):
            # Output external videos
            args = (joined_ext
                    .output(os.path.join(output_dir, '{}-external.mp4'.format(fname)))
                    .get_args())
            p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
            p.wait()
        if (join_videos is True) | (join_videos == 'both'):
            if join_type == 'horizontal-int-ext':
                pad_args = ['pad', '2*iw', 'ih']
                if pts_offset_ext > pts_offset_int:
                    x, y = ['w', '0']
                else:
                    vids = vids[::-1]
                    pad_args += ['ow-iw', '0']
                    x, y = ['0', '0']

            elif join_type == 'horizontal-ext-int':
                pad_args = ['pad', '2*iw', 'ih']
                if pts_offset_ext < pts_offset_int:
                    vids = vids[::-1]
                    x, y = ['w', '0']
                else:
                    pad_args += ['ow-iw', '0']
                    x, y = ['0', '0']

            elif join_type == 'vertical-int-ext':
                pad_args = ['pad', 'iw', '2*ih']
                if pts_offset_ext > pts_offset_int:
                    x, y = ['0', 'h']
                else:
                    vids = vids[::-1]
                    pad_args += ['0', 'oh-ih']
                    x, y = ['0', '0']

            elif join_type == 'vertical-ext-int':
                pad_args = ['pad', 'iw', '2*ih']
                if pts_offset_ext < pts_offset_int:
                    vids = vids[::-1]
                    x, y = ['0', 'h']
                else:
                    pad_args += ['0', 'oh-ih']
                    x, y = ['0', '0']

            vids[0] = vids[0].filter(*pad_args)
            joined_video = (vids[0].overlay(vids[1], x=x, y=y))

            # Output video, method depends on if audio is present or not
            if len(audio_trim) > 0:
                args = (ffmpeg
                        .output(joined_video, joined_audio, os.path.join(output_dir, '{}.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
            else:
                args = (joined_video
                        .output(os.path.join(output_dir, '{}.mp4'.format(fname)))
                        .get_args())
                p = subprocess.Popen(['ffmpeg'] + args[:-1] + ['-vsync', '2'] + [args[-1]] + ['-y'])
                p.wait()
        if extract_audio is True:
            if len(audio_trim) > 0:
                args = (joined_audio
                        .output(os.path.join(output_dir, '{}-audio.aac'.format(fname)))
                        .get_args())
                # .run(overwrite_output=True))
                p = subprocess.Popen(['ffmpeg'] + args + ['-y'])
                p.wait()
        return 'success'
    except:
        return 'failed_create_video'


def process_event_video(event, event_media, status, media_fname,
                        media_dir, output_dir,
                        join_videos=True, join_type='horizontal-ext-int',
                        preserve_frames=False, extract_audio=False):
    """Create event video."""
    if event_media.loc[event_media.media_type == 'video'].shape[0] == 0:
        status['create_video'] = 'failed_create_video'
        frames, frame_timestamps, frame_ts_lims = None, None, None
    else:
        frames = get_event_frames(event_media, media_dir, event['utc_basetime'])
        trims, frame_timestamps, frame_ts_lims = get_component_trims(frames, preserve_frames=preserve_frames)
        status['create_video'] = create_event_video(trims, output_dir, media_fname,
                                                    join_videos=join_videos, join_type=join_type,
                                                    extract_audio=extract_audio, preserve_frames=preserve_frames)
    return status, frames, frame_timestamps, frame_ts_lims


###############################################################################
# Process Sensor Data
###############################################################################
def get_recording(files, convert_utc=True):
    """Combine sensor files into a single CombinedRecording object."""
    proto_msgs = [
        protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, file_path)
        for file_path in files
    ]

    com_recordings = sensors.CombinedRecording.from_recordings(
        [sensors.Recording.from_pb(r_pb)
            for r_pb in proto_msgs])
    if convert_utc is True:
        com_recordings = com_recordings.to_utc_time()
    return com_recordings


def orient_data(data, phi_xz):
    """Simple device orientation using only xz rotation.

    TODO: Update to orient on X and Y.
    """
    trans_mtx = np.array([[np.cos(phi_xz), 0, -np.sin(phi_xz)],
                         [0, 1, 0],
                         [np.sin(phi_xz), 0, np.cos(phi_xz)]])

    return np.matmul(trans_mtx, data)


def get_oriented_imu(acc_in, gyro_in, orientation, method='global'):
    """Use device angle to orient accelerometer and gyroscope.

    TODO: Currently only orients on X. Update to orient on X and Y.
    """
    acc = acc_in.copy()
    gyro = gyro_in.copy()
    # rotate acc and gyro based on converged orientation angles
    axes = ['x', 'y', 'z']
    if method == 'global':
        deg2rad = np.pi / 180
        # reorient acc/gyr, remove estimated gravity
        phi_xz = -orientation.pitch * deg2rad
        acc_rot = orient_data(np.matrix(acc.loc[:, axes].values).T, phi_xz)
        gravity = np.matrix([0, 0, -np.array(acc_rot.mean(axis=1))[2][0]]).astype(np.float64).transpose()
        acc.loc[:, axes] = (acc_rot + gravity).T
        gyro.loc[:, axes] = orient_data(np.matrix(gyro.loc[:, axes].values).T, phi_xz).T

        return acc, gyro
    else:
        return None, None


def get_device_orientation(com_rec):
    """Load device orientation data from combined recording."""
    try:
        orientation = com_rec.applied_orientation.stream._to_df()
        if len(orientation) == 0:
            orientation = com_rec.device_orientation.stream._to_df()
            if len(orientation) == 0:
                logger.warning('No orientation data')
                return None
            else:
                orientation = orientation.loc[orientation.converged].iloc[-1][['pitch', 'roll']]
        else:
            orientation = orientation.iloc[0][['pitch', 'roll']]
        return orientation
    except:
        return None


def load_sensor_fields(com_rec, fields=['acc', 'gyro', 'gps']):
    """Load necessary sensor fields."""
    sensor_data = {}
    for field in fields:
        if field not in com_rec._fields:
            continue
        sensor_data[field] = (getattr(com_rec, field).stream._to_df()
                              .drop('system_ms', axis=1)
                              .rename(columns={'sensor_ns': 'timestamp_utc_ns'})
                              .drop_duplicates(subset=['timestamp_utc_ns'])
                              .reset_index(drop=True))
    if ('applied_orientation' in fields) & (len(sensor_data['applied_orientation']) == 0):
        orientation = get_device_orientation(com_rec)
        if orientation is not None:
            sensor_data['applied_orientation'] = orientation

    # TODO: Create oriented IMU from orientation if not present in com_rec
    return sensor_data


def load_sensor_data(event_media, media_dir, fields=['acc', 'gyro', 'gps']):
    """Load combined sensor data."""
    flist = event_media.loc[event_media.media_type == 'sensor'].sort_values('message_ts').fname.values
    files = [Path(os.path.abspath(media_dir)) / f for f in flist]
    com_rec = get_recording(files)
    sensor_data = load_sensor_fields(com_rec, fields=fields)

    return sensor_data


def trim_sensors_to_video(sensor_data, video_ts_lims, ignore_fields=['applied_orientation']):
    """Trim all sensor data to match video time range."""
    sensor_start, sensor_end = video_ts_lims
    for k in sensor_data.keys():
        if (type(sensor_data[k]) == pd.DataFrame) & (k not in ignore_fields):
            sensor_data[k] = (sensor_data[k].loc[(sensor_data[k].timestamp_utc_ns >= sensor_start) &
                                                 (sensor_data[k].timestamp_utc_ns <= sensor_end)]
                              .reset_index(drop=True))
    return sensor_data


def find_nearest_value(x, idx, tolerance=100):
    """Locate nearest value to a given index within some tolerance."""
    try:
        i = idx.get_loc(x, method='pad', tolerance=tolerance)
        return idx[i]
    except:
        return x


def match_cnn_output_to_frames(cnn_df, frames_ts, timestamp_field='timestamp_utc_ns', tolerance=230):
    """Attempt to match frame scores to an actual frame timestamp."""
    vid_int_idx = pd.Index(frames_ts)
    cnn_df[timestamp_field] = cnn_df[timestamp_field].apply(find_nearest_value, idx=vid_int_idx, tolerance=tolerance)
    return cnn_df


def moving_average(data, n_samples=20):
    """
    Apply N-points moving average to the data.

    Note: for the first i points, where i < N, take the cumsum(data[:i])/i
    Params:
        data: 1D array, the data to be averaged
        N: int, the size of the window
    Returns:
        out: 1D array, after averaged
    """
    return np.hstack((np.cumsum(data[:n_samples - 1]) / np.arange(1, n_samples),
                      np.convolve(data, np.ones((n_samples,)) / n_samples, mode='valid')))


def get_mag(sens, keys):
    """Calculate the magnitude of all vectors in "keys"."""
    sens_array = []
    for k in keys:
        sens_array.append(np.array(sens[k]))
    sens_normed = np.linalg.norm(np.array(sens_array), axis=0)
    return sens_normed


def get_imu_filt(imu, maneuver_filt=300, severe_g_filt=50,
                 interpolate_method='linear', interpolate_limit=5):
    """Calculate magnitudes and run moving average filters for selected axes."""
    imu_filt = imu.copy()

    if interpolate_method is not None:
        imu_filt = (imu_filt
                    .set_index('timestamp_utc_ns')
                    .interpolate(method=interpolate_method, limit=interpolate_limit)
                    .reset_index())

    if maneuver_filt is not None:
        for a in ['x', 'y', 'z']:
            imu_filt['acc_{}'.format(a)] = moving_average(imu_filt['acc_{}'.format(a)].values,
                                                          int(maneuver_filt / 1000 * 200))
            imu_filt['gyro_{}'.format(a)] = moving_average(imu_filt['gyro_{}'.format(a)].values,
                                                           int(maneuver_filt / 1000 * 200))
    if severe_g_filt is not None:
        imu_filt['acc_xy'] = get_mag(imu_filt, ['acc_x', 'acc_y'])
        imu_filt['acc_xyz'] = get_mag(imu_filt, ['acc_x', 'acc_y', 'acc_z'])
        for a in ['xy', 'xyz']:
            imu_filt['acc_{}'.format(a)] = moving_average(imu_filt['acc_{}'.format(a)].values,
                                                          int(severe_g_filt / 1000 * 200))
    return imu_filt


def get_combined_imu_df(acc, gyro, timestamp_field='timestamp_utc_ns'):
    """Combine acc and gyro into single dataframe."""
    cols = [timestamp_field, 'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
    imu_df = (acc.join(gyro.set_index(timestamp_field),
                       on=timestamp_field, how='outer',
                       lsuffix='_acc', rsuffix='_gyro')
              [cols]
              .sort_values(timestamp_field))
    cols = [timestamp_field, 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    imu_df.columns = cols
    return imu_df


def plot_event(sensor_data, timestamp_field='timestamp_utc_ns', timestamp_units='ns',
               title='', height=None, output_dir=None, output_filename=None):
    """Generate Plotly HTML plot for sensor data."""
    chart_data = []
    ax_num = 0
    ax_list = []

    try:
        # Plot GPS speed
        if len(sensor_data['gps']) > 0:
            to_plot = (sensor_data['gps']
                       .set_index(pd.to_datetime(sensor_data['gps'][timestamp_field],
                                                 unit=timestamp_units)))
            ax_id = 'gps'
            ax_num += 1
            ax_list.append({'ax_num': ax_num, 'id': ax_id, 'name': 'GPS'})
            chart_data.append(go.Scatter(
                x=to_plot.index,
                y=to_plot['speed'],
                name='{}_{}'.format(ax_id, 'speed'),
                yaxis='y{}'.format(ax_num),
                xaxis='x',
                mode='lines'
            ))

        # Plot IMU
        imu_rot = get_combined_imu_df(sensor_data['oriented_acc'], sensor_data['oriented_gyro'])
        if len(imu_rot) > 0:
            imu_filt = get_imu_filt(imu_rot)
            axes = {'acc': ['x', 'y', 'z', 'xy', 'xyz'],
                    'gyro': ['x', 'y', 'z']
                    }

            if len(imu_filt) > 0:
                to_plot = imu_filt.set_index(pd.to_datetime(imu_filt[timestamp_field], unit=timestamp_units))
                # Plot Gyroscope
                ax_id = 'gyro'
                ax_num += 1
                ax_list.append({'ax_num': ax_num, 'id': ax_id, 'name': 'Gyroscope'})
                for a in axes[ax_id]:
                    to_plot_ax = to_plot['{}_{}'.format(ax_id, a)]
                    chart_data.append(go.Scatter(
                        x=to_plot_ax.index,
                        y=to_plot_ax,
                        name=to_plot_ax.name,
                        yaxis='y{}'.format(ax_num),
                        xaxis='x',
                        mode='lines'
                    ))

                # Plot Accelerometer
                ax_id = 'acc'
                ax_num += 1
                ax_list.append({'ax_num': ax_num, 'id': ax_id, 'name': 'Accelerometer'})
                for a in axes[ax_id]:
                    to_plot_ax = to_plot['{}_{}'.format(ax_id, a)]
                    chart_data.append(go.Scatter(
                        x=to_plot_ax.index,
                        y=to_plot_ax,
                        name=to_plot_ax.name,
                        yaxis='y{}'.format(ax_num),
                        xaxis='x',
                        mode='lines'
                    ))

        # Plot Distraction
        if len(sensor_data['dist_multilabel']) > 0:
            to_plot = (sensor_data['dist_multilabel']
                       .set_index(pd.to_datetime(sensor_data['dist_multilabel'][timestamp_field],
                                                 unit=timestamp_units)))
            ax_id = 'dist'
            ax_num += 1
            ax_list.append({'ax_num': ax_num, 'id': ax_id, 'name': 'Distraction'})
            score_cols = [c for c in to_plot.columns if 'score' in c]
            for a in score_cols:
                chart_data.append(go.Scatter(
                    x=to_plot.index,
                    y=to_plot['{}'.format(a)],
                    name='{}_{}'.format(ax_id, a),
                    yaxis='y{}'.format(ax_num),
                    xaxis='x',
                    mode='lines',
                    line=dict(shape='hv')
                ))

        # Plot Tailgating Distance Estimate
        if len(sensor_data['tailgating']) > 0:
            to_plot = (sensor_data['tailgating']
                       .set_index(pd.to_datetime(sensor_data['tailgating'][timestamp_field],
                                                 unit=timestamp_units)))
            # Fill non-detected front vehicle frames with NaNs
            to_plot.loc[to_plot.front_box_index == -1, 'distance_estimate'] = None
            ax_id = 'tail'
            ax_num += 1
            ax_list.append({'ax_num': ax_num, 'id': ax_id, 'name': 'Tailgating'})
            chart_data.append(go.Scatter(
                x=to_plot.index,
                y=to_plot.distance_estimate,
                name='tail_distance_estimate',
                yaxis='y{}'.format(ax_num),
                xaxis='x',
                mode='lines',
                line=dict(shape='hv')
            ))

        layout = go.Layout(
            showlegend=True,
            title=title,
            xaxis=dict(domain=[0, 1]),
            hovermode='closest',
            template='plotly_white'
        )
        ax_domains = np.linspace(0, 1, ax_num + 1)
        for i in np.arange(ax_num):
            layout['yaxis{}'.format(ax_list[i]['ax_num'])] = dict(
                domain=[ax_domains[i], ax_domains[i + 1]],
                title=ax_list[i]['name']
            )

        if height is not None:
            layout['height'] = height

        fig = go.Figure(data=chart_data, layout=layout)

        if output_filename is not None:
            plot(fig, filename=os.path.join(output_dir, '{}.html'.format(output_filename)), auto_open=False)
        return 'success', fig
    except:
        return 'failed_to_create_plot', None


def get_event_metadata(event, metadata_cols, created_by_email, rights):
    """Create metadata for event json."""
    metadata = {'created_at': str(pd.datetime.utcnow()),
                'created_by': created_by_email,
                'rights': rights}
    for col in metadata_cols:
        if col in event.keys():
            metadata[col] = event[col]
    if 'utc_basetime' in metadata.keys():
        if metadata['utc_basetime'] > 0:
            metadata['utc_basetime'] = int(metadata['utc_basetime'])
        else:
            metadata['utc_basetime'] = 0
    if 'vehicle_type' in metadata.keys():
        if type(metadata['vehicle_type']) != str:
            metadata['vehicle_type'] = ''
    if 'year' in metadata.keys():
        if metadata['year'] > 0:
            metadata['year'] = int(metadata['year'])
        else:
            metadata['year'] = ''
        metadata['year'] = str(metadata['year'])
    return metadata


def get_event_data_dict(sensor_data=None):
    """Convert sensor data and parameters into dict for storing as JSON."""
    data = {}
    for k, v in sensor_data.items():
        if type(v) == pd.DataFrame:
            data[k] = v.to_dict(orient='list')
        else:
            data[k] = v

    # Handle bounding_boxes_external as a unique case
    if 'bounding_boxes_external' in data.keys():
        for i, box in enumerate(data['bounding_boxes_external']['bounding_box']):
            box_dict = dict(box._asdict())
            for k, v in box_dict.items():
                box_dict[k] = v.tolist()
            data['bounding_boxes_external']['bounding_box'][i] = box_dict

    return data


def get_event_json(metadata, data):
    """Create event sensor data json."""
    return {'metadata': metadata, 'data': data}


def save_event_json(event_json, output_dir, fname):
    """Save sensor data json."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, '{}.json'.format(fname))
        with gzip.open(save_path + '.gz', 'w') as f_out:
            f_out.write(json.dumps(event_json).encode('utf-8'))
        return 'success'
    except:
        return 'failed_save_event_json'


def process_sensor_data(event, event_media, media_dir,
                        fields=['gps', 'acc', 'gyro', 'oriented_acc', 'oriented_gyro', 'applied_orientation',
                                'dist_multilabel', 'tailgating', 'bounding_boxes_external'],
                        trim_to_video=False, frame_ts_lims=None, frame_timestamps=None,
                        match_cnn_to_frames=True, cnn_field_params={'dist_multilabel': 230 * 1e6,
                                                                    'tailgating': 100 * 1e6,
                                                                    'bounding_boxes_external': 100 * 1e6},
                        message_params_field=None, trim_ignore_fields=['applied_orientation']):
    """Process sensor data from sensor files."""
    if event_media.loc[event_media.media_type == 'sensor'].shape[0] == 0:
        sensor_data = None
    else:
        sensor_data = load_sensor_data(event_media, media_dir, fields=fields)
        if (trim_to_video is True) & (frame_ts_lims is not None):
            sensor_data = trim_sensors_to_video(sensor_data, frame_ts_lims, ignore_fields=trim_ignore_fields)

        if frame_timestamps is not None:
            # Get video and audio frame timestamps
            video_frames_ts = {'internal_timestamp_utc_ns': list(map(int, frame_timestamps[0])),
                               'external_timestamp_utc_ns': list(map(int, frame_timestamps[1]))}
            audio_frames_ts = {'internal_timestamp_utc_ns': list(map(int, frame_timestamps[2]))}

            if match_cnn_to_frames is True:
                for field in cnn_field_params.keys():
                    sensor_data[field] = match_cnn_output_to_frames(sensor_data[field],
                                                                    video_frames_ts['internal_timestamp_utc_ns'],
                                                                    timestamp_field='timestamp_utc_ns',
                                                                    tolerance=cnn_field_params[field])
            sensor_data['video_frames'] = video_frames_ts
            sensor_data['audio_frames'] = audio_frames_ts

        if message_params_field is not None:
            sensor_data['event_params'] = json.loads(event[message_params_field])

    return sensor_data


###############################################################################
# Process Media
###############################################################################
def handle_event_media_failure_cases(event_media, failure_status, preserve_frames):
    """Fix issues with video and sensor files to allow processing."""
    failure_handling_status = None
    try:
        if failure_status == 'failed_video_sensor_time':
            failure_handling_status = 'success'
        elif failure_status == 'failed_video_duration_valid':
            # This bug should only affect a single inside or outside video in an event.
            # Until we have a better method, we naively repair these by assuming
            # sensor_end and video_trans_duration are correct and using this to fix sensor_start.
            rows_to_fix = event_media.loc[(event_media.media_type == 'video') & ~event_media.valid_video_duration].index
            event_media.loc[rows_to_fix, 'sensor_start'] = event_media.loc[rows_to_fix].sensor_end - (event_media.loc[rows_to_fix].video_trans_duration * 1e9).astype(int)
            event_media.loc[rows_to_fix, 'sensor_duration'] = (event_media.loc[rows_to_fix].sensor_end - event_media.loc[rows_to_fix].sensor_start).div(1e9)
            failure_handling_status = 'success'
        elif failure_status in ('failed_media_downloaded', 'failed_media_available'):
            preserve_frames = True
            # failed_videos = event_media.loc[not event_media.downloaded].message_id.unique()
            # event_media = event_media.loc[~event_media.message_id.isin(failed_videos)]
            failure_handling_status = 'success'
        else:
            failure_handling_status = 'failed'
    except:
        failure_handling_status = 'failed'
    return (failure_handling_status, preserve_frames, event_media)


def process_event_media(event, video_messages, sensor_messages, media_fname=None,
                        output_dir=None, s3_output=True, s3_output_dir=None, s3_bucket=None,
                        temp_dir='tmp/events', overwrite_media=False,
                        preserve_frames=False, handle_failure_cases=[],
                        join_videos=True, join_type='horizontal-ext-int', extract_audio=False,
                        create_sensor_json=False, plot_sensor_data=False,
                        message_params_field='event_message_params',
                        metadata_cols=['fleet_id', 'device_id', 'message_id', 'message_type', 'message_ts'],
                        created_by_email='data@nauto.com',
                        rights='Nauto Confidential. For research and development purposes only',
                        profile_name=None, dry_run=False, debug_output=False,):
    """Check all event media. Includes option to download media for processing."""
    status = {}

    if profile_name is not None:
        session = boto3.Session(profile_name=profile_name)
        s3_client = session.client('s3')
    else:
        s3_client = boto3.client('s3')

    event_media = get_event_media_messages(event, video_messages, sensor_messages)

    # make media and output dir specific to event
    if media_fname is None:
        media_fname = '{}-{}'.format(event['device_id'], event['message_id'])
    media_dir = os.path.join(temp_dir, 'event-media', media_fname)
    if (s3_output is True) & (output_dir is None):
        output_dir = os.path.join(temp_dir, 'event-output', media_fname)

    # For dry_run, check the media and return status
    if dry_run is False:
        event_media = download_event_media(event_media, s3_client, media_dir=media_dir, overwrite=overwrite_media)
    download_media = not dry_run

    event_media = get_event_media_check_df(event, event_media,
                                           check_downloaded=download_media, media_dir=media_dir)
    status['check_event_media'] = check_event_media_valid(event_media, check_downloaded=download_media)

    if (dry_run is True) | (status['check_event_media'] not in ['success'] + handle_failure_cases):
        frames, fig, event_json = None, None, None
    else:
        # Handle failure cases
        if (status['check_event_media'] in handle_failure_cases) | (handle_failure_cases == 'all'):
            resp = handle_event_media_failure_cases(event_media, status['check_event_media'], preserve_frames)
            status['handle_check_event_media_failure'], preserve_frames, event_media = resp

        # Process video
        video_result = process_event_video(event, event_media, status, media_fname,
                                           media_dir, output_dir,
                                           join_videos, join_type, preserve_frames, extract_audio)
        status, frames, frame_timestamps, frame_ts_lims = video_result

        if join_videos == 'external':
            frame_timestamps = ([], frame_timestamps[1], [])
        elif join_videos == 'internal':
            frame_timestamps = (frame_timestamps[0], [], frame_timestamps[2])

        # Process sensor data
        sensor_data = process_sensor_data(event, event_media, media_dir, trim_to_video=True,
                                          frame_ts_lims=frame_ts_lims, frame_timestamps=frame_timestamps,
                                          message_params_field=message_params_field)

        # Create sensor json and/or plot if selected/available
        event_json, fig = None, None
        if sensor_data is not None:
            # Create sensor data json
            if create_sensor_json is True:
                event_metadata = get_event_metadata(event, metadata_cols=metadata_cols, created_by_email=created_by_email, rights=rights)
                event_data = get_event_data_dict(sensor_data)
                event_json = get_event_json(event_metadata, event_data)
                status['create_sensor_json'] = save_event_json(event_json, output_dir, media_fname)

            # Create HTML sensor data plot
            if plot_sensor_data is True:
                status['create_sensor_plot'], fig = plot_event(sensor_data, title=media_fname, height=None,
                                                               output_dir=output_dir, output_filename=media_fname)
        else:
            if create_sensor_json is True:
                status['create_sensor_json'] = 'failed_no_sensors_found'
            if plot_sensor_data is True:
                status['create_sensor_plot'] = 'failed_no_sensors_found'

        # Move output files to s3
        if (s3_output is True) & (s3_output_dir is not None):
            try:
                for f in os.listdir(output_dir):
                    if media_fname in f:
                        s3_client.upload_file(os.path.join(output_dir, f), s3_bucket, os.path.join(s3_output_dir, f))
                status['upload_to_s3'] = 'success'
            except:
                status['upload_to_s3'] = 'upload_failed'

    if debug_output is False:
        return status
    else:
        return (status, event_media, frames, fig, event_json)
