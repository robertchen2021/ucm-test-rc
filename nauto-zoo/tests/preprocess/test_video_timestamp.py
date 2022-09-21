import boto3
import json
import json
import os
import logging
from tempfile import NamedTemporaryFile
from typing import Tuple, Dict, List, Optional
from botocore.exceptions import ClientError

import sys
# sys.path.append('/opt/project/nauto-zoo')
from nauto_zoo import ModelInput
from nauto_zoo.preprocess.video import VideoPreprocessorTS

def download_s3key(s3key: str, destination_filename: Optional[str] = None, s3 = None):
    s3key = s3key.replace("s3://", "")
    if destination_filename is None:
        destination_filename = NamedTemporaryFile(delete=False).name
    bucket, path = s3key.split('/', 1)
    s3_bucket = s3.Bucket(bucket)
    with open(destination_filename, 'wb') as fd:
        try:
            s3_bucket.download_fileobj(path, fd)
        except ClientError as e:
            print(f"Failed to get media from S3: `{s3key}`")
            raise e
        return destination_filename



def test_sqs_messages(capsys):
    profile = 'prod_us'
    drowsiness_queue_url = 'https://sqs.us-east-1.amazonaws.com/375130608285/drt-universal-cm-drowsiness-prod-us'

    boto_sqs = boto3.Session(profile_name=profile).client('sqs', region_name='us-east-1')
    response = boto_sqs.receive_message(QueueUrl=drowsiness_queue_url)
    json_body = json.loads(response.get('Messages')[0].get('Body'))
    drt_task = json_body.get('event')

    # The drt_task can be processed by media_manager
    media_types = ['video-in']
    media_video_in = [media for media in drt_task['media'] if media['type'] in media_types]

    media_urls = {}
    tempfiles = {}
    media_message_ids = {}

    boto_s3 = boto3.Session(profile_name=profile).resource('s3', region_name='us-east-1')
    for media_type in media_types:
        media_urls[media_type] = [media['s3key'] for media in media_video_in]
        media_message_ids[media_type] = [media['message_id'] for media in media_video_in]
        tempfiles[media_type] = [download_s3key(media_url, s3=boto_s3) for media_url in media_urls[media_type]]
        tempfiles[media_type] = [i for _, i in sorted(zip(media_message_ids[media_type], tempfiles[media_type]))]
        media_urls[media_type] = [i for _, i in sorted(zip(media_message_ids[media_type], media_urls[media_type]))]

    model_input = ModelInput(metadata=drt_task)
    video_preprocessor = VideoPreprocessorTS()
    for media_type, medias in tempfiles.items():
        if media_type in media_types:
            frames, frames_ts = video_preprocessor.preprocess_video_files(medias, drt_task)
            model_input.set(media_type, frames)
            model_input.set(media_type+'-ts', frames_ts)

    frames_extracted  = len(frames)
    timestamp_extracted = frames_ts.shape[0]

    assert frames_extracted == timestamp_extracted