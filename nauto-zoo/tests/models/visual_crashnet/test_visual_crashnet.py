import boto3
import json
from tempfile import NamedTemporaryFile
from typing import Tuple, Dict, List, Optional
from botocore.exceptions import ClientError
from pathlib import Path
from unittest.mock import Mock, patch
from nauto_zoo import ModelInput
from nauto_zoo.preprocess.video import VisualCrashnetPreprocessor
from nauto_zoo.models.visual_crashnet import CollisionDetector

AWS_PROFILE = 'prod_us'

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

def test_visual_crashnet_model():
    with open(str(Path("./test_data/feabdb0f-5f7a-4b1f-8545-2f1398431e9b.json").resolve()), 'r') as fh:
        json_body = json.loads(fh.read())

    drt_task = json_body.get('event')

    media_types = ['video-in']
    media_video_in = [media for media in drt_task['media'] if media['type'] in media_types]

    media_urls = {}
    tempfiles = {}
    media_message_ids = {}

    boto_s3 = boto3.Session(profile_name=AWS_PROFILE).resource('s3', region_name='us-east-1')
    for media_type in media_types:
        media_urls[media_type] = [media['s3key'] for media in media_video_in]
        media_message_ids[media_type] = [media['message_id'] for media in media_video_in]
        tempfiles[media_type] = [download_s3key(media_url, s3=boto_s3) for media_url in media_urls[media_type]]
        tempfiles[media_type] = [i for _, i in sorted(zip(media_message_ids[media_type], tempfiles[media_type]))]
        media_urls[media_type] = [i for _, i in sorted(zip(media_message_ids[media_type], media_urls[media_type]))]

    model_input = ModelInput(metadata=json_body)
    video_preprocessor = VisualCrashnetPreprocessor(half_length=4, fps=0.5)
    for media_type, medias in tempfiles.items():
        if media_type in media_types:
            frames = video_preprocessor.preprocess_video_files(medias, drt_task)
            model_input.set(media_type, frames)

    model = CollisionDetector({'threshold': 0.5})
    model.set_logger(Mock())
    s3_client = boto3.resource('s3', region_name='us-east-1')
    model.set_s3(s3_client)
    model.bootstrap()
    response = model.run(model_input)

    assert response.summary == 'FALSE'
    assert round(response.score, 3) == 0.002
    assert response.confidence == 99