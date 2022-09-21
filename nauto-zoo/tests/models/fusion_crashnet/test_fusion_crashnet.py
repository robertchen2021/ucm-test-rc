import boto3
import json
import json
import os
import logging
from tempfile import NamedTemporaryFile
from typing import Tuple, Dict, List, Optional
from botocore.exceptions import ClientError
from pathlib import Path
from unittest.mock import Mock, patch
import sys
# sys.path.append('/opt/project/nauto-zoo')
from nauto_zoo import ModelInput
from nauto_zoo.preprocess.video import VisualCrashnetPreprocessor
from nauto_zoo.preprocess.sensor import CrashnetSensorPreprocessor
from nauto_zoo.models.fusion_crashnet import CollisionDetector
import glob
import shutil

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

def test_fusion_crashnet_model(capsys):
    root_dir = './test_data/feabdb0f-5f7a-4b1f-8545-2f1398431e9b'
    sensor_files = glob.glob(os.path.join(root_dir, 'sensor') + '/*', recursive=True)
    video_in_files = glob.glob(os.path.join(root_dir, 'video-in') + '/*', recursive=True)

    with open(str(Path(root_dir + ".json").resolve()), 'r') as fh:
        json_body = json.loads(fh.read())

    drt_task = json_body.get('event')
    model_input = ModelInput(metadata=json_body)

    # Process sensor
    sensor_preprocessor = CrashnetSensorPreprocessor(window_size=4000, use_oriented=False)
    imu_input = sensor_preprocessor.preprocess_sensor_files(sensor_files, drt_task)
    model_input.set('sensor', imu_input)

    # Process video-in
    video_preprocessor = VisualCrashnetPreprocessor()
    for video_in_file in video_in_files:
        shutil.copy(video_in_file, '/tmp')
    frames = video_preprocessor.preprocess_video_files(video_in_files, drt_task)
    model_input.set('video-in', frames)
    print('embedding input shape:')
    print(frames.shape)

    model = CollisionDetector({'threshold': 0.35})
    model.set_logger(Mock())
    s3_client = boto3.resource('s3', region_name='us-east-1')
    model.set_s3(s3_client)
    model.bootstrap()
    print(model.fusion_crashnet_model.model_fusion_crashnet.summary())
    response = model.run(model_input)
    
    assert response.summary == 'FALSE'
    assert round(response.score, 3) == 0.213
    assert response.confidence == 39