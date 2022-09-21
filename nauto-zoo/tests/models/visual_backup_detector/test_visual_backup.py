import sys
sys.path.append('/opt/project/nauto-zoo')
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
from nauto_zoo import ModelInput
from nauto_zoo.preprocess.video import StackedGrayscalePreprocessor
from nauto_zoo.models.visual_backup_detector import BackupDetector
import glob
import shutil

AWS_PROFILE = 'prod-us'

def test_visual_backup_model():
    with open(str(Path("./test_data/cacbd8413a58f497-1710a47a18204feb.json").resolve()), 'r') as fh:
        event_message = json.loads(fh.read())

    video_out_files = glob.glob(os.path.join('./test_data/cacbd8413a58f497-1710a47a18204feb', 'video-out') + '/*', recursive=True)

    config_provided_by_ucm = {'threshold': 0.78}

    model_input = ModelInput(metadata=event_message)
    video_preprocessor = StackedGrayscalePreprocessor()
    frames = video_preprocessor.preprocess_video_files(video_out_files, event_message)
    model_input.set('video-out', frames)


    model = BackupDetector(config_provided_by_ucm)
    model.set_logger(Mock())
    s3_client = boto3.resource('s3', region_name='us-east-1')
    model.set_s3(s3_client)
    model.bootstrap()
    response = model.run(model_input)

    assert response.summary == 'FALSE'
    assert round(response.score, 3) == 0.941
