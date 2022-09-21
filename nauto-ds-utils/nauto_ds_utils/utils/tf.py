from tensorflow.keras.models import load_model as tf_load
from typing import Union, List, Optional, Any
from io import BytesIO
from pathlib import Path
import boto3

from nauto_datasets.core.serialization import (FileHandler, FileLocation,
                                               FileSource)
from nauto_datasets.utils.boto import path_to_bucket_and_key


def load_model(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path.replace('s3://', ''))

    if path.parts[0].startswith('s3:'):
        path = Path('/'.join(path.parts[1:]))

    if path.parts[0].startswith('nauto-'):

        bucket, key = path_to_bucket_and_key(path)
        print(bucket, key)
        with BytesIO() as f:
            boto3.client('s3').download_fileobj(
                Bucket=bucket, Key=key, Fileobj=f)
            f.seek(0)
            model = tf_load(f)
    else:
        model = tf_load(path)
    return model
