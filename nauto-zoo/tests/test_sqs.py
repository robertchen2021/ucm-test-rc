import boto3
import json
from tempfile import NamedTemporaryFile
from typing import Tuple, Dict, List, Optional
from botocore.exceptions import ClientError


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

def test_visual_crashnet_sqs():
    crashnet_queue_url = 'https://sqs.us-east-1.amazonaws.com/375130608285/drt-universal-cm-crashnet-prod-us'
    crashnet_dlq_url = 'https://sqs.us-east-1.amazonaws.com/375130608285/drt-cloud-model-crashnet-tasks-dead-letter'

    boto_sqs = boto3.Session(profile_name=AWS_PROFILE).client('sqs', region_name='us-east-1')
    try:
        response = boto_sqs.receive_message(QueueUrl=crashnet_queue_url, WaitTimeSeconds=10)
        json_body = json.loads(response.get('Messages')[0].get('Body'))
    except:
        try:
            response = boto_sqs.receive_message(QueueUrl=crashnet_dlq_url, WaitTimeSeconds=10)
            json_body = json.loads(response.get('Messages')[0].get('Body'))
        except:
            assert False

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

    assert True
