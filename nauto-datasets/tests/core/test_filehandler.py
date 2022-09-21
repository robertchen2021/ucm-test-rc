import tempfile
import pytest
import boto3
import os

from pathlib import Path
from moto import mock_s3
from nauto_datasets.utils.boto import BotoS3Client, path_to_bucket_and_key
from nauto_datasets.core.serialization import (FileLocation, FileSource,
                                               FileHandler)

#------------------------------------LOCAL--------------------------------------


@pytest.fixture(scope='module')
def mock_dir() -> FileLocation:
    '''Create Mock Directory'''
    tmp_dir: str = tempfile.mkdtemp()
    return FileLocation(path=Path(tmp_dir), file_source=FileSource.LOCAL)


@pytest.fixture(scope='module')
def file_handler() -> FileHandler:
    '''Initiate File Handling'''
    return FileHandler()


def test_file_deletion(file_handler: FileHandler,
                       mock_dir: FileLocation) -> None:
    '''Delete the File'''
    file_name = 'test.txt'
    file_path: Path = mock_dir.path / file_name

    with open(file_path, 'w') as wf:
        wf.write('testing tests test')

    mock_file: FileLocation = FileLocation(
        path=file_path, file_source=FileSource.LOCAL)

    assert mock_file.path.exists()
    file_handler.delete(mock_file)
    assert mock_file.path.exists() is False


def test_dir_deletion(file_handler: FileHandler,
                      mock_dir: FileLocation) -> None:
    '''Delete the Directory'''
    sub_dir: Path = mock_dir.path / 'sub_dir'
    sub_dir.mkdir()
    mock_sub_dir: FileLocation = FileLocation(
        path=sub_dir, file_source=FileSource.LOCAL)

    assert mock_sub_dir.path.exists()
    file_handler.delete(mock_sub_dir)
    assert mock_sub_dir.path.exists() is False


#-------------------------------------S3----------------------------------------


BUCKET = 'test-bucket'
PREFIX = 'test_prefix'


@mock_s3
def test_delete_s3_files() -> None:
    # "Mock" the AWS credentials as they can't be mocked in Botocore currently
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")

    '''Test if the code paginates and deletes the files'''
    s3_client = BotoS3Client().client
    s3_client.create_bucket(Bucket=BUCKET)
    s3_path = Path(f'{BUCKET}/{PREFIX}')
    s3_loc: FileLocation = FileLocation(path=s3_path, file_source=FileSource.S3)
    file_handler: FileHandler = FileHandler(s3_client=s3_client)

    for num in range(1001):
        tmp_str = str(num)
        s3_client.put_object(
            Bucket=BUCKET, Key=f'{PREFIX}/{tmp_str}', Body=tmp_str)

    file_handler.delete(location=s3_loc)

    assert 'Contents' not in s3_client.list_objects_v2(Bucket=BUCKET).keys()
