import asyncio
from pathlib import Path
from typing import Any, List, Tuple, Union, IO, Dict, Iterable

import aiobotocore as aioboto
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from botocore.response import StreamingBody


def path_to_bucket_and_key(path: Path) -> Tuple[str, str]:
    """Split a path into S3 bucket name and key """
    splits = path.parts
    if path.is_absolute():
        # remove / from beginning
        splits = splits[1:]
    if len(splits) < 2:
        raise ValueError(
            f"Path '{path}' cannot be split into bucket and key")
    bucket = splits[0]
    key = '/'.join(splits[1:])
    return bucket, key


class BotoS3Client:

    def __init__(self, max_pool_connections: int = 100) -> None:
        """Creates `BotoS3Client`

        Args:
            max_pool_connections: the maximum number of concurrently opened
                connections used by urllib under the hood. 
        """
        self._client = boto3.client(
            's3',
            config=Config(max_pool_connections=max_pool_connections))

    @property
    def client(self) -> Any:
        return self._client

    def get_file_stream(self, path: Path) -> StreamingBody:
        """Returns a stream with a data from the body of the message
        returned by boto3 client for a given file.
        """
        bucket, key = path_to_bucket_and_key(path)
        bytes_stream = self._client.get_object(
            Bucket=bucket, Key=key)['Body']
        return bytes_stream

    def read_file(self, path: Path) -> bytes:
        return self.get_file_stream(path).read()

    def read_file_list(self, paths: List[Path]) -> List[bytes]:
        streams = [
            self.get_file_stream(p) for p in paths
        ]
        return [s.read() for s in streams]

    def write_file(self, contents: Union[bytes, IO], path: Path) -> None:
        bucket, key = path_to_bucket_and_key(path)
        self.client.put_object(Body=contents, Bucket=bucket, Key=key)

    def upload_file(self, local_path: Path, target_path: Path) -> None:
        bucket, key = path_to_bucket_and_key(target_path)
        self._client.upload_file(str(local_path), Bucket=bucket, Key=key)

    def head_file(self, path: Path) -> Any:
        bucket, key = path_to_bucket_and_key(path)
        return self._client.head_object(Bucket=bucket, Key=key)

    def list_files(self, path_prefix: Path, one_level: bool = False) -> Iterable[Path]:
        if '/' in str(path_prefix):
            bucket, key = path_to_bucket_and_key(path_prefix)
            key = key + '/'
        else:
            bucket, key = str(path_prefix), ''
        token = None
        all_returned = False
        result_prefix = Path(bucket)
        delimiter = '/' if one_level else ''
        while not all_returned:
            if token:
                result = self._client.list_objects_v2(Bucket=bucket, Prefix=key, ContinuationToken=token,
                                                      Delimiter=delimiter)
            else:
                result = self._client.list_objects_v2(Bucket=bucket, Prefix=key,
                                                      Delimiter=delimiter)
            all_returned = not result['IsTruncated']
            token = result.get('NextContinuationToken')
            if one_level:
                for content in result.get('CommonPrefixes', []):
                    if content['Prefix'] != key:
                        yield result_prefix / content['Prefix']
            else:
                for content in result.get('Contents', []):
                    if content['Key'] != key:
                        yield result_prefix / content['Key']


class AsyncBotoS3Client:
    """An asynchronous version S3 client using `aiobotocore`
    library unerneath

    This client should always be created with "async with..."
    directive, e.g.

    ```
    async with AsyncBotoS3Client(loop=loop) as async_client:
            file_data = await async_client.read_file(file_path)
    ```
    """

    def __init__(self, loop: asyncio.AbstractEventLoop,
                 **client_kw: Dict[str, Any]) -> None:
        """Creates asynchronous S3 client
        Args:
            loop: asyncio event loop
            **kw: keyword arguments passed to aiobotocore's `create_client`
        """
        self._loop = loop
        self._client_kw = client_kw

    async def __aenter__(self) -> 'AsyncBotoS3Client':
        self._session = aioboto.get_session(loop=self._loop)
        self._client_cm = self._session.create_client('s3', **self._client_kw)
        self._client = await self._client_cm.__aenter__()
        return self

    async def __aexit__(self, *args) -> Any:
        return await self._client_cm.__aexit__(*args)

    @property
    def client(self) -> Any:
        return self._client

    async def get_file_stream(
            self, path: Path
    ) -> aioboto.response.StreamingBody:
        bucket, key = path_to_bucket_and_key(path)
        response = await self._client.get_object(
            Bucket=bucket, Key=key)
        return response['Body']

    async def read_file(self, path: Path) -> bytes:
        resp_stream = await self.get_file_stream(path)
        async with resp_stream as stream:
            return await stream.read()

    async def read_file_list(self, paths: List[Path]) -> List[bytes]:
        return await asyncio.gather(*[self.read_file(p) for p in paths])

    async def write_file(
            self, contents: Union[bytes, IO], path: Path) -> None:
        bucket, key = path_to_bucket_and_key(path)
        await self._client.put_object(Body=contents, Bucket=bucket, Key=key)

    async def upload_file(self, local_path: Path, target_path: Path) -> None:
        bucket, key = path_to_bucket_and_key(target_path)
        with open(local_path, 'rb') as f:
            await self.write_file(f, target_path)
