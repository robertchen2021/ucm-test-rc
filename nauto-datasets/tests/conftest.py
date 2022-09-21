import asyncio
import shutil
import sys
import tempfile
import requests
import subprocess as sp
import signal
import time
from itertools import chain
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import aiobotocore
import pytest
from aiobotocore.config import AioConfig
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark_session() -> SparkSession:
    return SparkSession.builder.master('local[4]').getOrCreate()


@pytest.fixture
def temp_directory() -> Path:
    directory = Path(tempfile.mkdtemp(suffix='test_sharded_dataset'))
    yield directory

    if directory.exists():
        shutil.rmtree(directory)


def start_service(service_name: str, host: str, port: str) -> sp.Popen:
    args = [sys.executable, "-m", "moto.server", "-H", host,
            "-p", str(port), service_name]

    # If test fails stdout/stderr will be shown
    process = sp.Popen(args, stdin=sp.PIPE)
    url = "http://{host}:{port}".format(host=host, port=port)

    for i in range(0, 30):
        if process.poll() is not None:
            process.communicate()
            pytest.fail("service failed starting up: {}".format(service_name))
            break

        try:
            # we need to bypass the proxies due to monkeypatches
            requests.get(
                url, timeout=0.5, proxies={'http': None, 'https': None})
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    else:
        stop_process(process)  # pytest.fail doesn't call stop_process
        pytest.fail("Can not start service: {}".format(service_name))

    return process


def stop_process(process: sp.Popen) -> None:
    try:
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=20)
    except sp.TimeoutExpired:
        process.kill()
        outs, errors = process.communicate(timeout=20)
        exit_code = process.returncode
        msg = "Child process finished {} not in clean way: {} {}" \
            .format(exit_code, outs, errors)
        raise RuntimeError(msg)


@pytest.fixture(scope="session")
def aio_s3_server() -> str:
    host = "localhost"
    port = 5000
    url = "http://{host}:{port}".format(host=host, port=port)
    process = start_service('s3', host, port)

    try:
        yield url
    finally:
        stop_process(process)


@pytest.fixture
def aws_region() -> str:
    return 'us-east-1'


@pytest.fixture
def aio_session(event_loop: asyncio.BaseEventLoop) -> aiobotocore.AioSession:
    session = aiobotocore.get_session(loop=event_loop)
    return session


@pytest.fixture
def aio_config(aws_region: str) -> AioConfig:
    return AioConfig(region_name=aws_region, signature_version='s3',
                     read_timeout=5, connect_timeout=5)


def moto_config(endpoint_url: str) -> Dict[str, str]:
    AWS_ACCESS_KEY_ID = "xxx"
    AWS_SECRET_ACCESS_KEY = "xxx"
    kw = dict(endpoint_url=endpoint_url,
              aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
              aws_access_key_id=AWS_ACCESS_KEY_ID)
    return kw


@pytest.fixture
def aio_s3_client_kwargs(
        aws_region: str,
        aio_s3_server: str,
        aio_config: AioConfig
) -> Dict[str, Any]:
    return dict(
        config=aio_config,
        region_name=aws_region,
        **moto_config(aio_s3_server)
    )


@pytest.fixture
def aio_s3_client(
        request,
        aio_session: aiobotocore.AioSession,
        aio_s3_client_kwargs: Dict[str, Any],
        event_loop: asyncio.BaseEventLoop
) -> aiobotocore.client.AioBaseClient:

    async def f():
        return aio_session.create_client('s3', **aio_s3_client_kwargs)
    try:
        client = event_loop.run_until_complete(f())
        yield client
    finally:
        event_loop.run_until_complete(client.close())


def assert_status_code(response: Dict[str, str], status_code: Any) -> None:
    assert response['ResponseMetadata']['HTTPStatusCode'] == status_code


async def recursive_delete(
        aio_s3_client: aiobotocore.client.AioBaseClient,
        bucket_name: str
) -> Awaitable[None]:
    # Recursively deletes a bucket and all of its contents.
    async for n in aio_s3_client.get_paginator('list_object_versions').paginate(
            Bucket=bucket_name, Prefix=''):
        for obj in chain(
                n.get('Versions', []),
                n.get('DeleteMarkers', []),
                n.get('Contents', []),
                n.get('CommonPrefixes', [])):
            kwargs = dict(Bucket=bucket_name, Key=obj['Key'])
            if 'VersionId' in obj:
                kwargs['VersionId'] = obj['VersionId']
            resp = await aio_s3_client.delete_object(**kwargs)
            assert_status_code(resp, 204)

    resp = await aio_s3_client.delete_bucket(Bucket=bucket_name)
    assert_status_code(resp, 204)


@pytest.fixture
def create_bucket(
        request,
        aio_s3_client: aiobotocore.client.AioBaseClient,
        aws_region: str,
        event_loop: asyncio.BaseEventLoop
) -> Callable[[str, str], Awaitable[str]]:
    _bucket_name = None

    async def _f(bucket_name: Optional[str] = None) -> Awaitable[str]:
        nonlocal _bucket_name
        if bucket_name is None:
            bucket_name = f's3_bucket_{int(time.time())}'
        _bucket_name = bucket_name
        bucket_kwargs = {'Bucket': bucket_name}
        if aws_region != 'us-east-1':
            bucket_kwargs['CreateBucketConfiguration'] = {
                'LocationConstraint': aws_region,
            }
        response = await aio_s3_client.create_bucket(**bucket_kwargs)
        assert_status_code(response, 200)
        await aio_s3_client.put_bucket_versioning(
            Bucket=bucket_name, VersioningConfiguration={'Status': 'Enabled'})
        return bucket_name

    def fin():
        nonlocal _bucket_name
        if _bucket_name is not None:
            event_loop.run_until_complete(
                recursive_delete(aio_s3_client, _bucket_name))

    request.addfinalizer(fin)
    return _f
