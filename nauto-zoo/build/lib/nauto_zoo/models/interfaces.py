from typing import Optional, Dict, Any, Union
from nauto_zoo import ModelInput, ModelResponse, ModelResponseCollection
from tempfile import NamedTemporaryFile
from logging import Logger
import abc


class Model(abc.ABC):
    def __init__(self, config: Optional[Dict[str, Any]]=None):
        self._logger = None
        self._s3_client = None
        if config is None:
            self._config = {}
        else:
            self._config = config

    @abc.abstractmethod
    def run(self, model_input: ModelInput) -> Union[ModelResponse, ModelResponseCollection]:
        pass

    def set_logger(self, logger: Logger):
        self._logger = logger

    def set_s3(self, s3_client):
        self._s3_client = s3_client

    def bootstrap(self):
        pass

    def _download_from_s3(self, bucket_name: str, file_name: str, destination_filename: Optional[str] = None) -> str:
        assert self._s3_client is not None
        assert self._logger is not None
        self._logger.info(f"Downloading file `{file_name}` from bucket `{bucket_name}`...")
        if destination_filename is None:
            destination_filename = NamedTemporaryFile(delete=False).name
        s3_bucket = self._s3_client.Bucket(bucket_name)
        with open(destination_filename, 'wb') as fd:
            s3_bucket.download_fileobj(file_name, fd)
        return destination_filename
