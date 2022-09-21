from typing import List
from .structs import ModelBooks, ModelVersionBooks, ModelStatistics, TimeRange, PRData, PRCurve
from typing import Dict
import json
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from dateutil.parser.isoparser import DEFAULT_ISOPARSER as parser
import boto3
import os
from .config import S3_ROOT
from nauto_ucm_accountant.logger import logger


class AccountantReaderFS(object):
    # todo make it lazy
    def __init__(self, target: Path):
        self._target = target
        self._index = self._read_file(target / "index.json")
        self._models_cache = {}
        self._model_versions_cache = {}

    def get_models(self) -> List[ModelBooks]:
        return [self.get_model_books(model["name"]) for model in self._index["models"]]

    def get_model_books(self, model_name: str) -> ModelBooks:
        try:
            return self._models_cache[model_name]
        except KeyError:
            self._models_cache[model_name] = self._compute_model_books(model_name)
            return self._models_cache[model_name]

    def get_model_version_books(self, model_name: str, model_version: str) -> ModelVersionBooks:
        try:
            return self._model_versions_cache[model_name][model_version]
        except KeyError:
            if model_name not in self._model_versions_cache:
                self._model_versions_cache[model_name] = {}
            self._model_versions_cache[model_name][model_version] = \
                self._compute_model_version_books(model_name, model_version)
            return self._model_versions_cache[model_name][model_version]

    def _read_file(self, file_path: Path) -> Dict:
        with file_path.open("rb") as fh:
            contents = fh.read()
            serialized = contents.decode("ascii")
            return json.loads(serialized)

    def _compute_model_books(self, model_name: str) -> ModelBooks:
        model_versions = sorted([model["versions"] for model in self._index["models"] if model["name"] == model_name][0])
        return ModelBooks(
            model_name=model_name,
            versions=model_versions,
            statistics=ModelStatistics.aggregate([
                self.get_model_version_books(model_name, model_version).statistics
                for model_version in model_versions
            ])
        )

    def _compute_model_version_books(self, model_name: str, model_version: str) -> ModelVersionBooks:
        root = self._target / model_name / model_version
        dates = []
        true_labels = []
        predicted_labels = []
        predicted_scores = []
        total_cases = 0
        for daily_file in root.iterdir():
            if not daily_file.is_file():
                continue
            daily_file_content = self._read_file(daily_file)
            # todo this is inefficient
            dates = dates + [parser.parse_isodate(daily_file.name)] * len(daily_file_content["true_labels"])
            true_labels = true_labels + daily_file_content["true_labels"]
            predicted_labels = predicted_labels + daily_file_content["predicted_labels"]
            predicted_scores = predicted_scores + daily_file_content["predicted_scores"]
            total_cases = total_cases + daily_file_content["total_cases"]
        return ModelVersionBooks(
            model_name=model_name,
            version=model_version,
            statistics=ModelStatistics(
                time_range=TimeRange.from_dates(dates),
                pr_data=PRData.from_labels(true_labels, predicted_labels),
                pr_curve=PRCurve(true_labels=true_labels, predicted_scores=predicted_scores),
                judged_cases=len(predicted_labels),
                total_cases=total_cases
            )
        )


class AccountantReaderFSZipped(AccountantReaderFS):
    def __init__(self, target: Path = S3_ROOT):
        unzip_target = TemporaryDirectory().name
        ZipFile(target, "r").extractall(path=unzip_target)
        super().__init__(Path(unzip_target))


class AccountantReaderS3(AccountantReaderFS):
    def __init__(self, s3_target: Path = S3_ROOT):
        fs_target = TemporaryDirectory().name
        self._download_from_s3(s3_target, fs_target)
        super().__init__(Path(fs_target))

    def _download_from_s3(self, s3_target: Path, fs_target: str):
        # todo async
        s3_target = str(s3_target).replace("s3:/", "")
        bucket_name, remote_dir_name = s3_target.split('/', 1)
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=remote_dir_name):
            local_target = fs_target + obj.key.replace(remote_dir_name, "")
            if local_target == fs_target + "/":
                continue
            os.makedirs(os.path.dirname(local_target), exist_ok=True)
            logger.info(f"Downloading from {bucket_name}/{obj.key} to {local_target} ...")
            bucket.download_file(obj.key, local_target)
