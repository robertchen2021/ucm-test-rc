from typing import List, NamedTuple
import nautoflow as nf
import os
import json
import pandas as pd
from pathlib import Path
import datetime
from tempfile import TemporaryDirectory
import boto3
from .config import S3_ROOT, ALL_MODELS, HISTORY_START_DATE
from nauto_ucm_accountant.logger import logger


class Judgement(NamedTuple):
    ep_id: str
    device_id: str
    created_at: str
    uj_summary: str  # todo handle unsure-escalate user summary
    mj_summary: str
    mj_confidence: str  # todo analyze by confidence
    mj_score: float
    mj_version: str


class AccountantWriterFS(object):
    def __init__(self, root: Path, models: List[str] = None):
        self._fs_target = root
        if models is None:
            self._models = ALL_MODELS
        else:
            self._models = models
        if "QUBOLE_API_TOKEN" not in os.environ:
            raise RuntimeError("Qubole API token not available in env")
        self._executor = nf.ml.presto.PrestoQuboleExecutor(
            api_token=os.environ.get("QUBOLE_API_TOKEN"),
        )
        self._index = {"models": []}

    def write(self, last_days: int):
        end_date = datetime.date.today() - datetime.timedelta(days=1)
        if last_days == 0:
            start_date = HISTORY_START_DATE
        elif last_days > 0:
            start_date = end_date - datetime.timedelta(days=last_days)
        else:
            raise RuntimeError(f"Invalid number of days: {last_days}")
        return self._write(start_date=start_date, end_date=end_date)

    def _write(self, start_date: datetime.date, end_date: datetime.date):
        for model in self._models:
            self._write_model(model, start_date, end_date)
        self._write_index()

    def _write_model(self, model: str, start_date: datetime.date, end_date: datetime.date):
        model_data = self._get_model_data(model, start_date, end_date)
        if len(model_data) > 0:
            versions = model_data['mj_version'].unique()
            dates = model_data['created_at'].unique()
            self._index["models"].append({
                "name": model,
                "versions": list(versions),
            })
            for version in versions:
                for date in dates:
                    self._dump_chunk(model_data, model, version, date)

    def _get_model_data(self, model: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        # order of columns should match order of fields in response struct, ie Judgement
        select_query = "" \
                       "SELECT " \
                       "  mj.event_packager_id AS ep_id," \
                       "  mj.device_id AS device_id, " \
                       "  mj.created_at AS created_at, " \
                       "  uj.summary AS uj_summary, " \
                       "  mj.summary AS mj_summary, " \
                       "  mj.confidence AS mj_confidence, " \
                       "  ROUND(CAST(JSON_EXTRACT(mj.info, '$.model.score') AS REAL), 4) AS mj_score, " \
                       "  JSON_EXTRACT(mj.info, '$.model.version') AS mj_version " \
                       "FROM drt.judgements mj " \
                       "LEFT JOIN drt.judgements uj " \
                       "  ON uj.event_packager_id = mj.event_packager_id " \
                       "  AND (uj.summary = 'true' OR uj.summary = 'false') " \
                       f"WHERE (" \
                       f"    mj.created_at >= TIMESTAMP '{start_date.strftime('%Y-%m-%d')} 00:00' " \
                       f"    AND " \
                       f"    mj.created_at < TIMESTAMP '{end_date.strftime('%Y-%m-%d')} 23:59') " \
                       f"  AND (mj.source = '{model}') " \
                       f"  AND (uj.production_cloud_model IS NULL) "
        model_data = self._executor.get_results(select_query, Judgement._field_types.items())
        model_data['created_at'] = pd.to_datetime(model_data['created_at']).dt.date
        return model_data.replace("true", 1).replace("false", 0)

    def _write_index(self):
        with (self._fs_target / "index.json").open("wb") as fh:
            serialized_data = bytes(json.dumps(self._index), "ascii")
            fh.write(serialized_data)

    def _dump_chunk(self, model_data: pd.DataFrame, model: str, version: str, date: datetime.date):
        version_root = self._fs_target / model / version
        version_root.mkdir(parents=True, exist_ok=True)
        current_data = model_data[(model_data['mj_version'] == version) & (model_data['created_at'] == date)]
        if len(current_data) > 0:
            with (version_root / str(date)).open("wb") as fh:
                data = {
                    "true_labels": list(current_data['uj_summary']),
                    "predicted_labels": list(current_data['mj_summary']),
                    "predicted_scores": list(current_data['mj_score']),
                    "total_cases": len(current_data)  # todo real implementation
                }
                serialized_data = bytes(json.dumps(data), "ascii")
                fh.write(serialized_data)


class AccountantWriterS3(AccountantWriterFS):
    def __init__(self, s3_target: Path = S3_ROOT):
        # TODO : validate s3 path and permissions before any expensive operations
        self._s3_target = s3_target
        self._s3_client = boto3.client('s3')
        fs_target = TemporaryDirectory().name
        super().__init__(Path(fs_target))

    def write(self, **kwargs):
        super().write(**kwargs)
        self._upload_to_s3()

    def _upload_to_s3(self):
        # todo async
        # TODO check string maipulation for s3 path
        s3_target = str(self._s3_target).replace("s3:/", "")
        bucket_name, remote_dir_name = s3_target.split('/', 1)
        for root, dirs, files in os.walk(self._fs_target.absolute()):
            for file in files:
                source = Path(root) / file
                target = remote_dir_name + root.replace(str(self._fs_target), '') + "/" + file
                logger.info(f"Uploading from {source} to {bucket_name}/{target} ...")
                self._s3_client.upload_fileobj(
                    source.open("rb"),
                    bucket_name,
                    target
                )
