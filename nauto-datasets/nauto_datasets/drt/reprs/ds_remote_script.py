import json
import logging
import sys
from typing import Dict

from pyspark.sql import SparkSession

from nauto_datasets.drt.reprs.ds_config import RequestConfig, JoinVideos, JoinVideosType, FailureCases
from nauto_datasets.drt.reprs.ds_data import PackagedEvents, DRTEvents
from nauto_datasets.drt.reprs.ds_format import DSFormatter

DRT = 'drt'
EVENT_PACKAGER = 'ep'


def main(name: str, bucket: str, key: str, source: str, query: str, config: Dict):
    logging.info('Creating dataset in location s3://%s/%s/%s', bucket, key, name)
    logging.info('Config used:\n%s', config)
    logging.info('Using source [%s]', source)
    logging.info('Querying events from:\n%s', query)

    spark = SparkSession.builder.master('local[*]').enableHiveSupport().getOrCreate()

    if source == DRT:
        events = DRTEvents(query, spark=spark)
    elif source == EVENT_PACKAGER:
        events = PackagedEvents(query, spark=spark)
    else:
        raise ValueError('[source] should be set either to [drt] or [ep]')

    if 'join_videos' in config:
        config['join_videos'] = JoinVideos(config['join_videos'])
    if 'join_videos_type' in config:
        config['join_videos_type'] = JoinVideosType(config['join_videos_type'])
    if 'handle_failure_cases' in config:
        config['handle_failure_cases'] = [FailureCases(fc) for fc in config['handle_failure_cases']]
    request_config = RequestConfig(**config)

    DSFormatter(project_name=name, project_bucket=bucket, project_key=key, spark=spark, in_qubole=False). \
        generate_data_set(request_config, events)


if __name__ == '__main__':
    log_level = logging.getLevelName('INFO')
    log_format = '[%(levelname)s] %(asctime)s: %(message)s'
    logging.basicConfig(format=log_format, level=log_level)

    argv = json.loads(sys.argv[1])
    main(**argv)
