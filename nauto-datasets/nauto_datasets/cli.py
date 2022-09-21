import json
import logging
import shlex
from pathlib import Path

import click
import yaml
from pyfiglet import Figlet
from qds_sdk.commands import SparkCommand
from qds_sdk.qubole import Qubole

log_level = logging.getLevelName('INFO')
log_format = '[%(levelname)s] %(asctime)s: %(message)s'
logging.basicConfig(format=log_format, level=log_level)
logger = logging.getLogger(__name__)

@click.group()
def datasets():
    print(Figlet(font='slant').renderText('nauto-datasets'))


@datasets.group()
@click.option('--qubole-token', '-q', required=True, type=str, help='auth token for qubole', envvar='QUBOLE_API_TOKEN')
@click.option('--qubole-api-url', '-a', required=False, default='https://us.qubole.com/api/',
              type=str, help='api url for qubole', show_default=True, envvar='QUBOLE_API_URL')
@click.option('--cluster', '-c', required=True, type=str, help='qubole spark cluster label')
@click.option('--verbose', is_flag=True, default=False, show_default=True,
              help='prints such information like log messages from remote cluster')
@click.pass_context
def generate(ctx, qubole_token: str, qubole_api_url: str, cluster: str, verbose: bool):
    ctx.ensure_object(dict)

    ctx.obj['qubole_token'] = qubole_token
    ctx.obj['qubole_api_url'] = qubole_api_url
    ctx.obj['cluster'] = cluster
    ctx.obj['verbose'] = verbose


@generate.command()
@click.option('--name', '-n', type=str, required=True, help='name of the dataset')
@click.option('--bucket', '-b', type=str, required=False, default='nauto-prod-ai', show_default=True,
              help='S3 bucket the set will be generated to')
@click.option('--key', '-k', type=str, required=True, help='key at which dataset will be generated')
@click.option('--source', '-s', type=click.Choice(['drt', 'ep']), required=True,
              help='tells from which system the data is taken from')
@click.option('--query', '-q', type=Path, required=True, help='path to events query')
@click.option('--config', '-c', type=Path, required=True, help='path to configuration file')
@click.pass_context
def ds(ctx, name: str, bucket: str, key: str, source: str, query: Path, config: Path):
    logger.info(f'Generating dataset to s3://{bucket}/{key}/data/processed/{name}')
    config_content = yaml.load(config.read_text(), Loader=yaml.FullLoader)
    logger.info(f'Config to be used:\n{config_content}')
    query_content = query.read_text()
    logger.info(f'Query to be used:\n{query_content}')

    Qubole.configure(api_token=ctx.obj['qubole_token'], api_url=ctx.obj['qubole_api_url'])

    user_arguments = {'name': name, 'bucket': bucket, 'key': key, 'source': source, 'query': query_content,
                      'config': config_content}

    command_arguments = [
        '--cluster-label', ctx.obj['cluster'],
        '--script_location', str((Path(__file__).parent / 'drt' / 'reprs' / 'ds_remote_script.py').resolve()),
        '--user_program_arguments', shlex.quote(json.dumps(user_arguments)),
        '--arguments', '--conf spark.pyspark.driver.python=/usr/lib/anaconda2/envs/py36/bin/python '
                       '--conf spark.pyspark.python=/usr/lib/anaconda2/envs/py36/bin/python '
    ]

    logger.info(f'Launching spark job in cluster [{ctx.obj["cluster"]}]. Check https://us.qubole.com/v2/analyze')
    args = SparkCommand.parse(command_arguments)
    if ctx.obj['verbose']:
        args['print_logs_live'] = True
    cmd = SparkCommand.run(**args)

    logger.info(f'Command result is:\n{cmd}')


def main():
    datasets(obj={})


if __name__ == '__main__':
    main()
