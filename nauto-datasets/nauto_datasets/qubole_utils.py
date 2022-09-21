"""Common utilities for Qubole."""

import base64
import glob
import uuid
from io import BytesIO, StringIO
from typing import Sequence, Tuple, Optional
from uuid import uuid1, uuid4

import boto3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
from qds_sdk.commands import *


def initialize_qubole_display():
    """This function should be called once per notebook initialization"""
    pd.set_option('display.max_colwidth', -1)
    mpl.use('Agg')
    plt.rcdefaults()


def get_lists_unique_values(values):
    """Get unique set of values from multiple numpy lists."""
    if len(values) > 0:
        return list(sorted(set(np.concatenate(values))))
    else:
        return []


def get_list_columns_unique_values(row, cols):
    """Get unique set of values from columns containing lists."""
    values = []
    for col in cols:
        if type(row[col]) in (list, np.array, np.ndarray):
            values.append(row[col])
    return get_lists_unique_values(values)


def format_datetimes(x):
    """Clean formatting of datetime values for display."""
    if isinstance(x, pd.Timestamp) is True:
        return x.strftime('%Y-%m-%d %H:%M:%S.%f')
    elif isinstance(x, pd.Timedelta) is True:
        return '{:0>2}:{:0>2}:{:0>2}'.format(
            x.components.hours + 24 * x.components.days, x.components.minutes,
            x.components.seconds)
    else:
        return x


def format_html_df(df,
                   styles=None,
                   table_style=None,
                   uuid=None,
                   caption=None,
                   max_height=500):
    """Produce styled, formatted Pandas DataFrame for HTML display."""
    if uuid is None:
        uuid = str(uuid1()).replace("-", "_")

    if styles is None:
        styles = [
            dict(selector='*',
                 props=[('border-color', '#f0f0f0 !important'),
                        ('border', '1px solid'), ('margin', '6px')]),
            dict(selector='th',
                 props=[('background', 'lightgray'), ('text-align', 'center'),
                        ('padding', '6px')]),
            dict(selector='td',
                 props=[('text-align', 'right'), ('min-width', '5em')]),
            dict(selector='caption',
                 props=[('caption-side', 'top'), ('border', 'none')]),
            dict(selector='thead',
                 props=[('background-color', 'lightgray'),
                        ('margin', '20px')]),
            dict(selector='tr:nth-child(even)',
                 props=[('background-color', '#ffffff')]),
            dict(selector='tr:hover', props=[('background-color', '#ffff99')])
        ]

    if table_style is None:
        table_style = '''<style>
            table#{uuid} {{display: block; border-collapse: collapse;
                           table-layout: fixed; border: none;
                           overflow: auto; max-height: {max_height}px;}}
            </style>'''.format(uuid='T_{}'.format(uuid), max_height=max_height)

    df_style = df.style.set_table_styles(styles)
    df_style.set_uuid(uuid)

    if caption is not None:
        df_style.set_caption(caption)

    df_html = df_style.render()

    # Remove unnessary html tag
    df_html = re.sub(r'%html ', '', df_html)

    return table_style + df_html


def show_df(df,
            convert_timestamps=True,
            limit_results=100,
            caption=None,
            max_height=500):
    """Display Pandas DataFrame as HTML."""
    out_df = df.copy()

    if isinstance(out_df, pd.Series) is True:
        out_df = pd.DataFrame(out_df)

    if convert_timestamps is True:
        out_df = out_df.applymap(format_datetimes)

    if limit_results is False:
        limit_results = out_df.shape[0]

    if len(out_df) > limit_results:
        out_df_len = len(out_df)
        out_df = out_df.iloc[0:limit_results]
        limit_str = '\n%html <font size="3" color="red">{} results shown of {}</font>'.format(
            limit_results, out_df_len)
    else:
        limit_str = ''

    print('%html ' +
          format_html_df(out_df, caption=caption, max_height=max_height) +
          limit_str)


def show_plotly(plot_dic, width='100%', height=525, **kwargs):
    """Display Plotly charts as HTML."""
    figure = plotly.tools.return_figure_from_figure_or_data(plot_dic, True)
    width = figure.get('layout', {}).get('width', width)
    height = figure.get('layout', {}).get('height', height)
    plotdivid = str(uuid4()).replace("-", "_")
    kwargs['output_type'] = 'div'
    plot_str = plotly.offline.plot(plot_dic,
                                   config={
                                       'showLink': False,
                                       'displayModeBar': False
                                   },
                                   **kwargs)
    print("""%angular
          <div id="{}" style="height: {}; width: {};" class="plotly-graph-div"> {} </div>
          """.format(plotdivid, height, width, plot_str))


def show_matplotlib(p, width=100):
    """Display Matplotlib charts as HTML."""
    img = StringIO()
    p.savefig(img, format='svg')
    p.clf()
    img.seek(0)
    print('%html <div style="width:{}%">'.format(width) + img.read() +
          '</div>')


def create_sql_str(input_list):
    """Convert list into Spark SQL string."""
    if len(input_list) == 1:
        return "('{}')".format(input_list[0])
    else:
        return "{}".format(tuple(input_list))


def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """Get matching keys in an S3 bucket."""
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)
        if 'Contents' not in resp.keys():
            break
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def save_df_to_csv_s3(df, s3_resource, s3_bucket, s3_key, index=True):
    """Save pandas DataFrame to s3 as CSV."""
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=index)
        return s3_resource.Object(s3_bucket,
                                  s3_key).put(Body=csv_buffer.getvalue())
    except BaseException as ex:
        return None


def save_data_to_s3(data, s3_resource, s3_bucket, s3_key, index=True):
    """Save DataFrame or dict to s3."""
    if type(data) == pd.DataFrame:
        return save_df_to_csv_s3(data, s3_resource, s3_bucket, s3_key, index)
    elif type(data) == dict:
        buffer = data.copy()
        body = json.dumps(buffer)
        return s3_resource.Object(s3_bucket, s3_key).put(Body=body)


def load_df_from_csv_s3(s3_client,
                        s3_bucket,
                        s3_key,
                        usecols=None,
                        index_col=None):
    """Load pandas DataFrame from CSV in s3."""
    try:
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        return pd.read_csv(BytesIO(obj['Body'].read()),
                           usecols=usecols,
                           index_col=index_col)
    except:
        return None


def load_data_from_s3(s3_client,
                      s3_bucket,
                      s3_key,
                      usecols=None,
                      index_col=None):
    """Load DataFrame or dict from s3."""
    ext = s3_key.split('.')[-1]
    if ext == 'csv':
        return load_df_from_csv_s3(s3_client, s3_bucket, s3_key, usecols,
                                   index_col)
    elif ext == 'json':
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        return json.loads(obj['Body'].read().decode('utf-8'))


def create_presigned_url(bucket_name, object_name, s3_client, expiration=30):
    """Generate a presigned URL for the S3 object.

    Args:
        expiration (int): Time in seconds for the presigned URL to remain valid.
    """
    try:
        signed_url = s3_client.generate_presigned_url('get_object',
                                                      Params={
                                                          'Bucket':
                                                              bucket_name,
                                                          'Key': object_name
                                                      },
                                                      ExpiresIn=expiration)
        return signed_url
    except:
        return None


def get_param(params, key):
    """Retrieve key from message_params."""
    p = json.loads(params)
    if key in p.keys():
        return p[key]
    else:
        return None


def get_utc_basetime(p):
    """Get UTC basetime from message_params (required to convert any media
    timestamps to UTC)."""
    p = json.loads(p)
    if ('utc_boot_time_ns' in p.keys()) & (
            'utc_boot_time_offset_ns' in p.keys()):
        return p['utc_boot_time_ns'] + p['utc_boot_time_offset_ns']
    else:
        return None


def sync_s3_to_local(s3_resource, bucket, key, local='tmp'):
    """Copy files from s3 to local directory."""
    os.makedirs(local, exist_ok=True)
    bucket = s3_resource.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=key).all():
        file = obj.key.split(key + '/')[-1]
        bucket.download_file(os.path.join(key, file), os.path.join(local, file))


def load_s3_py_files(sc, s3_resource, bucket, key, local='tmp'):
    """Add py files to Spark Context."""
    sync_s3_to_local(s3_resource, bucket, key, local)
    for f in glob.glob(os.path.join(local, '*.py')):
        sc.addPyFile(f)


def play_event_video(s3_client, project_bucket, event, single_camera_video=False, show_event_data=True,
                     get_signed_url=False, expiration=60, width='100%'):
    """Generate a url and play video in HTML video player.

    Args:
        get_signed_url (bool): False to load video directly from S3 into local memory.
            True to generate presigned URL. Allows video to be downloaded. Default False.
        expiration (int): Time in seconds for presigned URL to expire
    """
    try:
        if single_camera_video == 'internal':
            video_filename = '{}-internal.mp4'.format(event.media_filename)
        elif single_camera_video == 'external':
            video_filename = '{}-external.mp4'.format(event.media_filename)
        elif single_camera_video is False:
            video_filename = '{}.mp4'.format(event.media_filename)
        else:
            raise
        vid_key = os.path.join(event.media_filepath, video_filename)
        if get_signed_url is True:
            vid_url = create_presigned_url(project_bucket, vid_key, s3_client, expiration=expiration)
        else:
            obj = s3_client.get_object(Bucket=project_bucket, Key=vid_key)
            vid_url = 'data:video/mp4;base64,{}'.format(base64.b64encode(obj['Body'].read()).decode('ascii'))
        print("""%html
        <video width="{}" controls>
          <source src="{}" type="video/mp4">
        </video>
        """.format(width, vid_url))
        if show_event_data is True:
            show_df(event)
    except:
        print("%html <div>Cannot play video</div>")


class QuboleQuery:
    DELIMITER = "\t"
    TEMP_DIR = '/tmp/qubole/'

    """ object stores query to run along with the expected
    column names used to build the returning dataframe """

    def __init__(self, sql: str, col_names: Sequence[str], tmp_dir: Optional[str] = TEMP_DIR) -> None:
        super(QuboleQuery, self).__init__()

        self.sql = sql
        self.col_names = col_names
        self.tmp_dir = tmp_dir

    def _string(self, concise: bool):
        q = self.sql
        if concise:
            q = q[:300] + '...'

        return "\n\t(query) {sql}\n\t(columns) {col_names}".format(
            sql=q,
            col_names=self.col_names)

    def __str__(self):
        return self._string(concise=True)

    def __repr__(self):
        return self._string(concise=False)

    def _get_temporary_filename(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        fname = 'result_%s.tsv' % str(uuid.uuid4())
        return os.path.join(self.tmp_dir, fname)

    @staticmethod
    def _read_file_content(filename):
        with open(filename, 'r') as content_file:
            content = content_file.read()
        return content

    def _execute_query(self):
        if self.sql is None or self.sql == "":
            return None
        cmd = HiveCommand.create(query=self.sql)
        while not HiveCommand.is_done(cmd.status):
            print(".")
            cmd = HiveCommand.find(cmd.id)
            time.sleep(5)
        return cmd

    def _get_command_results(self, command):
        if command is None:
            return None
        filename = self._get_temporary_filename()
        fp = open(filename, 'w')
        command.get_results(fp, delim=QuboleQuery.DELIMITER)
        while not HiveCommand.is_done(command.status):
            print("\b=>")
            time.sleep(5)

        fp.close()
        content = QuboleQuery._read_file_content(filename)
        return content, filename

    @staticmethod
    def _parse_tsv_result(tsv_str, names):
        return pd.read_csv(StringIO(tsv_str), names=names, sep=QuboleQuery.DELIMITER)

    def query_to_dataframe(self) -> Tuple[pd.DataFrame, str]:
        """
        Uses Qubole API to query and returns results as pandas DataFrame
        """
        command = self._execute_query()
        results_content, results_fpath = self._get_command_results(command)
        df = QuboleQuery._parse_tsv_result(results_content, names=self.col_names)
        return df, results_fpath

    def execute(self, assert_nonzero: bool = True) -> pd.DataFrame:
        """ runs the QuboleQuery object and saves the output dataframe to specified output file """

        df, _ = self.query_to_dataframe()

        if assert_nonzero:
            assert len(df) > 0

        return df
