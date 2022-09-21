from nauto_zoo import Model, ModelInput, ModelResponse
from nauto_zoo.models.utils import infer_confidence
import numpy as np
import pandas as pd
import numpy as np
from scipy import interpolate
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from typing import Dict, Any, Optional, NamedTuple, Union
from .irs import IRS
import boto3
import logging
import os

class InstantaneousRiskPredictor(Model):
    DEFAULT_S3_MODEL_VERSION_DIR = "0.1"
    MODEL_FILES_FOLDER = "/tmp/irs/"
    IRS_MODEL_FILE = MODEL_FILES_FOLDER + "IRS_MODEL"
    DISTANCE_NORM_FACTOR = 100
    TTC_NORM_FACTOR = 10
    SPEED_NORM_FACTOR = 50
    LOOKBACK_S = 8
    OBSERVATION_WINDOW = 30
    ALL_METADATA_COLUMNS = ['distance_estimate', 'score_looking_down', 'score_looking_up', 'score_looking_left',
                            'score_looking_right', 'score_cell_phone', 'score_smoking', 'score_holding_object',
                            'score_eyes_closed', 'score_no_face', 'ttc', 'speed']

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # assert "threshold" in self._config
        self.logger = logging.getLogger()
        self.logger.info('logger started')
        self.bootstrapped = False
        self._try_load()

    def bootstrap(self):
        if not self.bootstrapped:
            model_dir =  str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "irs/"+model_dir+"/model.hdf5", self.IRS_MODEL_FILE)
            self._try_load()

    def manual_bootstrap(self):
        if not self.bootstrapped:
            app_logger = logging.getLogger('model')
            self.set_logger(app_logger)

            session = boto3.session.Session(profile_name='test-us')
            s3_client = session.resource('s3', region_name='us-east-1')
            self.set_s3(s3_client)

            model_dir = str(self._config.get("model_version", self.DEFAULT_S3_MODEL_VERSION_DIR))
            _ = self._download_from_s3
            os.makedirs(self.MODEL_FILES_FOLDER, exist_ok=True)
            _("nauto-cloud-models-test-us", "irs/" + model_dir + "/model.hdf5", self.IRS_MODEL_FILE)
            self._try_load()

    def _try_load(self):
        if os.path.isfile(self.IRS_MODEL_FILE):
            run_config = RunConfig()
            self.irs_model = IRS(run_config, self.IRS_MODEL_FILE)
            self.logger.info('irs model loaded')
            self.bootstrapped = True

    def get_irs_input_tensor_from_com_rec(self, com_recordings):
        if com_recordings.rt_acc.stream.sensor_ns.shape[0] > 0:
            acc_stream_name = 'rt_acc'
        elif com_recordings.acc.stream.sensor_ns.shape[0] > 0:
            acc_stream_name = 'acc'
        else:
            return -1

        df_acc = getattr(com_recordings, acc_stream_name).stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)

        df_acc = df_acc.set_index('timestamp_utc_ns', drop=False, inplace=False)
        df_acc.sort_index(inplace=True)
        df_acc = df_acc.dropna()
        max_idx = (df_acc.x ** 2 + df_acc.y ** 2 + df_acc.z ** 2).values.argmax()
        max_ns = df_acc.timestamp_utc_ns.values[max_idx]

        # peak_ns =  max_ns + com_recordings.metadatas[0].utc_boot_time_ns + com_recordings.metadatas[0].utc_boot_time_offset_ns
        peak_ns = max_ns
        t_ref = np.arange(int(peak_ns - (self.LOOKBACK_S * 1e9)), peak_ns, 200 * 1e6, dtype=np.uint64)
        ref_df = pd.DataFrame(t_ref, columns=['timestamp_utc_ns'])

        dist_df = getattr(com_recordings, 'dist_multilabel').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        dist_df = dist_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        dist_df.sort_index(inplace=True)

        tail_df = getattr(com_recordings, 'tailgating').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        tail_df = tail_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        tail_df.sort_index(inplace=True)

        gps_df = getattr(com_recordings, 'gps').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        gps_df = gps_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        gps_df.sort_index(inplace=True)

        fcw_df = getattr(com_recordings, 'fcw').stream._to_df().drop('system_ms', axis=1).rename(
            columns={'sensor_ns': 'timestamp_utc_ns'}).drop_duplicates(subset=['timestamp_utc_ns']).reset_index(
            drop=True)
        fcw_df.set_index('timestamp_utc_ns', drop=False, inplace=False)
        fcw_df.sort_index(inplace=True)

        tail_df.drop(columns=['score', 'front_box_index'], inplace=True)

        self.logger.info('Stream lengths - ref: %d, gps: %d, dist: %d, tail: %d, fcw: %d' % (
            ref_df.shape[0], gps_df.shape[0], dist_df.shape[0], tail_df.shape[0], fcw_df.shape[0]))
        
        if dist_df.timestamp_utc_ns.values[0] > t_ref[0]:
            return None

        if tail_df.shape[0] > 1:
            get_from = interpolate.interp1d(tail_df.timestamp_utc_ns.values, tail_df.distance_estimate.values,
                                            fill_value='extrapolate', kind='nearest')
            ref_df['distance_estimate'] = get_from(ref_df.timestamp_utc_ns.values)
        else:
            ref_df['distance_estimate'] = 0 * ref_df.timestamp_utc_ns.values + 100
        dist_labels = ['score_looking_down', 'score_looking_up', 'score_looking_left', 'score_looking_right',
                       'score_cell_phone', 'score_smoking', 'score_holding_object', 'score_eyes_closed',
                       'score_no_face']
        for dist_label in dist_labels:
            get_from = interpolate.interp1d(dist_df.timestamp_utc_ns.values, dist_df[dist_label].values,
                                            fill_value='extrapolate', kind='nearest')
            ref_df[dist_label] = get_from(ref_df.timestamp_utc_ns.values)

        if fcw_df.shape[0] > 1:
            get_from = interpolate.interp1d(fcw_df.timestamp_utc_ns.values, fcw_df.ttc.values, fill_value='extrapolate',
                                            kind='nearest')
            ttc_vals = get_from(ref_df.timestamp_utc_ns.values)
            invalid_locations = np.where((ttc_vals < 0.5) | (ttc_vals > 6))[0]  # force all <0.5 and >6 s to 10
            ttc_vals[invalid_locations] = 10
            ref_df['ttc'] = ttc_vals
        else:
            ref_df['ttc'] = 0 * ref_df.timestamp_utc_ns.values + 10
        get_from = interpolate.interp1d(gps_df.timestamp_utc_ns.values, gps_df.speed.values, fill_value='extrapolate',
                                        kind='nearest')
        ref_df['speed'] = get_from(ref_df.timestamp_utc_ns.values)
        sample_input = ref_df.loc[:, self.ALL_METADATA_COLUMNS].values
        sample_input[:, 0] = sample_input[:, 0] / self.DISTANCE_NORM_FACTOR  # Normalize distance
        sample_input[:, 10] = sample_input[:, 10] / self.TTC_NORM_FACTOR  # Normalize ttc
        sample_input[:, 11] = sample_input[:, 11] / self.SPEED_NORM_FACTOR  # Normalize speed
        return np.expand_dims(np.expand_dims(sample_input[:self.OBSERVATION_WINDOW, :], axis=0), axis=-1)


    def run(self, model_input: ModelInput) -> ModelResponse:
        assert self.bootstrapped
        inputs = model_input.get('sensor')
        input_tensor = self.get_irs_input_tensor_from_com_rec(inputs)
        if input_tensor is None:
            raise ValueError("Not a valid input")
        raw_output = self.irs_model.model.predict(input_tensor)
        pred_class = np.argmax(raw_output)
        pred_score = np.max(raw_output)

        response = ModelResponse(
            summary="TRUE" if pred_class > 0 else "FALSE",
            score=np.max(raw_output).tolist(),
            confidence=int(100 * pred_score),
            raw_output=None
        )
        self.logger.info('message_id:%s, class:%f, score:%f' % (model_input.metadata['event']['message_id'], pred_class, pred_score))

        return response

class RunConfig(NamedTuple):
    class_name: Optional[str] = None
    tf_xla: bool = True
    preferred: bool = False