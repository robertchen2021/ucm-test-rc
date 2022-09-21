from collections import OrderedDict
from datetime import datetime
from typing import List, Dict

from nauto_zoo import Model, ModelResponse, ModelInput, IncompleteInputMediaError, MalformedModelInputError
from .processing_funcs import get_tailgating_time_intervals, get_time_intervals
from .utils import com_rec_to_df

LABELS_MAPPING = {
    "looking-away": "visual-distraction",
    "holding-object": "manual-distraction",
    "tailgating": "leading-vehicle",
    "no-face": "coachable-no-face"
}

META_KEY = "eventlabeler"


class EventLabelerModel(Model):

    def run(self, model_input: ModelInput) -> ModelResponse:

        assert 'tg_score_th' in self._config
        assert 'tg_duration_th' in self._config
        assert 'distraction_score_th' in self._config
        assert 'distraction_duration_th' in self._config
        assert 'holding_object_score_th' in self._config
        assert 'holding_object_duration_th' in self._config
        assert 'no_face_score_th' in self._config
        assert 'no_face_duration_th' in self._config

        if 'message_id' not in model_input.metadata['event']:
            raise MalformedModelInputError("'message_id' was not found in model_input.metadata['event']")

        message_id = model_input.metadata['event']['message_id']
        self._logger.info(f'EventLabelerModel, message_id: {message_id}, starting...')

        if 'event_packager_id' not in model_input.metadata['event']:
            raise MalformedModelInputError("'event_packager_id' was not found in model_input.metadata['event']")
        if 'type' not in model_input.metadata['event']:
            raise MalformedModelInputError("'type' was not found in model_input.metadata['event']")
        if 'params' not in model_input.metadata['event']:
            raise MalformedModelInputError("'params' was not found in model_input.metadata['event']")

        params = model_input.metadata['event']['params']
        device_event_type = model_input.metadata['event']['type']

        if 'maneuver_data' in params:  # parameters.version = 4
            event_start_sensor_ns = params['event_start_sensor_ns']
            event_end_sensor_ns = params['event_end_sensor_ns']
            threshold_file_info = params['maneuver_data']['threshold_file_info']
        elif 'abcc_data' in params:  # parameters.version = 3
            event_start_sensor_ns = params['abcc_data']['event_start_sensor_ns']
            event_end_sensor_ns = params['abcc_data']['event_end_sensor_ns']
            threshold_file_info = params['abcc_data']['threshold_file_info']
        else:
            raise MalformedModelInputError("Neither 'abcc_data' nor 'maneuver_data' was found in 'params'")

        if "utc_boot_time_ns" not in params:
            raise MalformedModelInputError("utc_boot_time_ns was not found in 'params'")
        if "utc_boot_time_offset_ns" not in params:
            raise MalformedModelInputError("utc_boot_time_offset_ns was not found in 'params'")

        utc_basetime = params["utc_boot_time_ns"] + params["utc_boot_time_offset_ns"]

        raw_output = OrderedDict({
            'device_id': '',
            'message_id': message_id,
            'event_packager_id': model_input.metadata['event']['event_packager_id'],
            'device_event': [device_event_type],
            'device_event_times_ns': [[int(event_start_sensor_ns), int(event_end_sensor_ns)]],
            'vehicle_profile': threshold_file_info.split('-')[1],
            'message_params': '',
            'model_params': self._config,
            'sensor_offset_ns': None,
            'tailgating_times_ns': [],
            'tailgating_min_dist': [],
            'looking_away_times_ns': [],
            'looking_away_max_score': [],
            'holding_object_times_ns': [],
            'holding_object_max_score': [],
            'no_face_times_ns': [],
            'no_face_max_score': [],
        })

        com_rec = model_input.get('sensor')

        if not com_rec:
            raise IncompleteInputMediaError('No sensor CombinedRecording was given in the model_input.')

        event_data = com_rec_to_df(com_rec)
        sensor_offset_ns = int(event_data.sensor_ns.min())

        tg_times_ns, tg_min_distance = get_tailgating_time_intervals(
            event_data,
            subject_cols=['front_box_index', 'distance_estimate', 'score_tailgating'],
            score_th=self._config['tg_score_th'],
            duration_th=self._config['tg_duration_th'])

        distraction_times_ns, distraction_max_score = get_time_intervals(
            event_data,
            subject_cols=['score_looking_down', 'score_looking_left', 'score_looking_right', 'score_looking_up'],
            score_th=self._config['distraction_score_th'],
            duration_th=self._config['distraction_duration_th'])

        holding_object_times_ns, holding_object_max_score = get_time_intervals(
            event_data,
            subject_cols=['score_cell_phone', 'score_holding_object'],
            score_th=self._config['holding_object_score_th'],
            duration_th=self._config['holding_object_duration_th'])

        no_face_times_ns, no_face_max_score = get_time_intervals(
            event_data,
            subject_cols=['score_no_face'],
            score_th=self._config['no_face_score_th'],
            duration_th=self._config['no_face_duration_th'])

        summary = 'TRUE'
        score = 1.
        confidence = 100

        LABELS_MAPPING["device-event"] = device_event_type

        label_value_ts = [
            ("tailgating", tg_times_ns),
            ("looking-away", distraction_times_ns),
            ("holding-object", holding_object_times_ns),
            ("no-face", no_face_times_ns),
            ("device-event", [[int(event_start_sensor_ns), int(event_end_sensor_ns)]])
        ]
        ep_label = []
        for label_value, ts_ns in label_value_ts:
            ep_label.extend(self._get_labels(label_value, ts_ns, utc_basetime, confidence))

        raw_output['sensor_offset_ns'] = sensor_offset_ns
        raw_output['tailgating_times_ns'] = tg_times_ns
        raw_output['tailgating_min_dist'] = tg_min_distance
        raw_output['looking_away_times_ns'] = distraction_times_ns
        raw_output['looking_away_max_score'] = distraction_max_score
        raw_output['holding_object_times_ns'] = holding_object_times_ns
        raw_output['holding_object_max_score'] = holding_object_max_score
        raw_output['no_face_times_ns'] = no_face_times_ns
        raw_output['no_face_max_score'] = no_face_max_score
        raw_output['ep_label'] = ep_label

        self._logger.info(f'EventLabelerModel, message_id: {message_id}, number of labels: {len(ep_label)}')

        return ModelResponse(summary, score, confidence, raw_output)

    def _get_labels(self, label_value: str, ts_ns: List[List], utc_basetime: int, confidence: int) -> List[Dict]:
        labels = []
        base_label_id = LABELS_MAPPING[label_value]
        for i, (start, end) in enumerate(ts_ns):
            start_utc = datetime.utcfromtimestamp((utc_basetime + start) / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_utc = datetime.utcfromtimestamp((utc_basetime + end) / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            labels.append(
                {
                    "label_value": label_value,
                    "label_id": f"{base_label_id}-{i}",
                    "meta_key": META_KEY,
                    "confidence": confidence,
                    "timestamps": [{"start": f"{start_utc}", "end": f"{end_utc}"}]
                }
            )

        return labels
