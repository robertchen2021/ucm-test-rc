"""
Run from console:
$ cd nauto-ai/nauto-zoo
$ python -m pytest tests/models/event_labeler/test_event_labeler_model.py
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from nauto_zoo import ModelInput
from nauto_zoo.models.event_labeler.event_labeler_model import EventLabelerModel
from nauto_zoo.preprocess import SensorPreprocessorCombined


class TestEventLabelerModel(unittest.TestCase):

    def test_no_labels(self):
        pwd = Path(__file__).resolve().parent

        files_dir = '166106dcb4aac720'
        with open(os.path.join(pwd, files_dir, "hard_brake.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, files_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {
            'tg_score_th': -1,
            'tg_duration_th': 1.,
            'distraction_score_th': 0.5,
            'distraction_duration_th': 2.0,
            'holding_object_score_th': 0.5,
            'holding_object_duration_th': 1.0,
            'no_face_score_th': 0.5,
            'no_face_duration_th': 1.0,
        }
        model = EventLabelerModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        ep_label_expected = [
            {'label_value': 'device-event',
             'label_id': 'braking-hard-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-06T02:14:46.770629Z", "end": "2021-02-06T02:14:47.000976Z"}]}]

        self.assertListEqual(response.raw_output["ep_label"], ep_label_expected)
        self.assertEqual(response.raw_output["message_id"], event_message["event"]["message_id"])
        self.assertEqual(response.raw_output["event_packager_id"], event_message["event"]["event_packager_id"])
        self.assertEqual(response.raw_output["device_event"], [event_message["event"]["type"]])
        self.assertEqual(response.raw_output["vehicle_profile"], "sedan")
        self.assertListEqual(response.raw_output["device_event_times_ns"],
                             [[int(event_message["event"]["params"]["event_start_sensor_ns"]),
                               int(event_message["event"]["params"]["event_end_sensor_ns"])]])
        self.assertEqual(response.raw_output["sensor_offset_ns"], 14783101418556)
        self.assertFalse(response.raw_output["tailgating_times_ns"])
        self.assertFalse(response.raw_output["looking_away_times_ns"])
        self.assertFalse(response.raw_output["holding_object_times_ns"])
        self.assertFalse(response.raw_output["no_face_times_ns"])
        self.assertFalse(response.raw_output["holding_object_max_score"])
        self.assertFalse(response.raw_output["no_face_max_score"])
        self.assertFalse(response.raw_output["tailgating_min_dist"])
        self.assertFalse(response.raw_output["looking_away_max_score"])

    def test_few_labels(self):
        pwd = Path(__file__).resolve().parent

        files_dir = '166106dcb4aac720'
        with open(os.path.join(pwd, files_dir, "hard_brake.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
        sensor_pb_gzip = [os.path.join(pwd, files_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {
            'tg_score_th': -1,
            'tg_duration_th': 0.5,
            'distraction_score_th': 0.5,
            'distraction_duration_th': 1.0,
            'holding_object_score_th': 0.5,
            'holding_object_duration_th': 1.0,
            'no_face_score_th': 0.5,
            'no_face_duration_th': 1.0,
        }
        model = EventLabelerModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        ep_label_expected = [
            {'label_value': 'tailgating',
             'label_id': 'leading-vehicle-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-06T02:14:30.381749Z", "end": "2021-02-06T02:14:30.962589Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-06T02:14:36.055027Z", "end": "2021-02-06T02:14:37.206699Z"}]},
            {'label_value': 'device-event',
             'label_id': 'braking-hard-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-06T02:14:46.770629Z", "end": "2021-02-06T02:14:47.000976Z"}]}]

        self.assertListEqual(response.raw_output["ep_label"], ep_label_expected)
        self.assertEqual(response.raw_output["message_id"], event_message["event"]["message_id"])
        self.assertEqual(response.raw_output["event_packager_id"], event_message["event"]["event_packager_id"])
        self.assertEqual(response.raw_output["device_event"], [event_message["event"]["type"]])
        self.assertEqual(response.raw_output["vehicle_profile"], "sedan")
        self.assertListEqual(response.raw_output["device_event_times_ns"],
                             [[int(event_message["event"]["params"]["event_start_sensor_ns"]),
                               int(event_message["event"]["params"]["event_end_sensor_ns"])]])
        self.assertEqual(response.raw_output["sensor_offset_ns"], 14783101418556)
        self.assertListEqual(response.raw_output["tailgating_times_ns"], [[14783101418556, 14783682258242]])
        self.assertListEqual(response.raw_output["looking_away_times_ns"], [[14788774695986, 14789926368349]])
        self.assertFalse(response.raw_output["holding_object_times_ns"])
        self.assertFalse(response.raw_output["no_face_times_ns"])
        self.assertFalse(response.raw_output["holding_object_max_score"])
        self.assertFalse(response.raw_output["no_face_max_score"])
        np.testing.assert_almost_equal(response.raw_output["tailgating_min_dist"], [24.6439], 4)
        np.testing.assert_almost_equal(response.raw_output["looking_away_max_score"], [0.8406], 4)

    def test_many_labels(self):
        pwd = Path(__file__).resolve().parent

        files_dir = '16608f33327128fb'
        with open(os.path.join(pwd, files_dir, "hard_brake.json"), 'r') as fh:
            event_message = json.loads(fh.read())

        protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz', 'sensor_3.gz']
        sensor_pb_gzip = [os.path.join(pwd, files_dir, pb) for pb in protobuf_files]
        preprocessor = SensorPreprocessorCombined()
        com_rec = preprocessor.preprocess_sensor_files(sensor_pb_gzip)

        model_input = ModelInput(metadata=event_message)
        model_input.set('sensor', com_rec)
        config_provided_by_ucm = {
            'tg_score_th': -1,
            'tg_duration_th': 0.5,
            'distraction_score_th': 0.5,
            'distraction_duration_th': 1.0,
            'holding_object_score_th': 0.5,
            'holding_object_duration_th': 1.0,
            'no_face_score_th': 0.5,
            'no_face_duration_th': 1.0,
        }
        model = EventLabelerModel(config_provided_by_ucm)
        model.set_logger(Mock())

        response = model.run(model_input)

        ep_label_expected = [
            {'label_value': 'tailgating',
             'label_id': 'leading-vehicle-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:07.018931Z", "end": "2021-02-04T13:42:07.653165Z"}]},
            {'label_value': 'tailgating',
             'label_id': 'leading-vehicle-1',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:11.772946Z", "end": "2021-02-04T13:42:12.536954Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:39.598549Z", "end": "2021-02-04T13:41:40.677193Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-1',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:58.934153Z", "end": "2021-02-04T13:42:01.096385Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-2',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:02.974040Z", "end": "2021-02-04T13:42:04.856669Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-3',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:05.006480Z", "end": "2021-02-04T13:42:06.949015Z"}]},
            {'label_value': 'looking-away',
             'label_id': 'visual-distraction-4',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:07.238706Z", "end": "2021-02-04T13:42:08.327329Z"}]},
            {'label_value': 'holding-object',
             'label_id': 'manual-distraction-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:39.558601Z", "end": "2021-02-04T13:41:40.911903Z"}]},
            {'label_value': 'holding-object',
             'label_id': 'manual-distraction-1',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:44.082863Z", "end": "2021-02-04T13:41:45.196449Z"}]},
            {'label_value': 'holding-object',
             'label_id': 'manual-distraction-2',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:58.609538Z", "end": "2021-02-04T13:42:03.318614Z"}]},
            {'label_value': 'holding-object',
             'label_id': 'manual-distraction-3',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:03.403483Z", "end": "2021-02-04T13:42:10.854123Z"}]},
            {'label_value': 'no-face',
             'label_id': 'coachable-no-face-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:41.211494Z", "end": "2021-02-04T13:41:42.499855Z"}]},
            {'label_value': 'no-face',
             'label_id': 'coachable-no-face-1',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:03.268656Z", "end": "2021-02-04T13:42:04.836711Z"}]},
            {'label_value': 'no-face',
             'label_id': 'coachable-no-face-2',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:42:10.619412Z", "end": "2021-02-04T13:42:15.842801Z"}]},
            {'label_value': 'device-event',
             'label_id': 'braking-hard-0',
             'meta_key': 'eventlabeler',
             'confidence': 100,
             'timestamps': [{"start": "2021-02-04T13:41:55.927901Z", "end": "2021-02-04T13:41:59.343608Z"}]}]

        self.assertListEqual(response.raw_output["ep_label"], ep_label_expected)
        self.assertEqual(response.raw_output["message_id"], event_message["event"]["message_id"])
        self.assertEqual(response.raw_output["event_packager_id"], event_message["event"]["event_packager_id"])
        self.assertEqual(response.raw_output["device_event"], [event_message["event"]["type"]])
        self.assertEqual(response.raw_output["vehicle_profile"], "sedan")
        self.assertListEqual(response.raw_output["device_event_times_ns"],
                             [[int(event_message["event"]["params"]["event_start_sensor_ns"]),
                               int(event_message["event"]["params"]["event_end_sensor_ns"])]])
        self.assertEqual(response.raw_output["sensor_offset_ns"], 22876741098466)
        self.assertListEqual(response.raw_output["tailgating_times_ns"],
                             [[22907926680650, 22908560914610],
                              [22912680696104, 22913444703672]])
        self.assertListEqual(response.raw_output["looking_away_times_ns"],
                             [[22880506298496, 22881584942295],
                              [22899841902818, 22902004134263],
                              [22903881789292, 22905764418687],
                              [22905914229478, 22907856764878],
                              [22908146455381, 22909235078428]])
        self.assertListEqual(response.raw_output["holding_object_times_ns"],
                             [[22880466350986, 22881819652988],
                              [22884990612461, 22886104198887],
                              [22899517287339, 22904226363267],
                              [22904311232652, 22911761872862]])
        self.assertListEqual(response.raw_output["no_face_times_ns"],
                             [[22882119244053, 22883407604648],
                              [22904176405991, 22905744460191],
                              [22911527162168, 22916750550840]])
        np.testing.assert_almost_equal(response.raw_output["holding_object_max_score"],
                                       [0.9951014645973029, 0.9883743547340298, 0.9941209216469475, 0.9993578912486433])
        np.testing.assert_almost_equal(response.raw_output["no_face_max_score"], [0.9255, 0.9472, 0.8046], 4)
        np.testing.assert_almost_equal(response.raw_output["tailgating_min_dist"], [7.3255, 5.8415], 4)
        np.testing.assert_almost_equal(response.raw_output["looking_away_max_score"],
                                       [0.9735, 0.9710, 0.9884, 0.9695, 0.9940], 4)


if __name__ == '__main__':
    unittest.main()
