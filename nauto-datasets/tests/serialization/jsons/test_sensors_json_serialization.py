import inspect
import json
from pathlib import Path

import tensorflow as tf

from nauto_datasets.core import sensors
from nauto_datasets.core.streams import CombinedStreamMixin
from nauto_datasets.serialization.jsons import sensors as s2j
from nauto_datasets.utils import protobuf
from sensor import sensor_pb2

SENSOR_DATA_DIR = Path(__file__).parents[2] / 'test_data' / 'sensor_data'


def get_recording() -> sensors.CombinedRecording:
    files = list((SENSOR_DATA_DIR / '2_11_bboxes_tailgating').iterdir())
    files = sorted(files, key=lambda p: p.name)

    proto_msgs = [
        protobuf.parse_message_from_gzipped_file(
            sensor_pb2.Recording, file_path)
        for file_path in files
    ]

    com_recordings = sensors.CombinedRecording.from_recordings(
        [sensors.Recording.from_pb(r_pb)
            for r_pb in proto_msgs])
    com_recordings = com_recordings.to_utc_time()
    return com_recordings


class CombinedRecordingJsonSerializationTest(tf.test.TestCase):

    def test_serialization_to_json(self):

        com_rec = get_recording()
        com_rec_json = s2j.combined_recording_to_json(com_rec)
        json_str = json.dumps(com_rec_json)
        des_json = json.loads(json_str)

        self.assertSetEqual(set(des_json.keys()), set(['data', 'metadata']))
        self.assertSetEqual(set(des_json['metadata'].keys()), set(['file_info']))

        tags = []
        for el in des_json['data']:
            tags.append(el['tag'])
            if el['tag'] != 'ekf':
                self.assertSetEqual(set(el.keys()), set(['data', 'tag', 'lengths']))
            else:
                self.assertSetEqual(
                    set(el.keys()),
                    set(['data', 'tag', 'configs', 'lengths']))

            com_str = getattr(com_rec, el['tag'])
            self.assertAllEqual(el['lengths'], com_str.lengths)

            data = el['data']
            stream_dict = com_str.stream._asdict()
            self.assertSetEqual(set(data.keys()), set(stream_dict.keys()))

            for key in data:
                if isinstance(stream_dict[key], list):
                    # nested List, e.g. BoundingBoxStream
                    for ind, elem in enumerate(stream_dict[key]):
                        elem_dict = elem._asdict()
                        for sub_key in elem_dict:
                            try:
                                self.assertAllClose(data[key][ind][sub_key], elem_dict[sub_key])
                            except TypeError:
                                self.assertAllEqual(data[key][ind][sub_key], elem_dict[sub_key])
                else:
                    try:
                        self.assertAllClose(data[key], stream_dict[key])
                    except TypeError:
                        self.assertAllEqual(data[key], stream_dict[key])

        exp_tags = [
            name for name, field_t in sensors.CombinedRecording._field_types.items()
            if inspect.isclass(field_t) and issubclass(field_t, CombinedStreamMixin)
        ]
        self.assertSetEqual(set(tags), set(exp_tags))

        self.assertEqual(
            len(des_json['metadata']['file_info']),
            len(com_rec.metadatas))
