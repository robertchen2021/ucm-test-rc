# python3.6

from typing import List
# from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import json
import os

from coachable_v0 import process_one_event_merged
# from coachable_v0 import process_one_event, process_one_event_merged
from utils import get_combined_recording


def run_pipeline(protobuf_gzip: List[str],
                 output_file: str) -> None:

    # read the list of protobufs
    com_rec = get_combined_recording(protobuf_gzip)

    # processing: get coachable labels, severity, and times
    coachable_res = process_one_event_merged(com_rec)

    # write to output json file
    with open(output_file, 'w') as output_json:
        json.dump(coachable_res, output_json)
        print(f'wrote: {output_file}')


def main():
    """
    Run one example
    :return:
    """
    output_dir = 'output/'

    for test_id in [0, 1, 2]:
        if test_id == 0:
            # file 0
            file_id = '000186fa0df93adf-15b866dea417653e'
            protobuf_dir = './sensor_files/000186fa0df93adf/2019-08-06/sensor/'
            protobuf_files = ['65b6bf94cdc854cb778575d540b3923109b09142', 'c6e8c0215f6810f52ae63e51bedf4296f1d7aeb6']

        elif test_id == 1:
            # file 1
            file_id = 'fe7c6555480d6e58-X'
            protobuf_dir = './sensor_files/fe7c6555480d6e58/2019-09-22/sensor/'
            protobuf_files = ['9e533b142a84094ac81911f57659479aeacf4208']

        elif test_id == 2:
            # file 2
            file_id = 'fe8657132ccb3ecb-X'
            protobuf_dir = './sensor_files/fe8657132ccb3ecb/2019-09-14/sensor/'
            protobuf_files = ['484c2c23843df4ba70f93c65ba079bcda2a73070']

        else:
            raise ValueError(f'test_id "{test_id}" does not exist.')

        sensor_pb_gzip = [os.path.join(protobuf_dir, pb) for pb in protobuf_files]

        output_file = f'{output_dir}{file_id}_coachable_v0.json'

        print(protobuf_dir)
        try:
            run_pipeline(sensor_pb_gzip, output_file)
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == '__main__':
    main()
