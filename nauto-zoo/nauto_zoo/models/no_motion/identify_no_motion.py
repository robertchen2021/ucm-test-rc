import json
import numpy as np
import pandas as pd
import pprint
from .utils import Utils
import os


class IdentifyNoMotion:
    def __init__(self, json_file_path, output_folder, logger):
        self.logger = logger
        self.tags = None
        self.df = {}
        self.channels = ['acc', 'gyro']
        self.json_file_path = json_file_path
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.logger.info("%s does not exist, creating one ...")

        self.max_mov_std = {'acc': -1, 'gyro': -1}
        self.NO_MOTION_THRESHOLD = {'acc': 0.1, 'gyro': 0.005}
        self.WINDOW_SIZE = 200
        self.AXES = ['x', 'y', 'z']
        self.results = {'std': {},
                        'max_mov_std': {},
                        'window_size': self.WINDOW_SIZE,
                        'is_no_motion': True,
                        'thresholds': self.NO_MOTION_THRESHOLD}

    def compute_moving_std(self, window_size=200):
        self.logger.info("computing moving stds based window size %s" % window_size)
        for channel in self.channels:
            mov_stds = []

            for axis in self.AXES:
                signal = self.df[channel][axis].values
                mov_std, max_val = Utils.movstd(signal, w=window_size)
                mov_stds.append(np.round(max_val, 8))

                if max_val > self.max_mov_std[channel]:
                    self.max_mov_std[channel] = max_val

            self.results['max_mov_std'][channel] = np.array(mov_stds)

    def get_data_from_json_ojb(self, json_obj):
        self.tags = {json_obj[i]['tag']: i for i in range(len(json_obj))}
        self.logger.debug(pprint.pformat(self.tags, indent=1))

        df = {}
        for channel in self.channels:
            df[channel] = pd.DataFrame(index=json_obj[self.tags[channel]]['data']['sensor_ns'])
            for axis in self.AXES:
                df[channel][axis] = json_obj[self.tags[channel]]['data'][axis]
            df[channel].sort_index(inplace=True)
        return df

    def get_data_from_json_file_path(self, json_file_path):
        self.logger.info("Extracting Json file from %s" % json_file_path)
        with open(json_file_path, 'rb') as f:
            data = json.load(f)['data']
        return self.get_data_from_json_ojb(data)

    def compute_std(self):
        self.logger.info("computing overall stds ...")
        for channel in self.channels:
            self.results['std'][channel] = np.std(self.df[channel].copy(), axis=0).values

    def make_judgement(self):

        for key in ['max_mov_std', 'std']:
            for channel in self.channels:
                if np.min(self.results[key][channel]) > self.NO_MOTION_THRESHOLD[channel]:
                    self.logger.info("Threshold exceeded: np.min(self.results[{key}][{channel}])"
                                     " > self.NO_MOTION_THRESHOLD[{channel}]".format(key=key, channel=channel))
                    self.results['is_no_motion'] = False
                    break

    def dump_json_results(self):
        result = pprint.pformat(self.results, indent=2)
        self.logger.info(result)
        json_dump_path = os.path.join(self.output_folder, "result.json")
        json.dump(result, open(json_dump_path, 'w'))
        self.logger.info("Results dumped to %s" % json_dump_path)

    def run(self):
        self.df = self.get_data_from_json_file_path(self.json_file_path)
        self.compute_std()
        self.compute_moving_std()

        self.make_judgement()
        self.dump_json_results()
