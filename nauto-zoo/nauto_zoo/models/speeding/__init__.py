"""
The definition of speeding:
speeding = max("Actual Speed" - "Posted Speed")
"max" has to be taken over the time interval where "Actual Speed" > "Posted Speed".

The definition of severity:
severity = low if                 speeding <=  speed_th1
severity = mid if      speed_th1< speeding <=  speed_th2
severity = high if     speed_th2< speeding

timeTh = 3 sec, grace period (for trip-report, the default value is 10 sec)
speed_th0 = 5 mph, the default tolerance
speed_th1 = 12mph, the first severity level
speed_th2 = 20 mph, the second severity level
"""

import json
import os
from typing import Any, Dict

import requests

from nauto_zoo import Model, ModelResponse, ModelInput, IncompleteInputMediaError

SPEEDING_SERVICE_URL = os.environ.get('LOCATION_API_ENDPOINT', 'http://127.0.0.1:3000/v1') + '/speeding'
MPH_TO_M_SEC = 0.44704  # From MPH to m/sec


class SpeedingModel(Model):

    def run(self, model_input: ModelInput) -> ModelResponse:

        self._logger.info(f"Speeding service URL: {SPEEDING_SERVICE_URL}")

        assert 'time_th' in self._config
        assert 'time_scale' in self._config
        assert 'speed_th0' in self._config
        assert 'speed_th1' in self._config
        assert 'speed_th2' in self._config

        com_rec = model_input.get('sensor')

        if not com_rec:
            raise IncompleteInputMediaError('No sensor CombinedRecording was given in the model_input.')

        # Uncomment it to not use the protected method "_to_df()"
        # data = np.vstack((
        #     com_rec.gps.stream.sensor_ns,
        #     com_rec.gps.stream.longitude,
        #     com_rec.gps.stream.latitude,
        #     com_rec.gps.stream.speed))
        # gps = pd.DataFrame(data=data.T, columns=["sensor_ns", "longitude", "latitude", "speed"])
        # gps.sort_values(by="sensor_ns", inplace=True)
        # gps.reset_index(inplace=True, drop=True)

        gps = com_rec.gps.stream._to_df().sort_values('sensor_ns').reset_index(drop=True)

        self.data = {
            "speed_th0": self._config["speed_th0"] * MPH_TO_M_SEC,
            "speed_th1": self._config["speed_th1"] * MPH_TO_M_SEC,
            "speed_th2": self._config["speed_th2"] * MPH_TO_M_SEC,
            "time_th": self._config["time_th"],
            "time_scale": self._config["time_scale"],
            "latitude": gps["latitude"].to_list(),
            "longitude": gps["longitude"].to_list(),
            "timestamps": gps["sensor_ns"].to_list(),
            "speeds": gps["speed"].to_list(),
        }

        self._logger.info(
            f"speed_th0={self.data['speed_th0']}, speed_th1={self.data['speed_th1']}, " +
            f"speed_th2={self.data['speed_th2']}")
        self._logger.info(f"time_th={self.data['time_th']}, time_scale={self.data['time_scale']}")
        self._logger.info(f"length of latitude={len(self.data['latitude'])}")
        self._logger.info(f"length of longitude={len(self.data['longitude'])}")
        self._logger.info(f"length of timestamps={len(self.data['timestamps'])}")

        res = self._do_post()
        self._logger.info(f"Speeding service response-json: {res}")
        if not res["success"]:
            return ModelResponse('FALSE', 0., 0, [])

        segs = res["response"]
        if not segs:
            return ModelResponse('FALSE', 1., 100, segs)

        # TODO: After collecting the experimental data, provide a sensible score and confidence.
        return ModelResponse('TRUE', 1., 100, segs)

    def _do_post(self) -> Dict[str, Any]:

        res = requests.post(
            url=SPEEDING_SERVICE_URL,
            data=json.dumps(self.data),
            headers={'content-type': 'application/json'}
        )
        self._logger.info(f"Speeding service response : {res}")
        return res.json()
