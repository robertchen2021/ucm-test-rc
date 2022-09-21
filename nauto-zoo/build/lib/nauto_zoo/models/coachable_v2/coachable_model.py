from collections import OrderedDict

from nauto_zoo import Model, ModelResponse, ModelInput, IncompleteInputMediaError, MalformedModelInputError
from .processing_funcs import get_panic_brake_times_v2, get_speed_bump_times, get_turn_times, get_tailgating_times, \
    get_distraction_times, get_holding_object_times, get_no_face_times, get_coachable_times
from .utils import sec_to_sensor_ns, com_rec_to_df


class CoachableModel(Model):

    # TODO: check and extract event metadata to use in the model

    def run(self, model_input: ModelInput) -> ModelResponse:

        # check model parameters
        assert 'turn_time_th' in self._config
        assert 'tg_score_th' in self._config
        assert 'tg_duration_th' in self._config
        assert 'distraction_score_th' in self._config
        assert 'distraction_duration_th' in self._config
        assert 'holding_object_score_th' in self._config
        assert 'holding_object_duration_th' in self._config
        assert 'no_face_score_th' in self._config
        assert 'no_face_duration_th' in self._config
        assert 'seq_time_delta_th' in self._config
        # The next 4 params are for the startle braking candidate algorithm.
        assert 'startle_slope_th' in self._config
        assert 'startle_a_max_th' in self._config
        assert 'startle_a_mid_th' in self._config
        assert 'startle_buffet_length' in self._config

        # check event metadata
        if 'message_id' not in model_input.metadata['event']:
            raise MalformedModelInputError("'message_id' was not found in model_input.metadata['event']")

        message_id = model_input.metadata['event']['message_id']
        self._logger.info(f'CoachableModel, message_id: {message_id}, starting...')

        if 'event_packager_id' not in model_input.metadata['event']:
            raise MalformedModelInputError("'event_packager_id' was not found in model_input.metadata['event']")
        if 'type' not in model_input.metadata['event']:
            raise MalformedModelInputError("'type' was not found in model_input.metadata['event']")
        if 'params' not in model_input.metadata['event']:
            raise MalformedModelInputError("'params' was not found in model_input.metadata['event']")
        event_packager_id = model_input.metadata['event']['event_packager_id']
        message_type = model_input.metadata['event']['type']
        params = model_input.metadata['event']['params']

        if 'hard' in message_type:  # if an ABC event (*-hard or *-hard-detection)
            if 'abcc_data' in params:  # parameters.version = 3
                event_start_sensor_ns = params['abcc_data']['event_start_sensor_ns']
                event_end_sensor_ns = params['abcc_data']['event_end_sensor_ns']
                threshold_file_info = params['abcc_data']['threshold_file_info']
            elif 'maneuver_data' in params:  # parameters.version = 4
                event_start_sensor_ns = params['event_start_sensor_ns']
                event_end_sensor_ns = params['event_end_sensor_ns']
                threshold_file_info = params['maneuver_data']['threshold_file_info']
            else:
                err_msg = f"Input was a '{message_type}' event, but neither 'abcc_data' nor 'maneuver_data' " \
                          f"was found in model_input.metadata['event']['params']"
                raise MalformedModelInputError(err_msg)
        elif message_type == 'risk':  # if a risk-ABC event
            if 'risk_type' not in params:
                raise MalformedModelInputError("'risk_type' was not found in model_input.metadata['event']['params']")
            risk_type = str(params['risk_type']).lower()
            event_start_sensor_ns = params['triggers'][f'final.maneuver_{risk_type}.start_timestamp_ns']
            event_end_sensor_ns = params['triggers'][f'final.maneuver_{risk_type}.end_timestamp_ns']
            threshold_file_info = params['triggers'][f'final.maneuver_{risk_type}.threshold_file_info']
        else:
            raise MalformedModelInputError(f"Only ABC, ABC-detection, or risk-ABC events are accepted. "
                                           f"Got '{message_type}' instead.")

        # initialize the output structure
        summary = 'FALSE'  # this is short value describing result of the inference
        score = 0.
        confidence = 100  # this tells how certain the output is, should be in range [0; 100]
        raw_output = OrderedDict({
            'device_id': '',
            'message_id': message_id,
            'event_packager_id': event_packager_id,
            'device_event': [message_type],
            'device_event_times_ns': [[int(event_start_sensor_ns), int(event_end_sensor_ns)]],
            'vehicle_profile': threshold_file_info.split('-')[1],
            'message_params': '',
            # 'message_params': model_input.metadata['message_params'],
            'model_params': self._config,  # save the model parameters
            'sensor_offset_ns': None,
            'coachable_times_ns': [],
            'coachable_severity': [],
            'brake_times_ns': [],
            'brake_severity': [],
            'startle_times_ns': [],
            'startle_severity': [],
            'speed_bump_times_ns': [],
            'speed_bump_severity': [],
            'turn_times_ns': [],
            'turn_radii': [],
            'tailgating_times_ns': [],
            'tailgating_min_dist': [],
            'looking_away_times_ns': [],
            'looking_away_max_score': [],
            'holding_object_times_ns': [],
            'holding_object_max_score': [],
            'no_face_times_ns': [],
            'no_face_max_score': [],
        })

        # assuming that model_input.sensor is a CombinedRecording object
        com_rec = model_input.get('sensor')

        if not com_rec:
            raise IncompleteInputMediaError('No sensor CombinedRecording was given in the model_input.')

        else:
            # Extract sensor data and video_scores from the com_rec
            event_data = com_rec_to_df(com_rec)
            # event_data['device_id'] = device_id
            # event_data['event_id'] = event_id

            # Get the sensor_offset_ns
            sensor_offset_ns = int(event_data.sensor_ns.min())

            # Process for imu-based algos
            event_data.sort_values('sensor_ns', inplace=True)
            event_data.reset_index(inplace=True, drop=True)

            event_data['time_s'] = (event_data.sensor_ns - sensor_offset_ns) * 1e-9
            # device_id = event_data.device_id.unique().tolist()[0]
            # event_id = event_data.event_id.unique().tolist()[0]
            time, accx, accy, gyrz, gpsspeed = \
                [event_data[col].values for col in ['time_s', 'acc_x', 'acc_y', 'gyr_z', 'speed']]
            # print([len(x) for x in [time, accx, accy, gyrz, gpsspeed]])

            # brakes and startle brakes
            brake_times, brake_severity, startle_times, startle_severity = get_panic_brake_times_v2(
                time,
                accx,
                gpsspeed,
                self._config["startle_slope_th"],
                self._config["startle_a_max_th"],
                self._config["startle_a_mid_th"],
                self._config["startle_buffet_length"])
            # speed bump
            sb_times, sb_severity = get_speed_bump_times(time, accx, gpsspeed)
            # turns
            turn_times, turn_radii = get_turn_times(time, accy, gyrz, gpsspeed,
                                                    turn_time_th=self._config['turn_time_th'])

            # convert to sensor time ns
            brake_times_ns, startle_times_ns, sb_times_ns, turn_times_ns = \
                [sec_to_sensor_ns(t, sensor_offset_ns) for t in [brake_times, startle_times, sb_times, turn_times]]

            # Process for video-based algos
            tg_times_ns, tg_min_distance = get_tailgating_times(event_data,
                                                                score_th=self._config['tg_score_th'],
                                                                duration_th=self._config['tg_duration_th'])
            distraction_times_ns, \
            distraction_max_score = get_distraction_times(event_data,
                                                          score_th=self._config['distraction_score_th'],
                                                          duration_th=self._config['distraction_duration_th'])
            holding_object_times_ns, \
            holding_object_max_score = get_holding_object_times(event_data,
                                                                score_th=self._config['holding_object_score_th'],
                                                                duration_th=self._config['holding_object_duration_th'])
            no_face_times_ns, \
            no_face_max_score = get_no_face_times(event_data,
                                                  score_th=self._config['no_face_score_th'],
                                                  duration_th=self._config['no_face_duration_th'])

            # Make judgement on "coachable"
            # for v0 model, we only use tailgating, distraction, startle-braking
            coachable_times_ns, \
            coachable_severity = get_coachable_times(tg_times_ns,
                                                     tg_min_distance,
                                                     distraction_times_ns,
                                                     distraction_max_score,
                                                     startle_times_ns,
                                                     startle_severity,
                                                     time_delta_ns=int(self._config['seq_time_delta_th'] * 1e9))

            # Package into output
            if len(coachable_times_ns) == 0:
                pass
            else:
                summary = 'TRUE'
                score = max(min(float(max(coachable_severity)), 1.), 0.)  # todo properly map score to the interval 0..1
                confidence = 100  # for now, set confidence as 100, TODO: to update the model to calculate confidence
            raw_output['sensor_offset_ns'] = sensor_offset_ns
            raw_output['coachable_times_ns'] = coachable_times_ns
            raw_output['coachable_severity'] = coachable_severity
            raw_output['brake_times_ns'] = brake_times_ns
            raw_output['brake_severity'] = brake_severity
            raw_output['startle_times_ns'] = startle_times_ns
            raw_output['startle_severity'] = startle_severity
            raw_output['speed_bump_times_ns'] = sb_times_ns
            raw_output['speed_bump_severity'] = sb_severity
            raw_output['turn_times_ns'] = turn_times_ns
            raw_output['turn_radii'] = turn_radii
            raw_output['tailgating_times_ns'] = tg_times_ns
            raw_output['tailgating_min_dist'] = tg_min_distance
            raw_output['looking_away_times_ns'] = distraction_times_ns
            raw_output['looking_away_max_score'] = distraction_max_score
            raw_output['holding_object_times_ns'] = holding_object_times_ns
            raw_output['holding_object_max_score'] = holding_object_max_score
            raw_output['no_face_times_ns'] = no_face_times_ns
            raw_output['no_face_max_score'] = no_face_max_score

            self._logger.info(f'CoachableModel, message_id: {message_id}, summary: {summary}')

        return ModelResponse(summary, score, confidence, raw_output)
