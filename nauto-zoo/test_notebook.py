# Databricks notebook source
# MAGIC %sh make local_install

# COMMAND ----------

import json
import os
import unittest
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from unittest.mock import Mock

from nauto_zoo import ModelInput
from nauto_zoo.models.coachable_v2.coachable_model import CoachableModel
from nauto_zoo.preprocess import SensorPreprocessorCombined, SensorCoachablePreprocessor

# COMMAND ----------

pwd = 'tests/models/coachable'
with open(os.path.join(pwd, "risk_BRAKING.json"), 'r') as fh:
    event_message = json.loads(fh.read())

# COMMAND ----------

model_path

# COMMAND ----------

protobuf_dir = '1660ad11b5a2299a'
protobuf_files = ['sensor_0.gz', 'sensor_1.gz', 'sensor_2.gz']
sensor_pb_gzip = [os.path.join(pwd, protobuf_dir, pb) for pb in protobuf_files]
preprocessor = SensorCoachablePreprocessor()
preprocessed_data = preprocessor.preprocess_sensor_files(sensor_pb_gzip)
  
# assemble the model input
model_input = ModelInput(metadata=event_message)
model_input.set('sensor', preprocessed_data)

# load the compiled model
#model_path = os.path.join(pwd, 'coachable_v3_model_0.1.h5')
model_path = os.path.join(pwd, 'coachable_event_multi_label_v1.0.1.h5')

model = load_model(model_path,compile = False)

# run inference using the style similar to the universal_model/model/keras_v2.py
inputs = model_input.get('sensor')
if type(inputs) is not list:
    inputs = [inputs]
raw_output = [
    np.squeeze(model.predict(_input, batch_size=1)).tolist()[1]
    for _input in inputs
]

# test the output
print(raw_output)
print(len(raw_output) ==  1)
print(raw_output[0] >= 0)
print(raw_output[0] <= 1)

# COMMAND ----------



# COMMAND ----------


