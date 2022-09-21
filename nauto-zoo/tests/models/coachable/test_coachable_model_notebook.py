# Databricks notebook source
"""
Run from console:
$ cd nauto-ai/nauto-zoo
$ python -m pytest tests/models/coachable/test_coachable_model.py --log-cli-level=INFO
"""

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

# MAGIC %sh make local_install

# COMMAND ----------


