from pathlib import Path
import datetime

#TODO change constant to support bucket read from any environment
S3_ROOT = Path("s3://nauto-cloud-models-test-us/accountant")
ALL_MODELS = ["distraction_detector", "coachable_event_detector", "no-motion-detector", "crashnet_v14_ml", "seat_belt_ensemble"]
HISTORY_START_DATE = datetime.date(year=2020, month=3, day=5)
