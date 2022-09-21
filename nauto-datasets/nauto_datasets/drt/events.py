from datetime import datetime
from typing import List, Optional

from pyspark.sql import types as dtype

from nauto_datasets.drt import sql as drt_sql
from nauto_datasets.drt.types import EventType


class EventColumns:
    ID = dtype.StructField('event_id', dtype.LongType(), False)
    TYPE = dtype.StructField('event_type', dtype.StringType(), False)
    DEVICE_ID = dtype.StructField('device_id', dtype.LongType(), True)
    MESSAGE_ID = dtype.StructField('event_message_id', dtype.LongType(), False)
    TIME = dtype.StructField('event_time', dtype.TimestampType(), False)
    RECEIVED_AT = dtype.StructField('event_received_at', dtype.TimestampType(), False)
    FLEET_ID = dtype.StructField('fleet_id', dtype.StringType(), True)
    REGION = dtype.StructField('aws_region', dtype.StringType(), False)

    @staticmethod
    def all() -> List[dtype.StructField]:
        return [
            cd for name, cd in vars(EventColumns).items()
            if not name.startswith('_') and not callable(cd)
            and name is not 'all'
        ]
