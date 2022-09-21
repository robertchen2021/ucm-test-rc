from collections import defaultdict
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional

from nauto_datasets.drt import events
from nauto_datasets.drt import sql as drt_sql
from nauto_datasets.drt.types import EventType, JudgmentType
from nauto_datasets.protos import drt_pb2
from pyspark.sql import types as dtype


class JudgmentSummaryFilter(NamedTuple):
    values: List[str]
    invert: bool = False

    def to_pb(self) -> int:
        """Serializes `JudgmentSummaryFilter` as protobuf `Message`"""
        return drt_pb2.JudgmentSummaryFilter(
            values=self.values,
            invert=self.invert
        )

    @staticmethod
    def from_pb(conf_pb: drt_pb2.JudgmentSummaryFilter) -> 'JudgmentSummaryFilter':
        """Reads `JudgmentSummaryFilter` from protobuf `Message`"""
        return JudgmentSummaryFilter(
            values=conf_pb.values,
            invert=conf_pb.invert
        )

    def get_where_clause(self, judgment_column_name: str) -> str:
        if self.values is not None and len(self.values) > 0:
            values = ["'{0}'".format(v) for v in self.values]
            summary_filter_values = '({0})'.format(','.join(values))

            summary_filter_where = '{judgment_column_name}.summary {invert} in {values_list}'.format(
                judgment_column_name=judgment_column_name,
                invert='NOT' if self.invert else '',
                values_list=summary_filter_values
            )
        else:
            summary_filter_where = 'true'

        return summary_filter_where


def get_label_column(judgment_type: JudgmentType) -> dtype.StructField:
    return dtype.StructField(
        f'{judgment_type.name.lower()}_label', dtype.BooleanType(), True)


def get_info_column(judgment_type: JudgmentType) -> dtype.StructField:
    return dtype.StructField(
        f'{judgment_type.name.lower()}_info', dtype.StringType(), True)


_main_query_sql = """
    SELECT
        e.id AS {event_id_name},
        e.type as {event_type_name},
        TO_TIMESTAMP(e.message_id / 1e+9) AS {time_name},
        e.received_at AS {received_at_name},
        e.message_id AS {message_id_name},
        e.device_id AS {device_id_name},
        e.fleet_id AS {fleet_id_name},
        e.region AS {region_name},
        {judgment_type_selects}
    FROM events e INNER JOIN {judgment_type_joins}
    WHERE
        e.preferred_judgement_id IS NOT NULL
        AND NOT e.confirmed_duplicate
        {event_type_constraints}
        {date_constraints}
    ORDER BY e.message_id"""


def _add_jt_select_clause(judgment_type: JudgmentType) -> str:
    label_column = get_label_column(judgment_type)
    info_column = get_info_column(judgment_type)
    sub_name = judgment_type.name
    return f"""
    CASE
        WHEN {sub_name}.summary = 'true' THEN TRUE
        WHEN {sub_name}.summary = 'false' THEN FALSE
        ELSE NULL
    END AS {label_column.name},
    {sub_name}.info as {info_column.name}"""


_join_on_clause_sql = """judgements {sub_name} ON e.id = {sub_name}.event_id
        AND {sub_name}.sufficient AND {sub_name}.preferred
        AND {sub_name}.type = '{judgment_type}'
        {summary_constraints}
"""


def hive_sql_query(
        judgment_types: List[JudgmentType],
        judgment_summary_filters: Optional[Dict[JudgmentType, JudgmentSummaryFilter]] = None,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
) -> str:
    if not judgment_types:
        raise ValueError('No judgment types to select')
    judgment_summary_filters = judgment_summary_filters or {}

    summary_constraints = defaultdict(lambda: '')
    for jt in judgment_types:
        if jt in judgment_summary_filters:
            summary_constraints[jt] = 'AND ' + judgment_summary_filters[jt].get_where_clause(jt.name)

    judgment_types = list(set(judgment_types))

    judgment_type_selects = ',\n\t\t'.join(
        _add_jt_select_clause(jt) for jt in judgment_types)

    if event_types:
        values = [f"'{et.value}'" for et in event_types]
        event_type_constraints = f"AND e.type in ({','.join(values)})"
    else:
        event_type_constraints = ''

    judgment_type_joins = 'INNER JOIN '.join(
        _join_on_clause_sql.format(
            sub_name=jt.name,
            judgment_type=jt.value,
            summary_constraints=summary_constraints[jt]
        )
        for jt in judgment_types)

    return _main_query_sql.format(
        # event metadata
        event_id_name=events.EventColumns.ID.name,
        event_type_name=events.EventColumns.TYPE.name,
        time_name=events.EventColumns.TIME.name,
        received_at_name=events.EventColumns.RECEIVED_AT.name,
        message_id_name=events.EventColumns.MESSAGE_ID.name,
        device_id_name=events.EventColumns.DEVICE_ID.name,
        fleet_id_name=events.EventColumns.FLEET_ID.name,
        region_name=events.EventColumns.REGION.name,
        event_type_constraints=event_type_constraints,
        # judgments
        judgment_type_selects=judgment_type_selects,
        judgment_type_joins=judgment_type_joins,
        date_constraints=drt_sql.get_time_constraints_where_clause(
            'e.message_id', since, until))
