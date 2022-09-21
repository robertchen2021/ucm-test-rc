from datetime import datetime
from typing import List, Optional

from pyspark.sql import types as dtype

from nauto_datasets.drt import events
from nauto_datasets.drt import sql as drt_sql
from nauto_datasets.drt.types import EventType, MediaType


def get_ids_column(
        media_type: MediaType, required: bool = True) -> dtype.StructField:

    return dtype.StructField(
        f'{media_type.name.lower()}_message_ids',
        dtype.ArrayType(
            elementType=dtype.LongType(), containsNull=not required),
        not required)


def get_paths_column(
        media_type: MediaType, required: bool = True) -> dtype.StructField:
    return dtype.StructField(
        f'{media_type.name.lower()}_paths',
        dtype.ArrayType(
            elementType=dtype.StringType(), containsNull=not required),
        not required)


_hive_media_sql = """
    SELECT
        ms.{event_id_name} as {event_id_name}
        {higher_order_transforms}
    FROM (
        SELECT
            m.event_id as {event_id_name},
            {media_aggregates}
            FROM (
                SELECT
                    e.id AS event_id,
                    m.type AS media_type,
                    m.message_id AS media_message_id,
                    m.s3_key AS media_path
                FROM events e INNER JOIN media m on m.event_id = e.id
                WHERE
                    e.preferred_judgement_id IS NOT NULL
                    AND NOT e.confirmed_duplicate
                    AND m.type in ({media_types})
                    {date_constraints}
            ) as m
        GROUP BY m.event_id
        {not_empty_aggregates}
        ) ms
"""

_hive_higher_order_array_tranform_sql = """
        transform(ms.{media_array_name}, x -> x.media_message_id) as {media_message_ids_name},
        transform(ms.{media_array_name}, x -> x.media_path) as {media_paths_name}
"""

_hive_media_type_aggregate_sql = """
       array_sort(
           collect_list(
                CASE WHEN m.media_type = '{media_type}' THEN
                    struct(m.media_message_id, m.media_path)
                ELSE
                    NULL
                END
            )
       ) AS {media_array_name}
"""

_hive_not_empty_aggregate_sql = """
         size({array_agg_name}) >= 0
"""


def hive_sql_query(media_ts: List[MediaType],
                   ignore_missing_media: bool,
                   since: Optional[datetime] = None,
                   until: Optional[datetime] = None) -> str:
    if not media_ts:
        raise ValueError('Empty media types')
    required = not ignore_missing_media
    ids_columns = {
        mt: get_ids_column(mt, required) for mt in media_ts
    }
    paths_columns = {
        mt: get_paths_column(mt, required) for mt in media_ts
    }
    media_arrays_columns = {
        mt: f'{mt.name.lower()}_arrays' for mt in media_ts
    }

    aggregates_sql = '   ,\n'.join(
        [
            _hive_media_type_aggregate_sql.format(
                media_type=mt.value,
                media_array_name=media_arrays_columns[mt])
            for mt in media_ts
        ]
    )
    if required:
        not_null_aggregates = 'HAVING ' + '        AND \n'.join([
            _hive_not_empty_aggregate_sql.format(
                array_agg_name=media_arrays_columns[mt])
            for mt in media_ts
        ])
    else:
        not_null_aggregates = ''
    media_types_str = ', '.join([f"'{mt.value}'" for mt in media_ts])

    higher_order_transforms = ', ' + '    ,\n'.join(
        [
            _hive_higher_order_array_tranform_sql.format(
                media_array_name=media_arrays_columns[mt],
                media_message_ids_name=ids_columns[mt].name,
                media_paths_name=paths_columns[mt].name)
            for mt in media_ts
        ]
    )
    return _hive_media_sql.format(
        event_id_name=events.EventColumns.ID.name,
        media_aggregates=aggregates_sql,
        media_types=media_types_str,
        date_constraints=drt_sql.get_time_constraints_where_clause(
            'e.message_id', since, until),
        higher_order_transforms=higher_order_transforms,
        not_empty_aggregates=not_null_aggregates
    )
