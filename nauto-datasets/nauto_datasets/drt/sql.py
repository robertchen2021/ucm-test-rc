import numpy as np

from datetime import datetime
from typing import Optional


def get_time_constraints_where_clause(
        message_id_column: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None) -> str:

    if since is not None:
        since_ns = np.uint64(since.timestamp() * 1e9)
    else:
        since_ns = None

    if until is not None:
        until_ns = np.int64(until.timestamp() * 1e9)
    else:
        until_ns = None

    if (since_ns, until_ns) == (None, None):
        return ''
    elif since_ns is not None and until_ns is not None:
        return f"AND {message_id_column} BETWEEN {since_ns} and {until_ns}"
    elif since_ns is not None:
        return f"AND {message_id_column} > {since_ns}"
    else:
        return f"AND {message_id_column} <= {until_ns}"


def wrap_query_for_jdbc(query: str, table_name: str) -> str:
    return f"""(
        {query}
    ) as {table_name}"""
