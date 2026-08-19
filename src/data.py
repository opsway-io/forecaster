import clickhouse_connect
import pandas as pd
import os
from typing import Optional

def get_clickhouse_client():
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    database = os.getenv("CLICKHOUSE_DB", "opsway")
    user = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    return clickhouse_connect.get_client(host=host, port=port, username=user, password=password, database=database)

def get_monitor_data(client, monitor_id: int) -> pd.DataFrame:
    query = f"""
    SELECT 
        created_at,
        timing_total / 1000000 AS response_time,
        timing_dns_lookup / 1000000 AS dns_lookup,
        timing_tcp_connection / 1000000 AS tcp_connection,
        timing_tls_handshake / 1000000 AS tls_handshake,
        timing_server_processing / 1000000 AS server_processing,
        timing_content_transfer / 1000000 AS content_transfer,
        status_code
    FROM checks
    WHERE monitor_id = {monitor_id}
    ORDER BY created_at ASC
    """
    return client.query_df(query)
