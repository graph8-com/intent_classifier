#!/usr/bin/env python3
"""ClickHouse connectivity check (no CLI args)

Hardcoded JDBC URL (from user). Reads credentials from env:
  - CLICKHOUSE_USER
  - CLICKHOUSE_PASSWORD
Executes a test query and prints the first row as JSON.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

JDBC_URL = os.getenv("CLICKHOUSE_JDBC_URL")
SQL = (
    """
SELECT id, created_at, priority, http_status, url, content, scraped_at, topic_ids, topic_resolved_at, `_version`
FROM intent.url_scraping_queue
LIMIT 1
"""
    .strip()
)


def parse_clickhouse_jdbc_url(jdbc_url: str) -> Dict[str, Any]:
    if not jdbc_url.startswith("jdbc:clickhouse://"):
        raise ValueError("Expected a ClickHouse JDBC URL starting with jdbc:clickhouse://")
    raw = jdbc_url[len("jdbc:clickhouse://") :]
    host_port_db, _, query_str = raw.partition("?")
    host_port, _, database = host_port_db.partition("/")
    host, _, port_str = host_port.partition(":")
    if not host:
        raise ValueError("Missing host in JDBC URL")
    port = int(port_str) if port_str else 8123
    params = urllib.parse.parse_qs(query_str, keep_blank_values=True)
    flat_params = {k: v[0] for k, v in params.items()}
    use_ssl = flat_params.get("ssl", "false").lower() in ("1", "true", "yes") or port == 8443
    return {
        "host": host,
        "port": port,
        "database": database or "default",
        "ssl": use_ssl,
        "params": flat_params,
    }


def run_query(jdbc_url: str, user: str, password: str, sql: str) -> Dict[str, Any]:
    conf = parse_clickhouse_jdbc_url(jdbc_url)
    scheme = "https" if conf["ssl"] else "http"
    base_url = f"{scheme}://{conf['host']}:{conf['port']}/"
    url = f"{base_url}?database={urllib.parse.quote(conf['database'])}"
    # Ensure JSON output
    q = sql.strip().rstrip(";") + " FORMAT JSON"
    data = q.encode("utf-8")

    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "text/plain; charset=utf-8",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url=url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        body = resp.read()
        return json.loads(body.decode("utf-8"))


def main() -> int:
    user = os.getenv("CLICKHOUSE_USER")
    password = os.getenv("CLICKHOUSE_PASSWORD")
    if not user or not password:
        print("Set CLICKHOUSE_USER and CLICKHOUSE_PASSWORD env vars.")
        return 2

    try:
        result = run_query(jdbc_url=JDBC_URL, user=user, password=password, sql=SQL)
    except Exception as exc:
        print(f"Query failed: {exc}")
        return 1

    data = result.get("data")
    if isinstance(data, list) and data:
        print(json.dumps(data[0], ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


