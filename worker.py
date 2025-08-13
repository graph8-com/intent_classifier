#!/usr/bin/env python3
"""Background worker that classifies pending ClickHouse rows.

Behavior
========
- Runs with no CLI arguments
- Repeatedly queries ClickHouse for records where topic_ids is NULL or empty
- Processes records in priority-desc order (then created_at asc for stability)
- Classifies text using the existing model/topic logic
- Updates the ClickHouse table with topic_ids and topic_resolved_at
- If no work is found, sleeps for COOLDOWN seconds (default 60) and retries

Configuration via environment variables (dotenv supported):
- CLICKHOUSE_JDBC_URL: full jdbc:clickhouse://... URL
- CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
- BATCH_SIZE: number of rows to process per iteration (default: 16)
- TOPK: number of topics to assign (default: 3)
- TOPICS_PATH: path to topics.csv (default: topics.csv)
- COOLDOWN_SEC: sleep duration when no work found (default: 60)
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import random

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Local imports from existing modules
from main import (  # type: ignore
    load_topics,
    build_snippet,
    embed_snippets_cached,
    topk_topics,
    SentenceTransformer,
)
from clickhouse_check import parse_clickhouse_jdbc_url  # type: ignore
from loguru import logger

LOG_FILE = os.getenv("WORKER_LOG_FILE")
if LOG_FILE:
    logger.add(LOG_FILE, rotation="50 MB", retention="14 days", level=os.getenv("LOG_LEVEL", "INFO"))


# ----------------------------------------------------------------------------
# ClickHouse HTTP client
# ----------------------------------------------------------------------------


class ClickHouseHttpClient:
    def __init__(self, jdbc_url: str, user: str, password: str, timeout: float = 15.0) -> None:
        self.conf = parse_clickhouse_jdbc_url(jdbc_url)
        self.scheme = "https" if self.conf["ssl"] else "http"
        self.base_url = f"{self.scheme}://{self.conf['host']}:{self.conf['port']}/"
        self.database = self.conf["database"]
        self.timeout = timeout
        # Retry configuration
        self.max_retries = int(os.getenv("CH_MAX_RETRIES", "3"))
        self.backoff_base = float(os.getenv("CH_BACKOFF_BASE", "0.5"))  # seconds
        self.backoff_max = float(os.getenv("CH_BACKOFF_MAX", "8"))  # seconds
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        self.headers_json = {
            "Authorization": f"Basic {token}",
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/json",
        }

    def _endpoint(self) -> str:
        return f"{self.base_url}?database={urllib.parse.quote(self.database)}"

    def query_json(self, sql: str) -> Dict[str, Any]:
        q = sql.strip().rstrip(";") + " FORMAT JSON"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url=self._endpoint(), data=q.encode("utf-8"), method="POST", headers=self.headers_json
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read()
                    return json.loads(body.decode("utf-8"))
            except urllib.error.HTTPError as e:
                # Do not retry on client errors except 429
                status = getattr(e, "code", 0)
                try:
                    err_body = e.read().decode("utf-8", "ignore")
                except Exception:
                    err_body = "<no body>"
                last_err = RuntimeError(
                    f"ClickHouse query failed: HTTP {status} | SQL: {sql}\n{err_body}"
                )
                if status and 400 <= status < 500 and status != 429:
                    break
            except urllib.error.URLError as e:
                last_err = e
            # Backoff before retry
            sleep_s = min(self.backoff_base * (2 ** attempt) + random.uniform(0, 0.25), self.backoff_max)
            logger.warning("query_json retry {} after error: {}", attempt + 1, last_err)
            time.sleep(sleep_s)
        assert last_err is not None
        raise last_err

    def execute(self, sql: str) -> None:
        q = sql.strip().rstrip(";")
        headers = dict(self.headers_json)
        headers["Accept"] = "text/plain"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url=self._endpoint(), data=q.encode("utf-8"), method="POST", headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout):
                    return None
            except urllib.error.HTTPError as e:
                status = getattr(e, "code", 0)
                try:
                    err_body = e.read().decode("utf-8", "ignore")
                except Exception:
                    err_body = "<no body>"
                last_err = RuntimeError(
                    f"ClickHouse execute failed: HTTP {status} | SQL: {sql}\n{err_body}"
                )
                if status and 400 <= status < 500 and status != 429:
                    break
            except urllib.error.URLError as e:
                last_err = e
            # Backoff before retry
            sleep_s = min(self.backoff_base * (2 ** attempt) + random.uniform(0, 0.25), self.backoff_max)
            logger.warning("execute retry {} after error: {}", attempt + 1, last_err)
            time.sleep(sleep_s)
        assert last_err is not None
        raise last_err


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


def ch_escape_string(value: str) -> str:
    """Escape a Python string for ClickHouse single-quoted literals.

    We escape backslashes first, then single quotes by doubling them.
    """
    return value.replace("\\", "\\\\").replace("'", "''")


def build_update_sql(
    table: str,
    row_id: str,
    topic_values: Sequence[str],
    id_type: str,
    topic_ids_inner_type: str,
) -> str:
    # Build topic_ids array literal per inner type
    inner = topic_ids_inner_type.upper()
    if inner == "UUID":
        values = ", ".join([f"toUUID('{ch_escape_string(v)}')" for v in topic_values])
    elif inner in ("INT32", "INT64", "UINT32", "UINT64"):
        values = ", ".join([str(int(v)) for v in topic_values])
    else:
        values = ", ".join([f"'{ch_escape_string(v)}'" for v in topic_values])

    id_t = id_type.upper()
    if id_t == "UUID":
        id_expr = f"toUUID('{ch_escape_string(row_id)}')"
    elif id_t in ("INT32", "INT64", "UINT32", "UINT64"):
        id_expr = str(int(row_id))
    else:
        id_expr = f"'{ch_escape_string(row_id)}'"

    return (
        f"ALTER TABLE {table} UPDATE topic_ids = [{values}], "
        f"topic_resolved_at = now() WHERE id = {id_expr}"
    )


def parse_db_table(qualified: str, default_db: str) -> Tuple[str, str]:
    if "." in qualified:
        db, tbl = qualified.split(".", 1)
        return db, tbl
    return default_db, qualified


def inspect_table_types(ch: ClickHouseHttpClient, qualified_table: str) -> Tuple[str, str]:
    db, tbl = parse_db_table(qualified_table, ch.database)
    sql = (
        "SELECT name, type FROM system.columns "
        f"WHERE database = '{ch_escape_string(db)}' AND table = '{ch_escape_string(tbl)}'"
    )
    result = ch.query_json(sql)
    id_type = "String"
    topic_ids_inner_type = "String"
    for row in result.get("data", []) or []:
        name = str(row.get("name"))
        type_str = str(row.get("type", "")).upper()
        if name == "id":
            if "UUID" in type_str and "ARRAY" not in type_str:
                id_type = "UUID"
            elif "INT32" in type_str:
                id_type = "Int32"
            elif "INT64" in type_str:
                id_type = "Int64"
            elif "UINT32" in type_str:
                id_type = "UInt32"
            elif "UINT64" in type_str:
                id_type = "UInt64"
            else:
                id_type = "String"
        if name == "topic_ids":
            # Expect format like Array(Int32) or Array(UUID)
            if "ARRAY(" in type_str:
                inner = type_str[type_str.find("(") + 1 : type_str.rfind(")")]
            else:
                inner = type_str
            if "UUID" in inner:
                topic_ids_inner_type = "UUID"
            elif "INT32" in inner:
                topic_ids_inner_type = "Int32"
            elif "INT64" in inner:
                topic_ids_inner_type = "Int64"
            elif "UINT32" in inner:
                topic_ids_inner_type = "UInt32"
            elif "UINT64" in inner:
                topic_ids_inner_type = "UInt64"
            else:
                topic_ids_inner_type = "String"
    return id_type, topic_ids_inner_type


def fetch_pending(
    ch: ClickHouseHttpClient,
    table: str,
    limit: int,
) -> List[Dict[str, Any]]:
    sql = f"""
    SELECT id, created_at, priority, url, content
    FROM {table}
    WHERE (topic_ids IS NULL OR length(topic_ids) = 0)
      AND (content IS NOT NULL OR url IS NOT NULL)
      AND http_status = 200
    ORDER BY priority DESC, created_at ASC
    LIMIT {limit}
    """
    result = ch.query_json(sql)
    data = result.get("data")
    if isinstance(data, list):
        return data
    return []


def select_text_source(row: Dict[str, Any]) -> Optional[str]:
    """Build input text including URL plus content when available.

    Prioritize including both to improve context. Falls back to whichever exists.
    """
    url = row.get("url")
    content = row.get("content")
    parts: List[str] = []
    if isinstance(url, str) and url.strip():
        parts.append(f"{url.strip()}")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    if parts:
        return "\n\n".join(parts)
    return None


def process_batch(
    rows: List[Dict[str, Any]],
    model: SentenceTransformer,
    topic_names: List[str],
    topic_mat,
    topk: int,
) -> List[Tuple[str, List[str]]]:
    # Prepare texts
    texts: List[str] = []
    row_ids: List[str] = []
    for r in rows:
        src = select_text_source(r)
        if src is None:
            continue
        texts.append(build_snippet(src))
        row_ids.append(str(r.get("id")))

    if not texts:
        return []

    # Embed and compute top-k
    logger.info("Starting embedding process for {} texts...", len(texts))
    start_time = time.time()
    vecs = embed_snippets_cached(model, texts)
    end_time = time.time()
    logger.info("Completed embedding process in {:.2f} seconds.", end_time - start_time)

    updates: List[Tuple[str, List[str]]] = []
    for idx, vec in enumerate(vecs):
        matches = topk_topics(vec, topic_mat, topic_names, topk)
        topics_only = [t for t, _ in matches]
        updates.append((row_ids[idx], topics_only))
    return updates


def build_topic_name_to_id_map(topics_path: str | os.PathLike[str]) -> Dict[str, str]:
    df = pd.read_csv(topics_path)
    if "topic_id" in df.columns and "topic_name" in df.columns:
        mapping = {}
        for _, r in df.iterrows():
            name = str(r["topic_name"]) if not pd.isna(r["topic_name"]) else None
            tid = r["topic_id"]
            if name is None or pd.isna(tid):
                continue
            mapping[name] = str(int(tid))
        return mapping
    # Fallback empty mapping
    return {}


def _load_model() -> SentenceTransformer:
    try:
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    # MODEL_NAME is defined in main.py and used by SentenceTransformer there
    return SentenceTransformer("intfloat/multilingual-e5-large", device=device)


def main() -> None:
    jdbc_url = os.getenv("CLICKHOUSE_JDBC_URL")
    user = os.getenv("CLICKHOUSE_USER")
    password = os.getenv("CLICKHOUSE_PASSWORD")
    if not jdbc_url or not user or not password:
        logger.error("CLICKHOUSE_JDBC_URL, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD must be set")
        raise SystemExit(2)

    table = os.getenv("CLICKHOUSE_TABLE", "intent.url_scraping_queue")
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    topk = int(os.getenv("TOPK", "3"))
    topics_path = os.getenv("TOPICS_PATH", "topics.csv")
    cooldown_sec = int(os.getenv("COOLDOWN_SEC", "60"))

    logger.info("Loading model and topics…")
    model = _load_model()
    topic_names, topic_mat = load_topics(path=topics_path, model=model)
    name_to_id = build_topic_name_to_id_map(topics_path)
    logger.info("Worker ready. Monitoring ClickHouse for pending work…")

    ch = ClickHouseHttpClient(jdbc_url=jdbc_url, user=user, password=password)

    try:
        id_type, topic_ids_inner_type = inspect_table_types(ch, table)
        logger.info(
            "Schema detected: id_type={}, topic_ids_inner_type={}",
            id_type,
            topic_ids_inner_type,
        )
    except Exception as exc:
        logger.warning("Could not inspect table types ({}). Assuming String types.", exc)
        id_type = "String"
        topic_ids_inner_type = "String"

    while True:
        try:
            rows = fetch_pending(ch, table=table, limit=batch_size)
        except Exception as exc:
            logger.error(f"Failed to fetch pending rows: {exc}")
            time.sleep(min(10, cooldown_sec))
            continue

        if not rows:
            logger.info(f"No pending rows. Sleeping {cooldown_sec}s…")
            time.sleep(cooldown_sec)
            continue

        logger.info("Processing {} rows…", len(rows))
        try:
            updates = process_batch(rows, model, topic_names, topic_mat, topk)
        except Exception as exc:
            logger.error("Batch processing failed: {}", exc)
            # Cooldown briefly to avoid hot loop on repeated failure
            time.sleep(min(10, cooldown_sec))
            continue

        applied = 0
        for row_id, topics in updates:
            try:
                # Convert topic names to proper value strings based on column type
                if topic_ids_inner_type.upper() in ("INT32", "INT64", "UINT32", "UINT64"):
                    topic_values = [name_to_id.get(t) for t in topics]
                    topic_values = [v for v in topic_values if v is not None]
                    if not topic_values:
                        logger.warning("No topic IDs found for row {} (names={})", row_id, topics)
                        continue
                else:
                    topic_values = topics

                sql = build_update_sql(
                    table=table,
                    row_id=row_id,
                    topic_values=topic_values,
                    id_type=id_type,
                    topic_ids_inner_type=topic_ids_inner_type,
                )
                ch.execute(sql)
                applied += 1
            except Exception as exc:
                logger.error("Update failed for id={}: {}", row_id, exc)

        logger.info("Applied updates: {}/{}", applied, len(updates))


if __name__ == "__main__":
    main()


