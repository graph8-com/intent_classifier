#!/usr/bin/env python3
"""Text intent/topic classifier - v1.5 (time-logging, Colab-friendly)

What's new
==========
* Accepts raw texts as input (one per line or comma-separated); no URL fetching.
* **Detailed timing logs** for snippet-prep, embedding, and total wall time.
* **Colab safe entry point** - keeps the *top-level* `await _amain()` pattern,
  but also works when executed as a normal script (with CLI args).

The rest (40 K token cap, disk cache, CSV output) is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import pathlib
import sys
import time
from typing import Iterable, List, Optional

import blingfire
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from uuid import uuid4

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
CACHE_DIR = pathlib.Path(f"{uuid4()}")  # embedding cache location

MODEL_NAME = "intfloat/multilingual-e5-large"  # 384‑dim multilingual model
MAX_TOKENS_DOC = 40_000  # generous window
MIN_SENT_TOKENS = 5  # skip trivial sentences

# ---------------------------------------------------------------------------
# Topic loading & embedding
# ---------------------------------------------------------------------------


def load_topics(path: pathlib.Path, model: SentenceTransformer):
    df = pd.read_csv(path)
    if "topic_name" not in df.columns:
        raise ValueError("topics.csv must have a 'topic_name' column")

    def full_text(row):
        parts = [str(row["topic_name"])]
        if "category" in row and not pd.isna(row["category"]):
            parts.append(str(row["category"]))
        if "theme" in row and not pd.isna(row["theme"]):
            parts.append(str(row["theme"]))
        return " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else parts[0]

    texts = [full_text(r) for _, r in df.iterrows()]
    names = df["topic_name"].astype(str).tolist()
    print(f"Embedding {len(names)} topics …")
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=512)
    return names, np.asarray(vecs, dtype=np.float32)



# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


def build_snippet(text: str) -> str:
    sentences = blingfire.text_to_sentences(text).split("\n")
    ranked = sorted(
        (s.strip() for s in sentences if len(s.split()) >= MIN_SENT_TOKENS),
        key=lambda s: -sum(ch.isalnum() for ch in s),
    )
    words: List[str] = []
    for sent in ranked:
        words.extend(sent.split())
        if len(words) >= MAX_TOKENS_DOC:
            break
    return " ".join(words[:MAX_TOKENS_DOC])


# ---------------------------------------------------------------------------
# Embedding cache utilities
# ---------------------------------------------------------------------------


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def cache_path(sha: str) -> pathlib.Path:
    return CACHE_DIR / f"{sha}.npy"


def load_cache(sha: str) -> Optional[np.ndarray]:
    p = cache_path(sha)
    if p.exists():
        return np.load(p)
    return None


def save_cache(sha: str, vec: np.ndarray) -> None:
    p = cache_path(sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, vec.astype(np.float32))


def embed_snippets_cached(model: SentenceTransformer, snippets: List[str]) -> np.ndarray:
    hashes = [sha1(s) for s in snippets]
    vecs: List[Optional[np.ndarray]] = [load_cache(h) for h in hashes]
    missing = [i for i, v in enumerate(vecs) if v is None]

    if missing:
        new_vecs = model.encode([snippets[i] for i in missing], normalize_embeddings=True, batch_size=32)
        new_vecs = [np.asarray(v, dtype=np.float32) for v in new_vecs]
        for i, v in zip(missing, new_vecs):
            vecs[i] = v
            save_cache(hashes[i], v)
    print(f"Embedding cache hit: {len(snippets) - len(missing)} / {len(snippets)}")
    return np.stack(vecs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


def topk_topics(vec: np.ndarray, topic_mat: np.ndarray, topic_names: List[str], k: int):
    sims = vec @ topic_mat.T
    if k >= len(topic_names):
        idx = sims.argsort()[::-1]
    else:
        idx = sims.argpartition(-k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
    return [(topic_names[i], float(sims[i])) for i in idx[:k]]


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


async def classify_urls(urls: List[str],texts: List[str], topk: int = 5):
    device = "cuda" if util.torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    topic_names, topic_mat = load_topics(pathlib.Path("topics.csv"), model)

    # ── Fetch pages ────────────────────────────────────────────
    t_fetch0 = time.perf_counter()
    async with httpx.AsyncClient(http2=True, follow_redirects=True) as session:
        sem = asyncio.Semaphore(NUM_CONCURRENT_FETCH)

        async def bounded(u):
            async with sem:
                return await fetch_url(session, u)

        pages = await asyncio.gather(*[bounded(u) for u in urls])
    print(f"Fetched {len(urls)} pages in {time.perf_counter() - t_fetch0:.2f}s ({len(urls) / (time.perf_counter() - t_fetch0):.2f}/s)")

    # ── Build snippets ────────────────────────────────────────
    t_snip0 = time.perf_counter()
    snippets = [f"{url} {page}" if page else url for url, page in zip(urls, pages)]
    print(f"Built snippets in {time.perf_counter() - t_snip0:.2f}s")
    # ── Embed with cache ──────────────────────────────────────
    t_emb0 = time.perf_counter()
    vecs = embed_snippets_cached(model, snippets)
    print(f"Embedding stage took {time.perf_counter() - t_emb0:.2f}s")

    # ── Classify & collect results ────────────────────────────
    t_cls0 = time.perf_counter()
    records = []
    for url, vec in zip(urls, vecs):
        matches = topk_topics(vec, topic_mat, topic_names, topk)
        flat = {f"topic_{i+1}": t for i, (t, _) in enumerate(matches)}
        flat.update({f"score_{i+1}": s for i, (_, s) in enumerate(matches)})
        flat["url"] = url
        records.append(flat)
    print(f"Similarity search for {len(urls)} docs took {time.perf_counter() - t_cls0:.2f}s")

    df = pd.DataFrame(records)
    out_path = "results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# CLI & entry helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str]):
    p = argparse.ArgumentParser(description="URL topic classifier (v1.4)")
    p.add_argument("inputs", nargs="?", help="File with URLs OR comma‑separated URLs")
    p.add_argument("--topk", type=int, default=5, help="Top‑k topics to save")
    return p.parse_args(argv)


def load_inputs(inp: Optional[str]) -> List[str]:
    if inp is None:
        return globals().get("urls", ["https://google.com"])
    path = pathlib.Path(inp)
    if path.exists():
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return [x.strip() for x in inp.split(",") if x.strip()]


async def _amain():
    args = 2
    urls_list = urls
    start = time.perf_counter()
    df = await classify_urls(urls_list, topk=2)
    print(f"TOTAL wall time: {time.perf_counter() - start:.2f}s")
    # show markdown preview in notebooks
    try:
        from IPython.display import Markdown, display

        display(Markdown(df.to_markdown(index=False)))
    except ImportError:
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Top‑level await for Colab / Jupyter
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # In standard Python >=3.11 you could use asyncio.run inside an if/else, but
    # for Colab we keep explicit await so users can re‑run cells.
    try:
        # If already inside an event loop (e.g., Jupyter) use await directly.
        import nest_asyncio  # type: ignore

        nest_asyncio.apply()
        await _amain()  # type: ignore[misc]
    except RuntimeError:
        # Fallback for plain scripts: no running loop → create one.
        asyncio.run(_amain())
