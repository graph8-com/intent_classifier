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
import re
import logging
import pathlib
import sys
import time
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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
    # Simple sentence split using regex on punctuation + whitespace
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()
    ]
    ranked = sorted(
        (s for s in sentences if len(s.split()) >= MIN_SENT_TOKENS),
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
# Core workflow (texts only)
# ---------------------------------------------------------------------------


async def classify_texts(texts: List[str], topk: int = 5, topics_path: pathlib.Path | str = "topics.csv"):
    try:
        import torch  # local import to avoid startup overhead if not needed
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    topic_names, topic_mat = load_topics(pathlib.Path(topics_path), model)

    # ── Build snippets ────────────────────────────────────────
    t_snip0 = time.perf_counter()
    snippets = [build_snippet(t) for t in texts]
    print(f"Built snippets in {time.perf_counter() - t_snip0:.2f}s")

    # ── Embed with cache ──────────────────────────────────────
    t_emb0 = time.perf_counter()
    vecs = embed_snippets_cached(model, snippets)
    print(f"Embedding stage took {time.perf_counter() - t_emb0:.2f}s")

    # ── Classify & collect results ────────────────────────────
    t_cls0 = time.perf_counter()
    records = []
    for idx, vec in enumerate(vecs):
        matches = topk_topics(vec, topic_mat, topic_names, topk)
        flat = {f"topic_{i+1}": t for i, (t, _) in enumerate(matches)}
        flat.update({f"score_{i+1}": s for i, (_, s) in enumerate(matches)})
        flat["text_index"] = idx
        records.append(flat)
    print(f"Similarity search for {len(texts)} docs took {time.perf_counter() - t_cls0:.2f}s")

    df = pd.DataFrame(records)
    out_path = "results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# CLI & entry helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str]):
    p = argparse.ArgumentParser(description="Text intent/topic classifier (v1.5)")
    p.add_argument("inputs", nargs="?", help="File with texts OR comma‑separated texts")
    p.add_argument("--topk", type=int, default=5, help="Top‑k topics to save")
    return p.parse_args(argv)


def load_inputs(inp: Optional[str]) -> List[str]:
    if inp is None:
        return globals().get("texts", ["example input text"])
    path = pathlib.Path(inp)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [x.strip() for x in inp.split(",") if x.strip()]


async def _amain():
    args = parse_args(sys.argv[1:])
    texts_list = load_inputs(args.inputs)
    start = time.perf_counter()
    df = await classify_texts(texts_list, topk=args.topk)
    print(f"TOTAL wall time: {time.perf_counter() - start:.2f}s")
    try:
        print(df.to_string(index=False))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Top‑level await for Colab / Jupyter
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(_amain())
