"""Hybrid retrieval: BM25 (always) + local embeddings (if sentence-transformers is installed).

Results are fused with reciprocal rank fusion. Chunks in the learner's current
section get a small boost so a bare "wait, what?" resolves to nearby text.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

STOP = set("a an the of to in on for and or is are was were be by with as at it its this that these those "
           "from into we our which what how why does do not can".split())
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if f:
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def _load_embedder():
    if os.environ.get("PAPERWORLD_EMBED", "auto").lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(os.environ.get("PAPERWORLD_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))


class Index:
    def __init__(self, chunks: list[dict], store_dir: Path | None = None, use_refs: bool = False):
        self.chunks = [c for c in chunks if use_refs or c["kind"] != "references"]
        self.by_id = {c["id"]: c for c in chunks}
        self.bm25 = BM25([tokenize(self._searchable(c)) for c in self.chunks])
        self.embedder = _load_embedder()
        self.emb = None
        if self.embedder is not None:
            import numpy as np
            cache = (store_dir / "embeddings.npy") if store_dir else None
            if cache and cache.exists():
                arr = np.load(cache)
                if arr.shape[0] == len(self.chunks):
                    self.emb = arr
            if self.emb is None:
                self.emb = self.embedder.encode([self._searchable(c) for c in self.chunks],
                                                normalize_embeddings=True, show_progress_bar=False)
                if cache:
                    np.save(cache, self.emb)

    @staticmethod
    def _searchable(c: dict) -> str:
        return f"{c['section_title']}. {c['label']} {c['text']}".strip()

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + embeddings)" if self.emb is not None else "BM25 only"

    def search(self, query: str, k: int = 6, section: str = "") -> list[dict]:
        ranks: dict[int, float] = {}
        bm = self.bm25.scores(tokenize(query))
        for r, i in enumerate(sorted(range(len(bm)), key=lambda i: -bm[i])[:50]):
            if bm[i] > 0:
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self.emb is not None:
            import numpy as np
            q = self.embedder.encode([EMBED_QUERY_PREFIX + query], normalize_embeddings=True)[0]
            sims = self.emb @ q
            for r, i in enumerate(np.argsort(-sims)[:50]):
                ranks[int(i)] = ranks.get(int(i), 0) + 1 / (60 + r)
        if section:
            for i, c in enumerate(self.chunks):
                sid = c["section_id"]
                if sid and (sid == section or sid.startswith(section + ".") or section.startswith(sid + ".")):
                    ranks[i] = ranks.get(i, 0) + 0.004
        top = sorted(ranks, key=lambda i: -ranks[i])[:k]
        return [dict(self.chunks[i], score=round(ranks[i], 5)) for i in top]
