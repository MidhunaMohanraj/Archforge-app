"""
Ground stage: retrieval over the team's own reference documents.

This is a plain TF-IDF index with no external dependencies, so the demo
runs anywhere without installing an embeddings model. A production
deployment would swap this for a vector index (e.g. FAISS or pgvector)
over embeddings of the same documents - the interface below
(`RetrievalIndex.search`) would not need to change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import RetrievalConfig

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    source: str
    text: str


class RetrievalIndex:
    """
    Splits every document under `docs_dir` into paragraph-sized chunks and
    builds a TF-IDF index over them. Call `.search(query)` to get the
    top-k most relevant chunks, each tagged with its source file so
    answers can cite where a fact came from.
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.chunks: list[Chunk] = []
        self._doc_freq: Counter = Counter()
        self._chunk_vectors: list[Counter] = []
        self._build()

    def _build(self) -> None:
        docs_dir = Path(self.config.docs_dir)
        if not docs_dir.exists():
            return

        for path in sorted(docs_dir.glob("**/*")):
            if not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            for paragraph in self._split_paragraphs(text):
                self.chunks.append(Chunk(source=path.name, text=paragraph))

        for chunk in self.chunks:
            tf = Counter(tokenize(chunk.text))
            self._chunk_vectors.append(tf)
            for term in tf:
                self._doc_freq[term] += 1

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return parts if parts else [text.strip()]

    def _idf(self, term: str) -> float:
        n = max(len(self.chunks), 1)
        df = self._doc_freq.get(term, 0)
        return math.log((n + 1) / (df + 1)) + 1.0

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        query_tf = Counter(tokenize(query))
        query_vec = {term: freq * self._idf(term) for term, freq in query_tf.items()}
        query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1.0

        scored: list[tuple[Chunk, float]] = []
        for chunk, tf in zip(self.chunks, self._chunk_vectors):
            chunk_vec = {term: freq * self._idf(term) for term, freq in tf.items()}
            chunk_norm = math.sqrt(sum(v * v for v in chunk_vec.values())) or 1.0

            dot = sum(query_vec.get(term, 0) * weight for term, weight in chunk_vec.items())
            score = dot / (query_norm * chunk_norm)
            if score >= self.config.min_score:
                scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: self.config.top_k]
