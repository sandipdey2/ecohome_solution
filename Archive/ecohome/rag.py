"""Local RAG over the energy-saving knowledge base.

Uses a lightweight TF-IDF + cosine retriever so the advisor works with
no OpenAI embeddings key and no Chroma daemon. Chunks are loaded from
data/documents/*.txt on first use and cached in memory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from ecohome.config import DOCUMENTS_DIR


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    source: str
    title: str
    text: str
    tf: dict[str, float]


class EnergyKnowledgeBase:
    def __init__(self, documents_dir: Path | None = None) -> None:
        self.documents_dir = documents_dir or DOCUMENTS_DIR
        self.chunks: list[Chunk] = []
        self.idf: dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        files = sorted(self.documents_dir.glob("*.txt"))
        raw_chunks: list[tuple[str, str, str]] = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            title = text.strip().splitlines()[0] if text.strip() else path.stem
            for piece in _split_paragraphs(text):
                raw_chunks.append((path.name, title, piece))

        n = len(raw_chunks) or 1
        df: dict[str, int] = {}
        prepared: list[tuple[str, str, str, dict[str, float]]] = []
        for source, title, piece in raw_chunks:
            tokens = _tokenize(piece)
            tf: dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0.0) + 1.0
            if tokens:
                norm = len(tokens)
                tf = {k: v / norm for k, v in tf.items()}
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
            prepared.append((source, title, piece, tf))

        self.idf = {tok: math.log((n + 1) / (df_ + 1)) + 1.0 for tok, df_ in df.items()}
        self.chunks = [
            Chunk(source=s, title=t, text=p, tf=tf) for s, t, p, tf in prepared
        ]
        self._loaded = True

    def search(self, query: str, k: int = 5) -> list[dict]:
        self.load()
        q_tokens = _tokenize(query)
        if not q_tokens or not self.chunks:
            return []
        q_tf: dict[str, float] = {}
        for tok in q_tokens:
            q_tf[tok] = q_tf.get(tok, 0.0) + 1.0
        for tok in list(q_tf):
            q_tf[tok] /= len(q_tokens)

        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            score = _cosine_tfidf(q_tf, chunk.tf, self.idf)
            # light lexical boost for exact device words
            boost = 0.0
            text_l = chunk.text.lower()
            for tok in set(q_tokens):
                if len(tok) > 3 and tok in text_l:
                    boost += 0.03
            scored.append((score + boost, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, chunk) in enumerate(scored[:k], start=1):
            if score <= 0:
                continue
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "source": chunk.source,
                    "title": chunk.title,
                    "content": chunk.text.strip(),
                    "relevance": "high" if rank <= 2 else "medium" if rank <= 4 else "low",
                }
            )
        return results


def _split_paragraphs(text: str, max_len: int = 900) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[str] = []
    buf = ""
    for block in blocks:
        if len(buf) + len(block) + 2 <= max_len:
            buf = f"{buf}\n\n{block}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = block
    if buf:
        chunks.append(buf)
    return chunks or [text]


def _cosine_tfidf(
    q_tf: dict[str, float], d_tf: dict[str, float], idf: dict[str, float]
) -> float:
    dot = 0.0
    qn = 0.0
    dn = 0.0
    for tok, qv in q_tf.items():
        w = qv * idf.get(tok, 0.0)
        qn += w * w
        if tok in d_tf:
            dot += w * (d_tf[tok] * idf.get(tok, 0.0))
    for tok, dv in d_tf.items():
        w = dv * idf.get(tok, 0.0)
        dn += w * w
    if qn <= 0 or dn <= 0:
        return 0.0
    return dot / math.sqrt(qn * dn)


KB = EnergyKnowledgeBase()
