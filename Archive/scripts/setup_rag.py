#!/usr/bin/env python3
"""Load and smoke-test the local TF-IDF knowledge base."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecohome.rag import KB


if __name__ == "__main__":
    KB.load()
    print(f"Loaded {len(KB.chunks)} chunks from {KB.documents_dir}")
    for query in (
        "when should I charge my EV with solar",
        "thermostat pre-cooling peak tariff",
        "dishwasher off-peak savings",
        "pool pump runtime",
    ):
        hits = KB.search(query, k=2)
        print(f"\nQ: {query}")
        for h in hits:
            print(f"  [{h['score']}] {h['source']}: {h['content'][:140].replace(chr(10), ' ')}...")
