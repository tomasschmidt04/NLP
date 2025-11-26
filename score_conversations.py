#!/usr/bin/env python3
"""
Score conversations from the B-shape dataset.

Scoring:
  - Per assistant turn: Cognitive Load proxy = len(content)  (Part A scorer)
  - Per conversation:   MEAN over assistant turns
Output:
[
  {
    "id": 1,
    "base_conv": <OAPIJSON>,
    "direct": 1.3,
    "hedging": 2.1,
    "clarify": 4.5,
    "best": "clarify"
  },
  ...
]
"""

import argparse
import json
from typing import Dict, Any, List

# --- Part A scorer (per assistant turn) ---
def score_turn_len(text: str) -> int:
    """Cognitive Load proxy = character length of the assistant message."""
    return len(text or "")

def mean_cognitive_load(conv: Dict[str, Any]) -> float:
    """Mean of assistant-turn scores (avoid bias to longer conversations)."""
    msgs = conv.get("messages", [])
    scores = [score_turn_len(m.get("content", "")) for m in msgs if m.get("role") == "assistant"]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def score_item(item: Dict[str, Any]) -> Dict[str, Any]:
    direct = round(mean_cognitive_load(item["direct"]), 1)
    hedging = round(mean_cognitive_load(item["hedging"]), 1)
    clarify = round(mean_cognitive_load(item["clarify"]), 1)
    best = min([("direct", direct), ("hedging", hedging), ("clarify", clarify)], key=lambda kv: kv[1])[0]
    return {
        "id": item["id"],
        "base_conv": item["base_conv"],
        "direct": direct,
        "hedging": hedging,
        "clarify": clarify,
        "best": best,
    }

def main():
    ap = argparse.ArgumentParser(description="Score B-shape conversations by mean assistant-turn length.")
    ap.add_argument("--infile", type=str, required=True, help="Input JSON (from generate_mock_conversations.py).")
    ap.add_argument("--outfile", type=str, default="scored_dataset_B.json", help="Output JSON with scores.")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results = [score_item(item) for item in data]

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote scored results for {len(results)} items to {args.outfile}")

if __name__ == "__main__":
    main()
