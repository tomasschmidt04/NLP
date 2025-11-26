#!/usr/bin/env python3
"""
Generate synthetic conversations in B-shape (mock only; no API).
Output example item:
{
  "id": 1,
  "base_conv": <OAPIJSON>,
  "direct": <OAPIJSON>,
  "hedging": <OAPIJSON>,
  "clarify": <OAPIJSON>
}
"""

import argparse
import json
import random
from typing import Dict, List, Any

def make_conv(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"messages": messages}

MOCK_SEEDS = [
    {
        "base_user": "What is the capital of France?",
        "direct": [
            {"role": "assistant", "content": "Paris."}
        ],
        "hedging": [
            {"role": "assistant", "content": "I’m fairly sure it’s Paris, if memory serves."},
            {"role": "user", "content": "So… Paris, right?"},
            {"role": "assistant", "content": "Yes—Paris is the capital."}
        ],
        "clarify": [
            {"role": "assistant", "content": "Do you mean the political capital or a cultural center?"},
            {"role": "user", "content": "Political."},
            {"role": "assistant", "content": "Got it—politically, the capital is Paris."}
        ],
    },
    {
        "base_user": "How do I reset my home router?",
        "direct": [
            {"role": "assistant", "content": "Press and hold the reset button for 10–15 seconds until the lights flash."}
        ],
        "hedging": [
            {"role": "assistant", "content": "It can vary by model, but generally you hold the recessed reset button for about 10–15 seconds."}
        ],
        "clarify": [
            {"role": "assistant", "content": "Which brand/model is your router? Steps differ by vendor and firmware."},
            {"role": "user", "content": "It’s a TP-Link Archer C6."},
            {"role": "assistant", "content": "For that model, hold reset/WPS ~10 seconds until LEDs blink, then wait for reboot."},
            {"role": "user", "content": "Will that erase my settings?"},
            {"role": "assistant", "content": "Yes, a factory reset clears settings. For a soft restart, power-cycle instead."}
        ],
    },
    {
        "base_user": "Summarize the plot of The Matrix in two sentences.",
        "direct": [
            {"role": "assistant", "content": "A hacker learns reality is a simulation run by machines and joins rebels. He embraces his role as The One and leads humanity’s resistance."}
        ],
        "hedging": [
            {"role": "assistant", "content": "If I recall, a hacker discovers reality is a simulation and joins rebels. I might miss nuances, but he becomes a prophesied hero against the machines."}
        ],
        "clarify": [
            {"role": "assistant", "content": "Do you want spoilers or a high-level teaser?"},
            {"role": "user", "content": "Full spoilers are fine."},
            {"role": "assistant", "content": "Neo learns reality is a machine-made simulation and joins rebels to free humanity. He accepts being The One and leads the fight."}
        ],
    },
]

def make_item(item_id: int, seed: Dict[str, Any]) -> Dict[str, Any]:
    base_conv = make_conv([{"role": "user", "content": seed["base_user"]}])
    direct = make_conv(seed["direct"])
    hedging = make_conv(seed["hedging"])
    clarify = make_conv(seed["clarify"])
    return {
        "id": item_id,
        "base_conv": base_conv,
        "direct": direct,
        "hedging": hedging,
        "clarify": clarify,
    }

def main():
    ap = argparse.ArgumentParser(description="Generate mock conversations (B-shape).")
    ap.add_argument("--out", type=str, default="mock_dataset_B.json", help="Output JSON path.")
    ap.add_argument("--n", type=int, default=5, help="How many items to generate.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = ap.parse_args()

    random.seed(args.seed)
    data: List[Dict[str, Any]] = []
    for i in range(1, args.n + 1):
        seed = random.choice(MOCK_SEEDS)
        data.append(make_item(i, seed))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} items to {args.out}")

if __name__ == "__main__":
    main()
