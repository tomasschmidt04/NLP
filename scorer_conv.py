#!/usr/bin/env python3
# scorer_conv.py
import argparse
import importlib.util
import json
from typing import Any, Dict, List

def load_skeleton(path: str):
    spec = importlib.util.spec_from_file_location("skeleton_mod", path)
    skel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skel)  # type: ignore
    return skel

def mean_cognitive_load(conv: Dict[str, Any], skel, model, feature_cols, model_path: str) -> float:
    msgs = conv.get("messages", [])
    vals = []
    for m in msgs:
        if m.get("role") == "assistant":
            text = m.get("content", "")
            vals.append(skel.predict_cognitive_load(text, model=model, feature_cols=feature_cols, model_path=model_path))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))

def score_item(item: Dict[str, Any], skel, model, feature_cols, model_path: str) -> Dict[str, Any]:
    direct = round(mean_cognitive_load(item["direct"], skel, model, feature_cols, model_path), 2)
    hedging = round(mean_cognitive_load(item["hedging"], skel, model, feature_cols, model_path), 2)
    clarify = round(mean_cognitive_load(item["clarify"], skel, model, feature_cols, model_path), 2)
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
    ap = argparse.ArgumentParser(description="Score conversations using pre-trained model saved by skeleton.py")
    ap.add_argument("--infile", required=True, help="Ruta al JSON B-shape (conversations)")
    ap.add_argument("--outfile", default="scored_dataset_B.json", help="Ruta de salida")
    ap.add_argument("--skeleton-path", default="skeleton.py", help="Ruta a skeleton.py editado")
    ap.add_argument("--model-path", default="fk_rf.joblib", help="Ruta al modelo guardado")
    args = ap.parse_args()

    skel = load_skeleton(args.skeleton_path)
    model, feature_cols = skel.load_model(args.model_path)

    with open(args.infile, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results = [score_item(item, skel, model, feature_cols, args.model_path) for item in data]

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote scored results for {len(results)} items to {args.outfile}")

if __name__ == "__main__":
    main()
