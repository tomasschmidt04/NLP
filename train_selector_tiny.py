#!/usr/bin/env python3
# Always-runs: embeddings + MLP para elegir estrategia con pocas muestras

import argparse
import json
from collections import Counter
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sentence_transformers import SentenceTransformer

LABEL2ID = {"direct": 0, "hedging": 1, "clarify": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def get_first_user_text(base_conv: Dict[str, Any]) -> str:
    for m in base_conv.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def load_texts_and_labels(infile: str) -> Tuple[List[str], List[int]]:
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        best = str(item.get("best", "")).lower().strip()
        if best not in LABEL2ID:
            continue
        text = get_first_user_text(item.get("base_conv", {})).strip()
        if not text:
            continue
        texts.append(text)
        labels.append(LABEL2ID[best])
    if not texts:
        raise ValueError("No hay textos/labels válidos. Revisa el JSON.")
    return texts, labels

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = 3, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def safe_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    """Intenta estratificar; si no se puede, sin estratify; si igual falla, sin validación."""
    if test_size <= 0.0 or len(y) < 4:
        print("⚠️  Dataset muy chico: entreno sin validación.")
        return X, X[:0], y, y[:0]
    counts = Counter(y.tolist())
    min_count = min(counts.values())
    n_classes = len(counts)
    stratify = y if (n_classes > 1 and min_count >= 2) else None
    if stratify is None:
        print(f"⚠️  Desactivo estratify (distribución {dict(counts)}; cada clase necesita ≥ 2).")
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except Exception as e:
        print(f"⚠️  Fallback sin validación por error en split: {e}")
        return X, X[:0], y, y[:0]

def main():
    ap = argparse.ArgumentParser(description="Embeddings + MLP simple (siempre corre con pocos datos).")
    ap.add_argument("--infile", required=True, help="scored_dataset_B.json")
    ap.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Modelo de sentence-transformers (por defecto all-MiniLM-L6-v2)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--test-size", type=float, default=0.2, help="0.0 = sin validación")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) Data
    texts, labels = load_texts_and_labels(args.infile)
    y = np.array(labels, dtype=np.int64)

    # 2) Embeddings
    embedder = SentenceTransformer(args.hf_model)
    X = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # 3) Split robusto
    X_tr, X_va, y_tr, y_va = safe_split(X, y, test_size=args.test_size, seed=args.seed)

    # 4) Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    has_val = len(y_va) > 0
    if has_val:
        val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                               torch.tensor(y_va, dtype=torch.long))
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # 5) Entrenamiento breve
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        msg = f"Epoch {ep:02d} | train_loss={(total_loss/len(train_ds)):.4f}"
        if has_val:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += int((preds == yb).sum().item())
                    total += int(yb.size(0))
            val_acc = correct / max(total, 1)
            msg += f" | val_acc={val_acc:.3f}"
        print(msg)

    # 6) Resultado rápido (preds sobre train o val si existe)
    model.eval()
    with torch.no_grad():
        if has_val and len(X_va) > 0:
            logits = model(torch.tensor(X_va, dtype=torch.float32).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            true = y_va
            print("\nEjemplos (val):")
        else:
            logits = model(torch.tensor(X_tr, dtype=torch.float32).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            true = y_tr
            print("\nEjemplos (train):")

    for i in range(min(5, len(preds))):
        print(f"- pred={ID2LABEL[int(preds[i])]}  true={ID2LABEL[int(true[i])]}")
    print("\nListo 🚀 (con pocos datos, la idea es solo probar la tubería).")

if __name__ == "__main__":
    main()
