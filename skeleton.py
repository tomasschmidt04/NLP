# simple_pipeline.py
# ------------------------------------------------------------
# 1) Build a tiny English dataset (20 texts)
# 2) Extract features: avg word length + readability scores
# 3) Train a small RandomForestRegressor (predict FK grade)
# ------------------------------------------------------------

import re

import numpy as np



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import textstat

EXAMPLES = [
    "Press the reset button for ten seconds, then release it.",
    "Use a strong password with at least twelve characters and a mix of symbols.",
    "I didn’t fully understand—could you clarify which model you are using?",
    "It might depend on your configuration; results can vary between versions.",
    "Install the package, restart the application, and check the logs for errors.",
    "Please specify the operating system, Python version, and package manager you prefer.",
    "The capital of France is Paris.",
    "A quick way to reduce overfitting is to add regularization and early stopping.",
    "Could you provide a minimal reproducible example so we can diagnose the issue?",
    "Set the learning rate to 3e-4 and run the training script for 10 epochs.",
    "If the dataset is imbalanced, consider class weights or a different sampling strategy.",
    "First, back up your configuration files; second, update; finally, validate the service.",
    "Your description suggests a dependency conflict; check the resolver and pin versions.",
    "For reproducibility, set explicit seeds and document the environment details.",
    "The function fails because the input schema mismatches the expected field types.",
    "In summary, verify paths, environment variables, and permissions before retrying.",
    "Try a smaller batch size if you encounter out-of-memory errors on the GPU.",
    "What exactly do you want to optimize—speed, accuracy, or memory usage?",
    "A lightweight baseline can reveal whether complex architectures are necessary at all.",
    "Document assumptions, constraints, and failure modes to make maintenance easier."
]

WORD_RE = re.compile(r"\b\w+\b")


def avg_word_length(text: str) -> float:
    words = WORD_RE.findall(text)
    return float(np.mean([len(w) for w in words])) if words else 0.0

def extract_features(text: str) -> dict:
    return {
        "avg_word_len": avg_word_length(text),
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
        "automated_readability_index": float(textstat.automated_readability_index(text)),
        # Target we want to predict (NOT included as a feature):
        "fk_grade": float(textstat.automated_readability_index(text)),
    }

def main():
    # Build dataframe of features
    rows = [extract_features(t) for t in EXAMPLES]
    df = pd.DataFrame(rows)
    # (Optional) inspect
    print("Feature preview:")
    print(df.head(), "\n")

    # Features (X) and target (y)
    X = df[["avg_word_len", "flesch_reading_ease", "automated_readability_index"]].values
    y = df["fk_grade"].values

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Small Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)

    # Evaluate
    y_pred = rf.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    print(f"R^2:  {r2:.3f}")
    print(f"MAE:  {mae:.3f}")

    # Feature importances
    importances = rf.feature_importances_
    cols = ["avg_word_len", "flesch_reading_ease", "automated_readability_index"]
    print("\nFeature importances:")
    for c, imp in sorted(zip(cols, importances), key=lambda x: -x[1]):
        print(f"  {c:28s} {imp:.3f}")

# --- ADD THIS TO skeleton.py (at the end of the file) ---

import argparse
import joblib

# Columns we will use as features (match your extract_features)
FEATURE_COLUMNS = ["avg_word_len", "flesch_reading_ease", "automated_readability_index"]

def train_model_full() -> RandomForestRegressor:
    """
    Entrena el RF sobre TODO el set EXAMPLES (sin split) para producir el modelo final.
    Esto se usa SOLO cuando querés guardar el modelo a disco (una vez).
    """
    rows = [extract_features(t) for t in EXAMPLES]
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLUMNS].values
    y = df["fk_grade"].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    return rf

def save_model(model_path: str = "fk_rf.joblib") -> None:
    """Guarda el RF y el orden de columnas en disco."""
    rf = train_model_full()
    joblib.dump({"model": rf, "feature_cols": FEATURE_COLUMNS}, model_path)

def load_model(model_path: str = "fk_rf.joblib"):
    """Carga el modelo y columnas de features desde disco."""
    pkg = joblib.load(model_path)
    model = pkg["model"]
    feature_cols = pkg.get("feature_cols", FEATURE_COLUMNS)
    return model, feature_cols

def predict_cognitive_load(text: str, model=None, feature_cols=None, model_path: str = "fk_rf.joblib") -> float:
    """
    Predice el FK-grade (proxy de Cognitive Load) para un texto.
    Si no se pasa model/feature_cols, carga desde model_path.
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model(model_path)
    feats = extract_features(text)
    x = np.array([[feats[c] for c in feature_cols]], dtype=float)
    return float(model.predict(x)[0])

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-model", type=str, default=None, help="Ruta para guardar el modelo entrenado (fk_rf.joblib)")
    parser.add_argument("--eval", action="store_true", help="(Opcional) correr la evaluación original con train/test split")
    args = parser.parse_args()

    if args.save_model:
        save_model(args.save_model)
        print(f"Modelo guardado en {args.save_model}")
    elif args.eval:
        # corre tu main() original para ver R^2/MAE/feature importances si querés
        main()
    else:
        # Si se ejecuta sin args, mostrá ayuda
        parser.print_help()

    
