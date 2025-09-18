import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

# Models we expect to evaluate
VALID_MODELS = {
    "LogisticRegression",
    "RandomForestClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "MLPClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "VotingClassifier",
    "SVC"
}

def main():
    # -------------------------
    # Load Test Data
    # -------------------------
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # -------------------------
    # Prepare output folders
    # -------------------------
    os.makedirs("experiments", exist_ok=True)

    results = {}
    metrics_history = []

    # -------------------------
    # Evaluate all models
    # -------------------------
    model_files = [
        f for f in os.listdir("models")
        if f.endswith(".pkl") and f.replace(".pkl", "") in VALID_MODELS
    ]

    if not model_files:
        print("⚠️ No valid models found in 'models/'. Check your train.py output.")
        return

    step = 1
    for model_file in model_files:
        model_name = model_file.replace(".pkl", "")
        print(f"\n🔍 Evaluating {model_name}...")

        # Load model
        model = joblib.load(os.path.join("models", model_file))

        # Special handling for XGBClassifier
        if model_name == "XGBClassifier":
            y_test_fixed = y_test - y_test.min()
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_fixed, y_pred)
            f1 = f1_score(y_test_fixed, y_pred, average="macro")
            if hasattr(model, "predict_proba"):
                loss = log_loss(y_test_fixed, model.predict_proba(X_test))
            else:
                loss = None
            cm = confusion_matrix(y_test_fixed, y_pred).tolist()
        else:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            if hasattr(model, "predict_proba"):
                loss = log_loss(y_test, model.predict_proba(X_test))
            else:
                loss = None
            cm = confusion_matrix(y_test, y_pred).tolist()

        # -------------------------
        # Store metrics
        # -------------------------
        results[model_name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "loss": round(loss, 4) if loss is not None else None,
            "confusion_matrix": cm
        }

        metrics_history.append({
            "step": step,
            "model": model_name,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "loss": round(loss, 4) if loss is not None else None
        })
        step += 1

    # -------------------------
    # Save results
    # -------------------------
    with open("experiments/results.json", "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(results).T.to_csv("experiments/results.csv", index=True)

    # For DVC plots
    with open("metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=4)

    # Best model selection (by F1-score)
    best_model = max(results, key=lambda m: results[m]["f1_macro"])
    best_metrics = results[best_model]

    with open("metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    print(f"\n✅ Evaluation complete. Best model = {best_model}")
    print("📊 Results saved → experiments/results.json, results.csv, metrics.json, metrics_history.json")

if __name__ == "__main__":
    main()