import os
import yaml
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------
# MODEL REGISTRY
# -------------------------
MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "MLPClassifier": MLPClassifier,
    "LGBMClassifier": LGBMClassifier,
    "CatBoostClassifier": CatBoostClassifier,
    "VotingClassifier": VotingClassifier,
    "SVC": SVC
}

# -------------------------
# Load Params
# -------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

models_params = params["models"]

# -------------------------
# Load Data
# -------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

os.makedirs("models", exist_ok=True)

# -------------------------
# Train Models with GridSearchCV
# -------------------------
for model_name, param_grid in models_params.items():
    print(f"\n🔹 Training {model_name} with GridSearchCV...")

    if model_name == "VotingClassifier":
        # VotingClassifier requires sub-estimators
        estimators = []
        if "estimators" in param_grid:
            for est_name, est_params in param_grid["estimators"].items():
                if est_name == "rf":
                    est = RandomForestClassifier(**{k: v[0] for k, v in est_params.items()})
                elif est_name == "xgb":
                    est = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                        **{k: v[0] for k, v in est_params.items()})
                elif est_name == "svc":
                    est = SVC(**{k: v[0] for k, v in est_params.items()})
                estimators.append((est_name, est))

        model = VotingClassifier(estimators=estimators)
        grid = GridSearchCV(
            model,
            {"voting": param_grid["voting"]},
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=2
        )
    else:
        model_class = MODEL_REGISTRY[model_name]
        grid = GridSearchCV(
            model_class(),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=2
        )

    # Fit model on training set
    grid.fit(X_train, y_train)

    print(f"✅ Best {model_name} params: {grid.best_params_}")
    print(f"🏆 Best {model_name} F1-score (CV on train): {grid.best_score_:.4f}")

    # Save trained model with its name
    model_path = os.path.join("models", f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    print(f"💾 Saved {model_name} → {model_path}")