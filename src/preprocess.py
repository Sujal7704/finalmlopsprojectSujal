import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os
import joblib

# -------------------------
# Load params.yaml
# -------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load dataset
data_path = params["dataset"]
df = pd.read_csv(data_path)

# Wine dataset: last column is "quality" (target)
X = df.drop("quality", axis=1)
y = df["quality"]

# -------------------------
# Class Grouping (optional)
# -------------------------
if params["preprocessing"].get("merge_classes", False):
    # Merge rare classes into broader bins (example rule)
    y = y.replace({3: 4, 9: 8})  # merge 3→4, 9→8

# -------------------------
# Feature Engineering
# -------------------------
# Custom interaction features
if "alcohol" in X.columns and "sulphates" in X.columns:
    X["alcohol_sulphates"] = X["alcohol"] * X["sulphates"]

if "pH" in X.columns and "volatile acidity" in X.columns:
    X["pH_volatile_acidity"] = X["pH"] * X["volatile acidity"]

# Polynomial interaction features (optional)
if params["preprocessing"].get("poly_features", False):
    degree = params["preprocessing"].get("poly_degree", 2)
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    X = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# -------------------------
# Scaling
# -------------------------
if params["preprocessing"]["scale"]:
    scaler_type = params["preprocessing"].get("scaler", "StandardScaler")
    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# -------------------------
# PCA (optional dimensionality reduction)
# -------------------------
if params["preprocessing"].get("pca", False):
    n_components = params["preprocessing"].get("pca_components", 0.95)  # keep 95% variance
    pca = PCA(n_components=n_components, random_state=params["training"]["random_state"])
    X = pd.DataFrame(pca.fit_transform(X))

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)

# -------------------------
# Encode target labels (important for XGBoost, LGBM, CatBoost)
# -------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Save LabelEncoder for later use (inverse transform in evaluate/prediction)
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

# -------------------------
# Class Balancing (SMOTE)
# -------------------------
if params["preprocessing"].get("smote", False):
    smote = SMOTE(random_state=params["training"]["random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)

# -------------------------
# Save processed data
# -------------------------
os.makedirs("data/processed", exist_ok=True)

X_train.to_csv("data/processed/X_train.csv", index=False)
pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
pd.Series(y_test).to_csv("data/processed/y_test.csv", index=False)

print("✅ Preprocessing complete! Saved X_train, y_train, X_test, y_test and label_encoder.pkl.")