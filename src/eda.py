import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set(style="whitegrid")

def load_params():
    """Load params.yaml configuration"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def main():
    # Load dataset
    params = load_params()
    dataset_path = params["dataset"]
    df = pd.read_csv(dataset_path)

    # Create output folder
    os.makedirs("reports/eda", exist_ok=True)

    # 1. Dataset info
    with open("reports/eda/dataset_info.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing Values:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nSummary Statistics:\n")
        f.write(str(df.describe()))

    # 2. Distribution plots for each feature
    for col in df.columns:
        if col != "quality":
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {col}")
            plt.savefig(f"reports/eda/{col}_distribution.jpg")
            plt.close()

    # 3. Boxplots for outlier detection
    for col in df.columns:
        if df[col].dtype != "object" and col != "quality":
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col], color="lightgreen")
            plt.title(f"Boxplot of {col}")
            plt.savefig(f"reports/eda/{col}_boxplot.jpg")
            plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("reports/eda/correlation_heatmap.jpg")
    plt.close()

    # 5. Pairplot (relationships between features & quality)
    # Only sample few columns for readability
    sample_cols = ["alcohol", "sulphates", "citric acid", "quality"]
    sns.pairplot(df[sample_cols], hue="quality", diag_kind="kde", palette="husl")
    plt.savefig("reports/eda/pairplot.jpg")
    plt.close()

    # 6. Countplot for target variable (quality distribution)
    plt.figure(figsize=(6, 4))
    sns.countplot(x="quality", data=df, palette="muted")
    plt.title("Class Distribution (Wine Quality)")
    plt.savefig("reports/eda/quality_distribution.jpg")
    plt.close()

    # 7. Violin plots (distribution per target class)
    for col in ["alcohol", "sulphates", "citric acid"]:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x="quality", y=col, data=df, palette="pastel")
        plt.title(f"{col} by Wine Quality")
        plt.savefig(f"reports/eda/{col}_violin.jpg")
        plt.close()

    # 8. 3D Scatter Plot (alcohol, sulphates, citric acid vs quality)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["alcohol"], df["sulphates"], df["citric acid"],
               c=df["quality"], cmap="viridis", alpha=0.7)
    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Sulphates")
    ax.set_zlabel("Citric Acid")
    plt.title("3D Scatter: Alcohol vs Sulphates vs Citric Acid (colored by Quality)")
    plt.savefig("reports/eda/3d_scatter_quality.jpg")
    plt.close()

    print("✅ EDA completed. Plots and summary saved in reports/eda/")

if __name__ == "__main__":
    main()