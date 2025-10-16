from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


RANDOM_STATE = 42
TEST_SIZE = 0.2
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
DATA_PATH = Path("diamonds.csv")


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = pd.read_csv(path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


def generate_descriptive_plots(data: pd.DataFrame) -> None:
    numeric_cols = ["carat", "depth", "table", "x", "y", "z", "price"]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.flatten(), numeric_cols[:-1]):
        sns.histplot(data[col], kde=True, ax=ax, bins=40, color="#4472C4")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_feature_distributions.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 5))
    sns.histplot(data["price"], kde=True, bins=50, color="#ED7D31")
    plt.title("Distribution of price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=200)
    plt.close()

    corr_cols = numeric_cols
    corr = data[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation matrix of numeric features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=200)
    plt.close()


def compute_vif(data: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["carat", "depth", "table", "x", "y", "z"]
    subset = data[numeric_cols].replace(0, 1e-6)
    vif_values = []
    for i, col in enumerate(numeric_cols):
        vif = variance_inflation_factor(subset.values, i)
        vif_values.append({"feature": col, "vif": vif})
    vif_df = pd.DataFrame(vif_values)
    vif_df.to_csv(RESULTS_DIR / "vif_values.csv", index=False)
    return vif_df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["carat", "depth", "table", "x", "y", "z"]
    categorical_features = ["cut", "color", "clarity"]

    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return preprocessor


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        "rmse": rmse,
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
    }

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "mape": "neg_mean_absolute_percentage_error",
    }
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring,
        n_jobs=None,
    )
    cv_metrics = {
        "rmse_cv": -np.mean(cv_results["test_rmse"]),
        "r2_cv": np.mean(cv_results["test_r2"]),
        "mape_cv": -np.mean(cv_results["test_mape"]),
    }
    return metrics, cv_metrics


def run_original_feature_models(data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop(columns=["price"])
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    preprocessor = build_preprocessor()

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=10)": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    }

    records = []
    for name, reg in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", reg),
            ]
        )
        test_metrics, cv_metrics = evaluate_model(
            pipeline, X_train, y_train, X_test, y_test
        )
        record = {"model": name}
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        record.update(cv_metrics)
        records.append(record)

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_DIR / "metrics_original_features.csv", index=False)
    return results


def run_pca_models(data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop(columns=["price"])
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    # Fit PCA on preprocessed training data to generate scree plot.
    preprocessed_train = preprocessor.fit_transform(X_train)
    pca.fit(preprocessed_train)

    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(explained) + 1),
        explained,
        marker="o",
        linestyle="-",
        color="#2F5597",
    )
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance (training set)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_explained_variance.png", dpi=200)
    plt.close()

    models = {
        "LinearRegression+PCA": LinearRegression(),
        "Ridge(alpha=10)+PCA": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    }

    records = []
    for name, reg in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("regressor", reg),
            ]
        )
        test_metrics, cv_metrics = evaluate_model(
            pipeline, X_train, y_train, X_test, y_test
        )
        record = {"model": name}
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        record.update(cv_metrics)
        records.append(record)

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_DIR / "metrics_pca_features.csv", index=False)
    return results


def main() -> None:
    ensure_output_dirs()
    data = load_dataset(DATA_PATH)
    print("Dataset shape:", data.shape)
    print("Column overview:\n", data.dtypes)
    print("\nMissing values per column:\n", data.isna().sum())

    generate_descriptive_plots(data)
    vif_df = compute_vif(data)
    print("\nVariance Inflation Factors:\n", vif_df)

    original_results = run_original_feature_models(data)
    print("\nModel performance on original features:\n", original_results)

    pca_results = run_pca_models(data)
    print("\nModel performance on PCA features:\n", pca_results)

    combined = pd.concat(
        [
            original_results.assign(dataset="original"),
            pca_results.assign(dataset="pca"),
        ],
        ignore_index=True,
    )
    combined.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    print("\nCombined metrics saved to results/metrics_comparison.csv")
    print("Figures saved to", FIGURES_DIR.resolve())


if __name__ == "__main__":
    main()

