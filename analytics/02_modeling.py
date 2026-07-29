from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "titanic.csv"
CHARTS_DIR = MODULE_DIR / "charts"
OUTPUTS_DIR = MODULE_DIR / "outputs"
MODELS_DIR = MODULE_DIR / "models"
RANDOM_STATE = 42


def preprocessor() -> ColumnTransformer:
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def metrics(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run analytics/01_eda.py first.")
    for directory in [CHARTS_DIR, OUTPUTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X, y = df[features], df["survived"]
    pd.DataFrame(
        {"count": y.value_counts(), "percentage": y.value_counts(normalize=True) * 100}
    ).to_csv(OUTPUTS_DIR / "class_balance.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    base = preprocessor()
    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", clone(base)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", clone(base)),
                ("classifier", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", clone(base)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                ),
            ]
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append(metrics(name, model, X_test, y_test))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["Not Survived", "Survived"]
        )
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close()
    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUTPUTS_DIR / "classification_comparison.csv", index=False)
    comparison.to_csv(OUTPUTS_DIR / "final_classifier_metrics.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probability):.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classifier ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "roc_curves.png", dpi=160)
    plt.close()

    tree_pipeline = models["Decision Tree"]
    plt.figure(figsize=(24, 12))
    plot_tree(
        tree_pipeline.named_steps["classifier"],
        feature_names=tree_pipeline.named_steps["preprocessor"].get_feature_names_out(),
        class_names=["Not Survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "decision_tree.png", dpi=160, bbox_inches="tight")
    plt.close()

    imbalance_models = {
        "Baseline": Pipeline(
            [
                ("preprocessor", clone(base)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Class Weight Balanced": Pipeline(
            [
                ("preprocessor", clone(base)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "SMOTE": ImbPipeline(
            [
                ("preprocessor", clone(base)),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
    }
    imbalance_rows = []
    for strategy, model in imbalance_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        imbalance_rows.append(
            {
                "strategy": strategy,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
    pd.DataFrame(imbalance_rows).sort_values("f1", ascending=False).to_csv(
        OUTPUTS_DIR / "imbalance_comparison.csv", index=False
    )

    tuning = Pipeline(
        [
            ("preprocessor", clone(base)),
            (
                "classifier",
                RandomForestClassifier(
                    oob_score=True, bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
    grid = GridSearchCV(
        tuning,
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    pd.DataFrame(grid.cv_results_).to_csv(OUTPUTS_DIR / "grid_search_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "best_parameters": str(grid.best_params_),
                "best_cv_f1": grid.best_score_,
                "oob_score": best_pipeline.named_steps["classifier"].oob_score_,
            }
        ]
    ).to_csv(OUTPUTS_DIR / "best_random_forest.csv", index=False)

    reg_features = ["survived", "pclass", "sex", "age", "sibsp", "parch", "embarked"]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        df[reg_features], df["fare"], test_size=0.2, random_state=RANDOM_STATE
    )
    reg_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["survived", "pclass", "age", "sibsp", "parch"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["sex", "embarked"],
            ),
        ]
    )
    regression = Pipeline(
        [("preprocessor", reg_preprocessor), ("regressor", LinearRegression())]
    )
    regression.fit(X_reg_train, y_reg_train)
    predicted = regression.predict(X_reg_test)
    mae = mean_absolute_error(y_reg_test, predicted)
    rmse = mean_squared_error(y_reg_test, predicted) ** 0.5
    r2 = r2_score(y_reg_test, predicted)
    p = len(regression.named_steps["preprocessor"].get_feature_names_out())
    n = len(y_reg_test)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    regression_metrics = pd.DataFrame(
        [
            {
                "model": "Multivariate Linear Regression",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
            }
        ]
    )
    regression_metrics.to_csv(OUTPUTS_DIR / "regression_metrics.csv", index=False)
    regression_metrics.to_csv(OUTPUTS_DIR / "final_regression_metrics.csv", index=False)
    sns.scatterplot(x=predicted, y=y_reg_test - predicted)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "regression_residual_plot.png", dpi=160)
    plt.close()

    path = MODELS_DIR / "best_titanic_pipeline.joblib"
    joblib.dump(best_pipeline, path)
    loaded = joblib.load(path)
    sample = pd.DataFrame(
        [
            {
                "pclass": 1,
                "sex": "female",
                "age": 29,
                "sibsp": 0,
                "parch": 0,
                "fare": 80.0,
                "embarked": "S",
            }
        ]
    )
    print(
        f"Reload test: prediction={loaded.predict(sample)[0]}, "
        f"probability={loaded.predict_proba(sample)[0, 1]:.4f}"
    )
    print(f"Module 2 modeling completed. Saved model: {path}")


if __name__ == "__main__":
    main()
