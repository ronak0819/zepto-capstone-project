from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "titanic.csv"
CHARTS_DIR = MODULE_DIR / "charts"
OUTPUTS_DIR = MODULE_DIR / "outputs"


def save_chart(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def calculate_iqr_outliers(series: pd.Series) -> dict[str, float | int]:
    values = series.dropna()
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(((values < lower) | (values > upper)).sum()),
    }


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Using offline dataset: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    raw = sns.load_dataset("titanic")
    raw.to_csv(DATA_PATH, index=False)
    return raw


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    print("\nDATASET INFO")
    raw.info()
    print("\nDATASET DESCRIPTION\n", raw.describe())
    print("\nDATASET SHAPE\n", raw.shape)

    missing = pd.DataFrame(
        {"missing_count": raw.isna().sum(), "missing_percentage": raw.isna().mean() * 100}
    )
    missing = missing[missing.missing_count > 0].sort_values(
        "missing_percentage", ascending=False
    )
    missing.to_csv(OUTPUTS_DIR / "missing_values.csv")
    print("\nMISSING VALUES\n", missing)

    df = raw.drop(columns=["deck", "embark_town"], errors="ignore").copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.dropna(subset=["embarked"]).reset_index(drop=True)

    for column, title in [("age", "Age"), ("fare", "Fare")]:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f"{title} Distribution")
        save_chart(f"{column}_histogram.png")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=column)
        plt.title(f"{title} Box Plot")
        save_chart(f"{column}_boxplot.png")

    outliers = pd.DataFrame(
        [
            {"column": "age", **calculate_iqr_outliers(df["age"])},
            {"column": "fare", **calculate_iqr_outliers(df["fare"])},
        ]
    )
    outliers.to_csv(OUTPUTS_DIR / "outlier_report.csv", index=False)
    fare_mean, fare_median, fare_mode = (
        float(df.fare.mean()),
        float(df.fare.median()),
        float(df.fare.mode().iloc[0]),
    )
    skew = (
        "right-skewed"
        if fare_mean > fare_median
        else "left-skewed"
        if fare_mean < fare_median
        else "approximately symmetric"
    )
    pd.DataFrame(
        [{"mean": fare_mean, "median": fare_median, "mode": fare_mode, "interpretation": skew}]
    ).to_csv(OUTPUTS_DIR / "fare_statistics.csv", index=False)

    rows: list[dict[str, object]] = []
    for sex in ["female", "male"]:
        rows.append(
            {
                "analysis": "sex",
                "sex": sex,
                "pclass": None,
                "survival_rate": float(df.loc[df.sex == sex, "survived"].mean()),
            }
        )
    for pclass in [1, 2, 3]:
        rows.append(
            {
                "analysis": "pclass",
                "sex": None,
                "pclass": pclass,
                "survival_rate": float(df.loc[df.pclass == pclass, "survived"].mean()),
            }
        )
    for sex in ["female", "male"]:
        for pclass in [1, 2, 3]:
            mask = (df.sex == sex) & (df.pclass == pclass)
            rows.append(
                {
                    "analysis": "sex_and_pclass",
                    "sex": sex,
                    "pclass": pclass,
                    "survival_rate": float(df.loc[mask, "survived"].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "survival_rates.csv", index=False)

    columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    correlations = df[columns].corr()
    correlations.to_csv(OUTPUTS_DIR / "correlation_matrix.csv")
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlations, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Matrix: Required Six Columns")
    save_chart("correlation_heatmap.png")
    pairs = [
        {
            "feature_1": left,
            "feature_2": right,
            "correlation": float(correlations.loc[left, right]),
            "absolute_correlation": abs(float(correlations.loc[left, right])),
        }
        for left, right in combinations(columns, 2)
    ]
    (
        pd.DataFrame(pairs)
        .sort_values("absolute_correlation", ascending=False)
        .head(2)
        .to_csv(OUTPUTS_DIR / "strongest_correlations.csv", index=False)
    )

    charts = [
        ("bar", {"x": "sex", "y": "survived"}, "Survival Rate by Sex", "survival_by_sex.png"),
        (
            "bar",
            {"x": "pclass", "y": "survived"},
            "Survival Rate by Passenger Class",
            "survival_by_class.png",
        ),
        (
            "bar",
            {"x": "pclass", "y": "survived", "hue": "sex"},
            "Survival Rate by Sex and Class",
            "survival_by_sex_and_class.png",
        ),
        (
            "box",
            {"x": "survived", "y": "age"},
            "Age Distribution by Survival",
            "age_by_survival.png",
        ),
        (
            "box",
            {"x": "pclass", "y": "fare", "hue": "survived"},
            "Fare by Class and Survival",
            "fare_class_survival.png",
        ),
    ]
    for kind, kwargs, title, filename in charts:
        plt.figure(figsize=(9, 5))
        if kind == "bar":
            sns.barplot(data=df, errorbar=None, **kwargs)
        else:
            sns.boxplot(data=df, **kwargs)
        plt.title(title)
        save_chart(filename)

    standardized = StandardScaler().fit_transform(df[["age", "fare"]])
    standardized_df = pd.DataFrame(standardized, columns=["age_z", "fare_z"])
    pd.DataFrame(
        {
            "original_mean": [df.age.mean(), df.fare.mean()],
            "original_std_ddof0": [df.age.std(ddof=0), df.fare.std(ddof=0)],
            "standardized_mean": [standardized_df.age_z.mean(), standardized_df.fare_z.mean()],
            "standardized_std_ddof0": [
                standardized_df.age_z.std(ddof=0),
                standardized_df.fare_z.std(ddof=0),
            ],
        },
        index=["age", "fare"],
    ).to_csv(OUTPUTS_DIR / "standardization_report.csv")
    print(f"\nModule 2 EDA completed. Offline data: {DATA_PATH}")


if __name__ == "__main__":
    main()
