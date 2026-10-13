"""
Stroke Prediction Using Machine Learningggg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("healthcare-dataset-stroke-data.csv")
RANDOM_STATE = 42


def load_and_explore_data(path: Path) -> pd.DataFrame:
    """Load the CSV and display basic information about the dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place healthcare-dataset-stroke-data.csv "
            "in the same folder as this script."
        )

    df = pd.read_csv(path)

    print("\n=== Dataset overview ===")
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nColumn information:")
    df.info()
    print("\nNumerical summary:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nStroke counts (0 = no stroke, 1 = stroke):")
    print(df["stroke"].value_counts())
    print("\nStroke percentages:")
    print((df["stroke"].value_counts(normalize=True) * 100).round(2))

    for column in [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]:
        print(f"\nUnique values in {column}:")
        print(df[column].unique())

    return df


def visualize_data(df: pd.DataFrame) -> None:
    """Create exploratory plots before preprocessing."""
    sns.set_theme(style="whitegrid")

    
    sns.countplot(data=df, x="stroke")
    plt.title("Stroke Distribution (0 = No Stroke, 1 = Stroke)")
    plt.tight_layout()
    plt.show()

    
    sns.boxplot(data=df, x="stroke", y="age")
    plt.title("Age Distribution by Stroke Status")
    plt.tight_layout()
    plt.show()

    
    sns.boxplot(data=df, x="stroke", y="avg_glucose_level")
    plt.title("Average Glucose Level by Stroke Status")
    plt.tight_layout()
    plt.show()

    
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    
    hypertension_rates = pd.crosstab(
        df["hypertension"], df["stroke"], normalize="index"
    ) * 100
    print("\nStroke percentage within each hypertension group:")
    print(hypertension_rates.round(2))

    hypertension_rates.plot(kind="bar", stacked=True)
    plt.title("Stroke Distribution by Hypertension Status")
    plt.xlabel("Hypertension (0 = No, 1 = Yes)")
    plt.ylabel("Percentage within group")
    plt.legend(title="Stroke", labels=["No stroke (0)", "Stroke (1)"])
    plt.tight_layout()
    plt.show()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing choices used in the assignment."""
    df = df.copy()

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    df = df[df["gender"] != "Other"].copy()

    
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
    df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})

    
    df = pd.get_dummies(
        df,
        columns=["work_type", "smoking_status"],
        drop_first=True,
        dtype=int,
    )

    print("\n=== After preprocessing ===")
    print("Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nProcessed columns:")
    print(df.columns.tolist())

    if df.isnull().any().any():
        raise ValueError("Missing values remain after preprocessing.")
    return df


def evaluate_model(name, model, X_test, y_test):
    """Print common classification metrics and return predictions and scores."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["No stroke", "Stroke"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"{name} - ROC Curve")
    plt.tight_layout()
    plt.show()

    return predictions, probabilities


def main():
    original_df = load_and_explore_data(DATA_PATH)

    
    visualize_data(original_df)

    df = preprocess_data(original_df)
    
    X = df.drop(["id", "stroke"], axis=1)
    y = df["stroke"]

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("\nTraining feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    logistic_model.fit(X_train, y_train)
    evaluate_model("Logistic Regression", logistic_model, X_test, y_test)

    
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    random_forest_model.fit(X_train, y_train)
    evaluate_model("Random Forest", random_forest_model, X_test, y_test)

    
    importances = pd.Series(
        random_forest_model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    print("\nTop feature importances from Random Forest:")
    print(importances.head(10))

    importances.head(15).sort_values().plot(kind="barh", figsize=(9, 6))
    plt.title("Random Forest Feature Importances (Top 15)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    print(
        "\nReminder: these are educational model results. "
        "They are not clinical evidence and should not be used for medical decisions."
    )


if __name__ == "__main__":
    main()
