
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from utils.evaluation import evaluate_classification


def run_classification(
    df,
    target_column,
    test_size=0.2,
    random_state=42
):
    """Run four classification algorithms and compare performance."""

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    df = df.dropna(subset=[target_column])
    df = df.dropna(axis=1, how="all")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty:
        raise ValueError("No feature columns available.")

    if y.nunique() < 2:
        raise ValueError("Classification requires at least two classes.")

    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers)

    class_counts = y.value_counts()
    number_of_classes = y.nunique()
    number_of_test_rows = int(np.ceil(len(y) * test_size))
    number_of_train_rows = len(y) - number_of_test_rows

    stratify = None

    if (
        class_counts.min() >= 2
        and number_of_test_rows >= number_of_classes
        and number_of_train_rows >= number_of_classes
    ):
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    models = {
        "K-NN": KNeighborsClassifier(
            n_neighbors=min(5, len(X_train))
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        )
    }

    results = {}

    for name, algorithm in models.items():

        model_pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", algorithm)
        ])

        try:
            model_pipeline.fit(X_train, y_train)

            predictions = model_pipeline.predict(X_test)

            metrics = evaluate_classification(
                y_test,
                predictions
            )

            results[name] = {
                "model": model_pipeline,
                "metrics": metrics,
                "predictions": predictions,
                "actual": y_test,
                "classes": model_pipeline.classes_,
                "error": None
            }

        except Exception as error:
            results[name] = {
                "error": str(error)
            }

    return {
        "results": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target": target_column
    }
