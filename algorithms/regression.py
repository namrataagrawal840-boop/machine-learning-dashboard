
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from utils.evaluation import evaluate_regression


def run_regression(
    df,
    target_column,
    test_size=0.2,
    random_state=42
):
    """Run and compare simple, multiple and polynomial regression."""

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    df = df.dropna(subset=[target_column])
    df = df.dropna(axis=1, how="all")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("Regression requires a numerical target column.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty:
        raise ValueError("No feature columns available.")

    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if not numerical_columns:
        raise ValueError(
            "At least one numerical feature is required for simple regression."
        )

    if len(df) < 5:
        raise ValueError("At least 5 usable rows are required.")

    # Preprocessing for multiple linear regression
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = [
        ("numerical", numerical_pipeline, numerical_columns)
    ]

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    multiple_preprocessor = ColumnTransformer(transformers)

    # Preprocessing for simple linear regression
    simple_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for polynomial regression
    polynomial_numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False))
    ])

    polynomial_transformers = [
        ("numerical", polynomial_numerical, numerical_columns)
    ]

    if categorical_columns:
        polynomial_transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    polynomial_preprocessor = ColumnTransformer(
        polynomial_transformers
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    models = {
        "Simple Linear Regression": Pipeline([
            (
                "preprocessing",
                simple_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]),

        "Multiple Linear Regression": Pipeline([
            (
                "preprocessing",
                multiple_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]),

        "Polynomial Regression": Pipeline([
            (
                "preprocessing",
                polynomial_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ])
    }

    results = {}

    for name, model in models.items():

        try:
            if name == "Simple Linear Regression":
                # Use the first numerical feature only
                simple_column = numerical_columns[0]

                model.fit(
                    X_train[[simple_column]],
                    y_train
                )

                predictions = model.predict(
                    X_test[[simple_column]]
                )

            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            metrics = evaluate_regression(
                y_test,
                predictions
            )

            results[name] = {
                "model": model,
                "metrics": metrics,
                "predictions": predictions.tolist(),
                "actual": y_test.tolist(),
                "error": None
            }

        except Exception as error:
            results[name] = {
                "error": str(error)
            }

    return {
        "results": results,
        "target": target_column,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
