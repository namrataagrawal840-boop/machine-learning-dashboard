
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from utils.evaluation import evaluate_clustering


def run_clustering(df, n_clusters=3, random_state=42):
    """Run K-Means and Hierarchical Clustering."""

    df = df.copy()

    df = df.select_dtypes(
        include=[np.number, "object", "category", "bool"]
    )

    df = df.dropna(axis=1, how="all")

    if len(df) < n_clusters + 1:
        raise ValueError("Not enough rows for the selected number of clusters.")

    if df.empty:
        raise ValueError("No usable columns found.")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers)

    X = preprocessor.fit_transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    if X.shape[1] == 0:
        raise ValueError("No usable features after preprocessing.")

    if n_clusters < 2 or n_clusters >= len(df):
        raise ValueError("Choose between 2 and rows minus 1 clusters.")

    models = {
        "K-Means": KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        ),
        "Hierarchical Clustering": AgglomerativeClustering(
            n_clusters=n_clusters
        )
    }

    results = {}

    for name, algorithm in models.items():

        try:
            labels = algorithm.fit_predict(X)

            metrics = evaluate_clustering(X, labels)

            results[name] = {
                "labels": labels.tolist(),
                "metrics": metrics,
                "error": None
            }

        except Exception as error:
            results[name] = {
                "error": str(error)
            }

    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X)
    else:
        reduced_data = np.column_stack([
            X.ravel(),
            np.zeros(len(df))
        ])

    results["visualisation"] = {
        "x": reduced_data[:, 0].tolist(),
        "y": reduced_data[:, 1].tolist()
    }

    return {
        "results": results,
        "rows": len(df),
        "features": df.columns.tolist()
    }