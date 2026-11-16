
import pandas as pd
import numpy as np


def load_dataset(file):
    """Load a CSV or Excel dataset."""

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)

    else:
        raise ValueError("Please upload a CSV or Excel file.")

    return df


def get_column_types(df):
    """Identify numerical and categorical columns."""

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return numerical_columns, categorical_columns


def clean_dataset(df, target_column=None):
    """Clean missing values and duplicate records."""

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove columns containing only missing values
    df = df.dropna(axis=1, how="all")

    # Remove rows with missing target values
    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    # Identify column types
    numerical, categorical = get_column_types(df)

    # Fill missing numerical values using median
    for column in numerical:
        if column != target_column:
            df[column] = df[column].fillna(
                df[column].median()
            )

    # Fill missing categorical values using mode
    for column in categorical:
        if column != target_column and df[column].isna().any():
            mode_values = df[column].mode()

            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "Unknown"
            )

            df[column] = df[column].fillna(fill_value)

    return df


def get_data_summary(df):
    """Generate a basic dataset report."""

    numerical, categorical = get_column_types(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
