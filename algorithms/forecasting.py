
import pandas as pd
import numpy as np

from math import ceil

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.evaluation import evaluate_regression


def run_forecasting(
    df,
    date_column,
    value_column,
    forecast_periods=5,
    test_size=0.2
):
    """Run and compare three time-series forecasting methods."""

    df = df.copy()

    if date_column not in df.columns:
        raise ValueError("Date column not found.")

    if value_column not in df.columns:
        raise ValueError("Value column not found.")

    if forecast_periods < 1:
        raise ValueError("Forecast periods must be at least 1.")

    # Prepare dates and numerical values
    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce"
    )

    df[value_column] = pd.to_numeric(
        df[value_column], errors="coerce"
    )

    df = df.dropna(
        subset=[date_column, value_column]
    )

    # Sort observations chronologically
    df = df.sort_values(date_column)

    # Combine duplicate dates
    df = df.groupby(
        date_column, as_index=False
    )[value_column].mean()

    if len(df) < 10:
        raise ValueError(
            "At least 10 observations are required for forecasting."
        )

    values = df[value_column].astype(float).reset_index(drop=True)
    dates = df[date_column].reset_index(drop=True)

    # Keep the latest observations for testing
    test_rows = max(2, ceil(len(values) * test_size))
    train_rows = len(values) - test_rows

    if train_rows < 5:
        raise ValueError("Not enough historical observations for training.")

    train = values.iloc[:train_rows]
    test = values.iloc[train_rows:]

    results = {}

    # 1. Time Series Trend Forecasting
    try:
        train_index = np.arange(len(train)).reshape(-1, 1)

        trend_model = LinearRegression()
        trend_model.fit(train_index, train)

        test_index = np.arange(
            len(train), len(values)
        ).reshape(-1, 1)

        predictions = trend_model.predict(test_index)

        metrics = evaluate_regression(
            test, predictions
        )

        # Refit using all observations for future forecasting
        final_model = LinearRegression()
        final_model.fit(
            np.arange(len(values)).reshape(-1, 1),
            values
        )

        future_index = np.arange(
            len(values),
            len(values) + forecast_periods
        ).reshape(-1, 1)

        future_forecast = final_model.predict(future_index)

        results["Time Series Trend"] = {
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "future_forecast": future_forecast.tolist(),
            "error": None
        }

    except Exception as error:
        results["Time Series Trend"] = {
            "error": str(error)
        }

    # 2. ARIMA Forecasting
    try:
        arima_model = ARIMA(
            train,
            order=(1, 1, 1)
        ).fit()

        predictions = arima_model.forecast(
            steps=len(test)
        )

        metrics = evaluate_regression(
            test, predictions
        )

        final_arima = ARIMA(
            values,
            order=(1, 1, 1)
        ).fit()

        future_forecast = final_arima.forecast(
            steps=forecast_periods
        )

        results["ARIMA"] = {
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "future_forecast": future_forecast.tolist(),
            "error": None
        }

    except Exception as error:
        results["ARIMA"] = {
            "error": str(error)
        }

    # 3. Exponential Smoothing
    try:
        smoothing_model = ExponentialSmoothing(
            train,
            trend="add",
            initialization_method="estimated"
        ).fit()

        predictions = smoothing_model.forecast(
            steps=len(test)
        )

        metrics = evaluate_regression(
            test, predictions
        )

        final_smoothing = ExponentialSmoothing(
            values,
            trend="add",
            initialization_method="estimated"
        ).fit()

        future_forecast = final_smoothing.forecast(
            steps=forecast_periods
        )

        results["Exponential Smoothing"] = {
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "future_forecast": future_forecast.tolist(),
            "error": None
        }

    except Exception as error:
        results["Exponential Smoothing"] = {
            "error": str(error)
        }

    # Estimate future dates using observed date intervals
    frequency = pd.infer_freq(dates)

    if frequency is not None:
        future_dates = pd.date_range(
            start=dates.iloc[-1],
            periods=forecast_periods + 1,
            freq=frequency
        )[1:]
    else:
        average_interval = dates.diff().median()

        future_dates = pd.date_range(
            start=dates.iloc[-1] + average_interval,
            periods=forecast_periods,
            freq=average_interval
        )

    return {
        "results": results,
        "historical_dates": dates.dt.strftime("%Y-%m-%d").tolist(),
        "historical_values": values.tolist(),
        "future_dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "date_column": date_column,
        "value_column": value_column,
        "train_size": len(train),
        "test_size": len(test)
    }
