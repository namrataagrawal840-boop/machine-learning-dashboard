
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import load_dataset, get_data_summary

from algorithms.classification import run_classification
from algorithms.clustering import run_clustering
from algorithms.regression import run_regression
from algorithms.forecasting import run_forecasting
from algorithms.association import run_association
from algorithms.lca import run_lca


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Learning Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Dashboard")
st.caption(
    "An interactive platform for running, evaluating "
    "and comparing machine learning algorithms."
)


# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset from the sidebar to begin.")
    st.markdown("""
    ### Available analysis modules

    1. Classification
    2. Clustering
    3. Latent Class Analysis
    4. Regression
    5. Forecasting
    6. Association Mining

    Once you upload a dataset, you can select a module and run its algorithms.
    """)
    st.stop()

try:
    df = load_dataset(uploaded_file)

    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("Dataset Overview")

summary = get_data_summary(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", summary["rows"])
col2.metric("Total Columns", summary["columns"])
col3.metric("Missing Values", summary["missing_values"])
col4.metric("Duplicate Rows", summary["duplicates"])

with st.expander("Preview Dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Dataset Information"):
    st.write("Numerical columns:", summary["numerical_columns"])
    st.write("Categorical columns:", summary["categorical_columns"])


# --------------------------------------------------
# ALGORITHM SELECTION
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.header("Machine Learning")

category = st.sidebar.selectbox(
    "Choose an algorithm category",
    [
        "Classification",
        "Clustering",
        "Latent Class Analysis",
        "Regression",
        "Forecasting",
        "Association Mining"
    ]
)

st.header(category)


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if category == "Classification":

    st.write("Predict categorical outcomes and compare four classification models.")

    target = st.selectbox(
        "Select target column",
        df.columns,
        key="classification_target"
    )

    if st.button("Run Classification", type="primary"):

        try:
            with st.spinner("Running classification models..."):

                output = run_classification(df, target)

            st.subheader("Model Comparison")

            rows = []

            for name, result in output["results"].items():

                if result.get("error"):
                    st.warning(f"{name}: {result['error']}")
                    continue

                metrics = result["metrics"]

                rows.append({
                    "Algorithm": name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"]
                })

            if rows:

                comparison = pd.DataFrame(rows)
                st.dataframe(
                    comparison.round(4),
                    use_container_width=True
                )

                st.subheader("Confusion Matrices")

                for name, result in output["results"].items():

                    if result.get("error"):
                        continue

                    classes = result["classes"]
                    cm = result["metrics"]["confusion_matrix"]

                    st.markdown(f"**{name}**")

                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"Actual {x}" for x in classes],
                        columns=[f"Predicted {x}" for x in classes]
                    )

                    st.dataframe(cm_df, use_container_width=True)

                    metrics = result["metrics"]

                    if "false_positive" in metrics:
                        st.write(
                            f"False Positives: {metrics['false_positive']} | "
                            f"False Negatives: {metrics['false_negative']}"
                        )

        except Exception as error:
            st.error(f"Classification failed: {error}")


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

elif category == "Clustering":

    st.write("Discover groups of similar records without a target column.")

    number_of_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=max(2, min(10, len(df) - 1)),
        value=min(3, max(2, len(df) - 1))
    )

    if st.button("Run Clustering", type="primary"):

        try:
            with st.spinner("Running clustering models..."):

                output = run_clustering(
                    df,
                    n_clusters=number_of_clusters
                )

            for name in ["K-Means", "Hierarchical Clustering"]:

                result = output["results"][name]

                st.subheader(name)

                if result.get("error"):
                    st.warning(result["error"])
                    continue

                st.write("Evaluation Metrics")
                st.json(result["metrics"])

                visualisation = output["results"]["visualisation"]

                chart_df = pd.DataFrame({
                    "Component 1": visualisation["x"],
                    "Component 2": visualisation["y"],
                    "Cluster": result["labels"]
                })

                fig = px.scatter(
                    chart_df,
                    x="Component 1",
                    y="Component 2",
                    color="Cluster",
                    title=f"{name} - Cluster Visualisation"
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as error:
            st.error(f"Clustering failed: {error}")


# --------------------------------------------------
# LATENT CLASS ANALYSIS
# --------------------------------------------------

elif category == "Latent Class Analysis":

    st.write(
        "Identify hidden customer or respondent segments "
        "from categorical response patterns."
    )

    indicators = st.multiselect(
        "Select categorical indicator columns",
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    col1, col2 = st.columns(2)

    with col1:
        min_classes = st.number_input(
            "Minimum number of classes",
            min_value=2,
            max_value=10,
            value=2
        )

    with col2:
        max_classes = st.number_input(
            "Maximum number of classes",
            min_value=min_classes,
            max_value=10,
            value=max(2, min_classes + 2)
        )

    if st.button("Run Latent Class Analysis", type="primary"):

        try:
            with st.spinner("Estimating latent class models..."):

                output = run_lca(
                    df,
                    indicators,
                    int(min_classes),
                    int(max_classes)
                )

            st.subheader("Model Selection")

            comparison = pd.DataFrame(
                output["model_comparison"]
            )

            st.dataframe(
                comparison.round(4),
                use_container_width=True
            )

            st.metric(
                "Best Number of Classes (BIC)",
                output["best_class_count"]
            )

            st.subheader("Class Sizes")

            sizes = pd.DataFrame({
                "Latent Class": [
                    f"Class {key}"
                    for key in output["class_sizes"]
                ],
                "Customers": list(output["class_sizes"].values())
            })

            st.bar_chart(
                sizes.set_index("Latent Class")
            )

            st.subheader("Customer Class Assignments")

            assignments = pd.DataFrame(output["assignments"])
            probabilities = pd.DataFrame(output["probabilities"])

            final_assignments = pd.concat(
                [assignments, probabilities],
                axis=1
            )

            st.dataframe(
                final_assignments,
                use_container_width=True
            )

            st.download_button(
                "Download LCA Results",
                final_assignments.to_csv(index=False),
                file_name="lca_results.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Latent Class Analysis failed: {error}")


# --------------------------------------------------
# REGRESSION
# --------------------------------------------------

elif category == "Regression":

    st.write("Predict numerical outcomes and compare regression models.")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numerical_columns:
        st.warning("This dataset has no numerical columns for regression.")

    else:

        target = st.selectbox(
            "Select numerical target column",
            numerical_columns,
            key="regression_target"
        )

        if st.button("Run Regression", type="primary"):

            try:
                with st.spinner("Running regression models..."):

                    output = run_regression(df, target)

                rows = []

                for name, result in output["results"].items():

                    if result.get("error"):
                        st.warning(f"{name}: {result['error']}")
                        continue

                    metrics = result["metrics"]

                    rows.append({
                        "Algorithm": name,
                        "MAE": metrics["mae"],
                        "MSE": metrics["mse"],
                        "RMSE": metrics["rmse"],
                        "R² Score": metrics["r2_score"]
                    })

                if rows:

                    st.subheader("Regression Comparison")

                    st.dataframe(
                        pd.DataFrame(rows).round(4),
                        use_container_width=True
                    )

                    for name, result in output["results"].items():

                        if result.get("error"):
                            continue

                        chart_df = pd.DataFrame({
                            "Actual": result["actual"],
                            "Predicted": result["predictions"]
                        })

                        st.markdown(f"**{name}: Actual vs Predicted**")

                        fig = px.scatter(
                            chart_df,
                            x="Actual",
                            y="Predicted",
                            title=name
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

            except Exception as error:
                st.error(f"Regression failed: {error}")


# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

elif category == "Forecasting":

    st.write(
        "Forecast future numerical values using historical time-series data."
    )

    date_column = st.selectbox(
        "Select date column",
        df.columns,
        key="forecast_date"
    )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numerical_columns:

        value_column = st.selectbox(
            "Select numerical value to forecast",
            numerical_columns,
            key="forecast_value"
        )

        forecast_periods = st.number_input(
            "Number of future periods",
            min_value=1,
            max_value=100,
            value=5
        )

        if st.button("Run Forecasting", type="primary"):

            try:
                with st.spinner("Running forecasting models..."):

                    output = run_forecasting(
                        df,
                        date_column,
                        value_column,
                        int(forecast_periods)
                    )

                st.subheader("Forecast Comparison")

                rows = []

                for name, result in output["results"].items():

                    if result.get("error"):
                        st.warning(f"{name}: {result['error']}")
                        continue

                    rows.append({
                        "Method": name,
                        **result["metrics"]
                    })

                if rows:
                    st.dataframe(
                        pd.DataFrame(rows).round(4),
                        use_container_width=True
                    )

                st.subheader("Future Forecasts")

                future_df = pd.DataFrame({
                    "Date": output["future_dates"]
                })

                for name, result in output["results"].items():

                    if not result.get("error"):
                        future_df[name] = result["future_forecast"]

                st.dataframe(
                    future_df.round(2),
                    use_container_width=True
                )

                if len(future_df.columns) > 1:

                    chart_df = future_df.melt(
                        id_vars="Date",
                        var_name="Method",
                        value_name="Forecast"
                    )

                    fig = px.line(
                        chart_df,
                        x="Date",
                        y="Forecast",
                        color="Method",
                        title="Future Forecast Comparison"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            except Exception as error:
                st.error(f"Forecasting failed: {error}")

    else:
        st.warning("This dataset has no numerical columns to forecast.")


# --------------------------------------------------
# ASSOCIATION MINING
# --------------------------------------------------

elif category == "Association Mining":

    st.write(
        "Discover products frequently purchased together "
        "and generate association rules."
    )

    col1, col2 = st.columns(2)

    with col1:
        transaction_column = st.selectbox(
            "Select transaction ID column",
            df.columns,
            key="transaction_column"
        )

    with col2:
        item_column = st.selectbox(
            "Select product/item column",
            df.columns,
            key="item_column"
        )

    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=1.0,
        value=0.05,
        step=0.01
    )

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

    if st.button("Run Association Mining", type="primary"):

        try:
            with st.spinner("Finding product associations..."):

                output = run_association(
                    df,
                    transaction_column,
                    item_column,
                    min_support,
                    min_confidence
                )

            st.metric(
                "Total Transactions",
                output["total_transactions"]
            )

            for name, result in output["results"].items():

                st.subheader(name)

                if result.get("error"):
                    st.warning(result["error"])
                    continue

                itemsets = pd.DataFrame(
                    result["frequent_itemsets"]
                )

                rules = pd.DataFrame(
                    result["rules"]
                )

                st.write("Frequent Itemsets")

                if not itemsets.empty:
                    st.dataframe(
                        itemsets,
                        use_container_width=True
                    )
                else:
                    st.info(result.get("message", "No frequent itemsets found."))

                st.write("Association Rules")

                if not rules.empty:

                    rules["antecedents"] = rules[
                        "antecedents"
                    ].apply(lambda x: ", ".join(x))

                    rules["consequents"] = rules[
                        "consequents"
                    ].apply(lambda x: ", ".join(x))

                    st.dataframe(
                        rules.round(4),
                        use_container_width=True
                    )

                else:
                    st.info("No rules found with the selected confidence.")

        except Exception as error:
            st.error(f"Association Mining failed: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Dashboard | Classification | Clustering | "
    "LCA | Regression | Forecasting | Association Mining"
)
