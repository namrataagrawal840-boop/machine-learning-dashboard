
import pandas as pd
import numpy as np

from stepmix.stepmix import StepMix


def run_lca(
    df,
    indicator_columns,
    min_classes=2,
    max_classes=5,
    random_state=42
):
    """Run categorical Latent Class Analysis and compare class solutions."""

    df = df.copy()

    if not indicator_columns:
        raise ValueError("Select at least two indicator columns.")

    missing_columns = [
        col for col in indicator_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    if len(indicator_columns) < 2:
        raise ValueError("LCA requires at least two indicators.")

    data = df[indicator_columns].copy()

    # Convert indicators to categorical values
    for column in indicator_columns:
        mode = data[column].mode()

        if mode.empty:
            raise ValueError(f"No valid values in {column}.")

        data[column] = data[column].fillna(mode.iloc[0]).astype(str)

    # Remove indicators with only one category
    usable_columns = [
        col for col in data.columns
        if data[col].nunique() >= 2
    ]

    if len(usable_columns) < 2:
        raise ValueError(
            "At least two indicators must have two or more categories."
        )

    data = data[usable_columns]

    if len(data) < 10:
        raise ValueError("LCA requires at least 10 observations.")

    # Integer-encode each categorical indicator separately
    encoded_data = pd.DataFrame(index=data.index)
    category_maps = {}

    for column in usable_columns:
        codes, categories = pd.factorize(
            data[column],
            sort=True
        )

        encoded_data[column] = codes

        category_maps[column] = {
            int(index): str(category)
            for index, category in enumerate(categories)
        }

    X = encoded_data.to_numpy(dtype=int)

    if min_classes < 2:
        raise ValueError("Minimum classes must be at least 2.")

    if max_classes < min_classes:
        raise ValueError("Maximum classes must be >= minimum classes.")

    if max_classes >= len(data):
        raise ValueError(
            "Maximum classes must be smaller than the number of observations."
        )

    model_comparison = []
    fitted_models = {}

    # Fit multiple latent-class solutions
    for number_of_classes in range(
        min_classes, max_classes + 1
    ):

        try:
            model = StepMix(
                n_components=number_of_classes,
                measurement="categorical",
                n_init=5,
                max_iter=500,
                random_state=random_state,
                verbose=0
            )

            model.fit(X)

            aic = model.aic(X)
            bic = model.bic(X)

            fitted_models[number_of_classes] = model

            model_comparison.append({
                "classes": number_of_classes,
                "aic": float(aic),
                "bic": float(bic),
                "converged": bool(model.converged_),
                "error": None
            })

        except Exception as error:
            model_comparison.append({
                "classes": number_of_classes,
                "error": str(error)
            })

    if not fitted_models:
        raise ValueError(
            "No LCA model could be fitted. Check the data and indicators."
        )

    # Select the model with the lowest BIC
    best_class_count = min(
        fitted_models,
        key=lambda k: fitted_models[k].bic(X)
    )

    best_model = fitted_models[best_class_count]

    # Assign customers to their most likely latent class
    class_labels = best_model.predict_class(X)

    # Calculate posterior membership probabilities
    probabilities = best_model.predict_proba_class(X)

    assignments = pd.DataFrame({
        "original_row": data.index,
        "latent_class": class_labels + 1
    })

    probability_columns = [
        f"Class_{i + 1}_probability"
        for i in range(best_class_count)
    ]

    probability_df = pd.DataFrame(
        probabilities,
        columns=probability_columns,
        index=data.index
    )

    class_sizes = assignments[
        "latent_class"
    ].value_counts().sort_index()

    return {
        "best_model": best_model,
        "best_class_count": best_class_count,
        "model_comparison": model_comparison,
        "assignments": assignments.to_dict("records"),
        "probabilities": probability_df.reset_index(drop=True).to_dict("records"),
        "class_sizes": {
            int(class_id): int(count)
            for class_id, count in class_sizes.items()
        },
        "category_maps": category_maps,
        "indicators": usable_columns,
        "observations": len(data)
    }
