
import pandas as pd

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def run_association(
    df,
    transaction_column,
    item_column,
    min_support=0.05,
    min_confidence=0.30
):
    """Run Apriori and FP-Growth association mining."""

    df = df.copy()

    if transaction_column not in df.columns:
        raise ValueError("Transaction ID column not found.")

    if item_column not in df.columns:
        raise ValueError("Item column not found.")

    if not 0 < min_support <= 1:
        raise ValueError("Minimum support must be between 0 and 1.")

    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")

    df = df.dropna(
        subset=[transaction_column, item_column]
    )

    if df.empty:
        raise ValueError("No valid transactions found.")

    # Create a transaction-item basket
    basket = pd.crosstab(
        df[transaction_column],
        df[item_column]
    )

    basket = basket > 0

    if basket.shape[1] < 2:
        raise ValueError("At least two different items are required.")

    if basket.shape[0] < 2:
        raise ValueError("At least two transactions are required.")

    results = {}

    algorithms = {
        "Apriori": apriori,
        "FP-Growth": fpgrowth
    }

    for name, algorithm in algorithms.items():

        try:
            # Find frequently purchased item combinations
            frequent_itemsets = algorithm(
                basket,
                min_support=min_support,
                use_colnames=True
            )

            if frequent_itemsets.empty:
                results[name] = {
                    "frequent_itemsets": [],
                    "rules": [],
                    "message": "No frequent item combinations found. Try lowering minimum support.",
                    "error": None
                }
                continue

            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            itemsets_output = []

            for _, row in frequent_itemsets.iterrows():
                itemsets_output.append({
                    "items": sorted(list(row["itemsets"])),
                    "support": float(row["support"])
                })

            rules_output = []

            for _, row in rules.iterrows():
                rules_output.append({
                    "antecedents": sorted(list(row["antecedents"])),
                    "consequents": sorted(list(row["consequents"])),
                    "support": float(row["support"]),
                    "confidence": float(row["confidence"]),
                    "lift": float(row["lift"])
                })

            results[name] = {
                "frequent_itemsets": itemsets_output,
                "rules": rules_output,
                "error": None
            }

        except Exception as error:
            results[name] = {
                "error": str(error)
            }

    return {
        "results": results,
        "total_transactions": len(basket),
        "total_items": len(basket.columns),
        "items": basket.columns.tolist()
    }
