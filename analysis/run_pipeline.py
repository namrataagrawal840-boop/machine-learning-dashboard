"""Regenerate every synthetic dataset and model output in data/.

Usage (from the repository root):
    pip install -r requirements-analysis.txt
    python -m analysis.run_pipeline

Fully reproducible (fixed seeds). Writes only to data/; the dashboard reads these files.
"""
from __future__ import annotations

from pathlib import Path

from analysis import cleaning, models, synthetic

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    consumers = synthetic.generate_consumers()
    raw = synthetic.add_quality_issues(consumers)
    raw.to_csv(DATA / "consumer_survey_raw_synthetic.csv", index=False)
    clean, log = cleaning.clean(raw)
    clean.to_csv(DATA / "consumer_survey_clean.csv", index=False)
    log.to_csv(DATA / "data_cleaning_log.csv", index=False)
    print(f"consumer: raw {len(raw):,} -> clean {len(clean):,}")

    partners = synthetic.generate_partners()
    partners.to_csv(DATA / "partner_survey.csv", index=False)
    print(f"partners: {len(partners):,}")

    cm, cp = models.run_classification(clean)
    cm.to_csv(DATA / "classification_metrics.csv", index=False)
    cp.to_csv(DATA / "classification_predictions.csv", index=False)
    print(cm[cm.split == "test"][["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    ca, prof = models.run_clustering(clean)
    ca.to_csv(DATA / "cluster_assignments.csv", index=False)
    prof.to_csv(DATA / "cluster_profiles.csv", index=False)
    print(f"clusters: k={prof['k_selected'].iloc[0]} sizes={prof['size'].tolist()} {prof['silhouette_by_k'].iloc[0]}")

    rules = models.run_association(clean)
    rules.to_csv(DATA / "association_rules.csv", index=False)
    print(f"association rules: {len(rules):,} (max lift {rules['lift'].max():.2f})")

    rm, rp, rc = models.run_regression(clean)
    rm.to_csv(DATA / "regression_metrics.csv", index=False)
    rp.to_csv(DATA / "regression_predictions.csv", index=False)
    rc.to_csv(DATA / "regression_coefficients.csv", index=False)
    print(rm[rm.split == "test"][["model", "target", "mae", "rmse", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
