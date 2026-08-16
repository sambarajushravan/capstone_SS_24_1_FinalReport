"""
Generates the engineered IDP (Internal Developer Platform) telemetry dataset
described in the capstone prospectus (AILML-11-1-Capstone.docx).

No public dataset captures IDP adoption telemetry alongside deployment cycle
times at the squad level, so this script simulates one (~1,000 deployment
lifecycles) with realistic, non-trivial relationships between features and
the target so that PCA / regularized regression / clustering all have real
signal to recover. Missingness, duplicate rows, and outlier rows are
injected on purpose so the EDA notebook has real cleaning work to do.

Run: python generate_dataset.py
Output: idp_deployment_telemetry.csv (in this same directory)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1000


def generate_clean_data(n: int) -> pd.DataFrame:
    # Latent factors drive correlated feature clusters, mirroring how real
    # architectural complexity and platform adoption behave in practice.
    complexity = RNG.normal(0, 1, n)      # systemic structural complexity
    adoption = RNG.normal(0, 1, n)        # platform adoption / maturity

    # --- Team demographics ---
    team_size = RNG.integers(3, 16, n)                      # 3-15 engineers
    offshore_ratio = np.clip(RNG.beta(2, 3, n), 0, 1)        # skewed toward partial offshore

    # --- Architecture & debt (driven by `complexity`) ---
    microservice_deps = np.clip(
        np.round(6 + complexity * 5 + RNG.normal(0, 2, n)), 1, 45
    ).astype(int)
    helm_template_lines = np.clip(
        150 + microservice_deps * 35 + complexity * 120 + RNG.normal(0, 80, n), 20, None
    ).round(0)
    repo_code_churn = np.clip(
        80 + microservice_deps * 12 + team_size * 6 + complexity * 60 + RNG.normal(0, 70, n),
        5, None,
    ).round(0)

    # --- Platform adoption (driven by `adoption`, complexity makes manual overrides worse) ---
    idp_api_calls = np.clip(
        40 + adoption * 25 + RNG.normal(0, 12, n), 0, None
    ).round(0)
    gitops_prob = 1 / (1 + np.exp(-(adoption * 1.4 - 0.2)))
    gitops_auto_sync = RNG.binomial(1, gitops_prob)
    manual_kubectl_count = np.clip(
        15 - adoption * 6 + complexity * 3 - gitops_auto_sync * 4 + RNG.normal(0, 4, n),
        0, None,
    ).round(0)

    # --- Target: Deployment_Cycle_Time (hours, PR-open -> production-ready) ---
    base_hours = 18
    cycle_time = (
        base_hours
        + microservice_deps * 0.9
        + helm_template_lines * 0.01
        + repo_code_churn * 0.02
        + manual_kubectl_count * 1.6
        + offshore_ratio * 9
        + team_size * 0.4
        - idp_api_calls * 0.18
        - gitops_auto_sync * 7.5
        + RNG.gamma(shape=2.0, scale=3.0, size=n)  # right-skewed operational noise
    )
    cycle_time = np.clip(cycle_time, 2, None).round(2)

    df = pd.DataFrame(
        {
            "Team_Size": team_size,
            "Offshore_Ratio": offshore_ratio.round(3),
            "Microservice_Deps": microservice_deps,
            "Helm_Template_Lines": helm_template_lines,
            "Repo_Code_Churn": repo_code_churn,
            "IDP_API_Calls": idp_api_calls,
            "GitOps_Auto_Sync": gitops_auto_sync,
            "Manual_Kubectl_Count": manual_kubectl_count,
            "Deployment_Cycle_Time": cycle_time,
        }
    )
    return df


def inject_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # Missing values (MCAR) in a handful of columns, ~3-6% each.
    for col, frac in [
        ("Helm_Template_Lines", 0.05),
        ("Offshore_Ratio", 0.04),
        ("IDP_API_Calls", 0.03),
        ("Repo_Code_Churn", 0.04),
    ]:
        idx = RNG.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Duplicate rows (e.g. a telemetry export re-ingested twice).
    dup_idx = RNG.choice(n, size=15, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Genuine outlier incidents: major production incidents that blow out
    # cycle time well past the normal distribution (kept, not data errors).
    outlier_idx = RNG.choice(df.index, size=12, replace=False)
    df.loc[outlier_idx, "Deployment_Cycle_Time"] *= RNG.uniform(4, 7, size=12)
    df["Deployment_Cycle_Time"] = df["Deployment_Cycle_Time"].round(2)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    clean_df = generate_clean_data(N)
    final_df = inject_data_quality_issues(clean_df)
    out_path = __file__.rsplit("/", 1)[0] + "/idp_deployment_telemetry.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Wrote {len(final_df)} rows to {out_path}")
    print(final_df.isna().sum())
