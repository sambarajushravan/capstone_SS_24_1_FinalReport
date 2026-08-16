### Quantifying the Operational ROI and Engineering Bottlenecks of Internal Developer Platforms (IDP) Using Predictive Telemetry

**Author**

#### Executive summary
Enterprises invest heavily in Internal Developer Platforms (IDPs) to speed up software delivery, but leadership rarely has hard evidence of whether that investment is paying off or where bottlenecks remain. This project uses engineering infrastructure telemetry — platform adoption behavior, architectural complexity/technical debt, and team demographics — to predict deployment cycle time, identify the strongest real drivers of delivery friction, and segment engineering squads for targeted intervention. A tuned Lasso regression model explains 93% of the variance in deployment cycle time and beats a naive baseline by 74%; a single PCA component captures the shared "systemic structural complexity" behind five correlated architecture metrics; and K-Means splits squads into two clearly differentiated cohorts — platform adopters running 73% faster than manual-override holdouts — giving leadership a concrete, data-backed answer to all three parts of the original research question.

#### Rationale
IDP spend is usually justified on faith, not evidence — engineering leadership signs off on the tooling budget without a reliable way to measure whether it actually shortens delivery, or which teams are still struggling despite it. That evidence gap pushes organizations toward one of two bad defaults: keep funding a platform nobody's really using, or roll out a blanket adoption mandate that ignores how differently each squad's bottlenecks actually look. Framing this as a measurement problem — is the platform paying for itself, what specifically is slowing squads down, and who needs help versus a mandate — turns "is this working?" from a hunch into a resourcing decision leadership can act on.

#### Research Question
Does infrastructure telemetry (platform usage, architectural complexity, team composition) carry enough signal to predict deployment cycle time, pin down the technical-debt factors driving it, and cluster engineering squads into distinct IDP-adoption profiles?

#### Data Sources
IDP adoption telemetry paired with squad-level deployment cycle times isn't publicly available anywhere, so this project works from a **purpose-built simulated dataset** (~1,000 deployment lifecycles) generated to mirror the schema laid out in the capstone prospectus. [`data/generate_dataset.py`](data/generate_dataset.py) documents the generating process — cycle time responds to platform usage (down) and architectural complexity/manual overrides (up) — and deliberately injects missing values, duplicate rows, and outlier "incident" deployments so the cleaning work in the notebook isn't trivial. Output lands in `data/idp_deployment_telemetry.csv`.

| Feature | Type | Description |
|---|---|---|
| `Team_Size` | Discrete | Active contributors on the engineering squad |
| `Offshore_Ratio` | Continuous (0-1) | Share of the team distributed across time zones |
| `Microservice_Deps` | Discrete | Downstream architectural dependencies |
| `Helm_Template_Lines` | Continuous | Size/complexity of infrastructure-as-code manifests |
| `Repo_Code_Churn` | Continuous | Lines of code modified/deleted in the deployment window |
| `IDP_API_Calls` | Continuous | Developer interactions with the platform API per sprint |
| `GitOps_Auto_Sync` | Binary | Whether automated GitOps deployment is enabled |
| `Manual_Kubectl_Count` | Continuous | Direct cluster bypass commands executed by developers |
| `Deployment_Cycle_Time` | Continuous (hours) | **Target.** PR-open to production-ready elapsed time |

#### Methodology
1. **Data cleaning** — median imputation for MCAR missingness (~3-5% in four columns), de-duplication of re-ingested rows, and validity checks for structurally impossible values.
2. **Outlier analysis** — IQR-based screening across all continuous features; genuine high-cycle-time incident outliers were retained (they represent real tail risk), while impossible values would have been removed had any existed.
3. **Feature engineering** — derived `Manual_Override_Ratio`, `Deps_per_Engineer`, `Churn_per_Dependency`, and `Offshore_Heavy` to normalize raw counts into more interpretable ratios/flags.
4. **EDA** — distribution, correlation, and categorical-vs-continuous visualizations (Matplotlib, Seaborn, Plotly) to test the proposal's hypotheses about adoption vs. complexity effects.
5. **Baseline & comparison models** — a Lasso-regularized linear regression (per the original proposal, chosen because its L1 penalty doubles as an objective feature-selection step), compared against a dummy mean-prediction baseline, a `GridSearchCV`-tuned Ridge regression, and a `GridSearchCV`-tuned Random Forest, all evaluated on the same held-out test set and cross-validated for robustness.
6. **Dimensionality reduction** — PCA collapses five correlated architecture/technical-debt features into a single "systemic structural complexity" score.
7. **Squad segmentation** — K-Means (elbow method + silhouette analysis to select K) segments engineering squads into adoption cohorts using platform-adoption behavior and the PCA complexity score.

#### Results
- The **Lasso regression** achieved **MAE = 5.30 hours** and **R² = 0.932** on held-out test data, a **73.6% MAE improvement** over a dummy mean-prediction baseline (MAE = 20.07 hours). MAE was chosen as the primary metric because it's directly interpretable to a non-technical audience ("predictions are off by ~5 hours on average"), while RMSE would overweight the deliberately-retained incident outliers and R² alone wouldn't communicate a practical margin of error.
- Lasso's coefficient shrinkage confirms the proposal's hypothesis: `Manual_Kubectl_Count` and architectural-complexity features (`Microservice_Deps`, `Helm_Template_Lines`, `Repo_Code_Churn`) push cycle time up, while `IDP_API_Calls` and `GitOps_Auto_Sync` pull it down. Three engineered/demographic features (`Manual_Override_Ratio`, `Deps_per_Engineer`, `Offshore_Heavy`) were shrunk to exactly zero, indicating team demographics matter far less than platform adoption and technical debt.
- **Model comparison:** despite tuning with `GridSearchCV` and 5-fold cross-validation, neither Ridge (MAE = 5.62h, R² = 0.925) nor Random Forest (MAE = 6.88h, R² = 0.770) beat the simpler Lasso baseline — good news for interpretability, since the best-performing model is also the easiest to explain to a non-technical audience. Random Forest's feature importances independently confirm the same top drivers Lasso identified.
- **PCA:** a single component (`PC1`) explains **64.4%** of the variance across the five complexity features and correlates with `Deployment_Cycle_Time` at **r = 0.37**, confirming technical debt behaves like one underlying construct rather than five independent signals.
- **K-Means:** silhouette analysis selects **K = 2**, splitting squads into "High-Velocity Platform Adopters" (462 deployments, ~54 IDP API calls/sprint, 94% GitOps adoption, **42.6-hour** average cycle time) vs. "Legacy Scripting Holdouts" (538 deployments, ~26 IDP API calls/sprint, 3% GitOps adoption, **73.6-hour** average cycle time — **73% slower**).

#### Next steps
- Validate the engineered data-generating assumptions against real IDP/CI telemetry once available, since this project currently relies on a simulated dataset built to match the original proposal's schema.
- Extend the squad segmentation with a time dimension (e.g., track whether squads migrate between clusters after a platform intervention) to measure causal impact rather than a single cross-sectional snapshot.
- Explore gradient-boosted trees (e.g., XGBoost/LightGBM) and SHAP-based interpretation for a deeper, non-linear feature-attribution story if a future dataset shows more nonlinear structure than this one did.

#### Outline of project

- [Final Report Notebook](capstone_final_report.ipynb) — EDA, data cleaning, feature engineering, PCA, K-Means segmentation, and model comparison (Lasso/Ridge/Random Forest)
- [Dataset generation script](data/generate_dataset.py)
- [Original capstone prospectus](AILML-11-1-Capstone%20(1).docx)


##### Contact and Further Information
sambaraju.shravan@gmail.com | [GitHub repository](https://github.com/sambarajushravan/capstone_SS_24_1_FinalReport)
