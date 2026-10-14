"""
Linear regression for Objective 2.2.

Research question: which team and opponent characteristics explain how many
goals this team scores?

The dataset keeps exactly eight pre-match predictors. Model variations below
are experiments on that set. They do not add same-match information.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42
DATA_PATH = Path("Objective_2_2_Dataset.csv")
FIG_DIR = Path("figures_2_2")

PREDICTORS = [
    "X1_team_ppg",
    "X2_team_avg_goals_for",
    "X3_opponent_avg_goals_against",
    "X4_host_nation_flag",
    "X5_team_draw_rate",
    "X6_team_failed_to_score_rate",
    "X7_opponent_loss_rate",
    "X8_opponent_clean_sheet_rate",
]
TARGET = "goals_scored"

# Plain-language names for the report figures.
LABELS = {
    "X1_team_ppg": "Team points per game",
    "X2_team_avg_goals_for": "Team average goals for",
    "X3_opponent_avg_goals_against": "Opponent average goals against",
    "X4_host_nation_flag": "Host nation",
    "X5_team_draw_rate": "Team draw rate",
    "X6_team_failed_to_score_rate": "Team failed-to-score rate",
    "X7_opponent_loss_rate": "Opponent loss rate",
    "X8_opponent_clean_sheet_rate": "Opponent clean-sheet rate",
    TARGET: "Goals scored",
}

# Objective 2.1 used these penalties. They are the pre-specified comparison.
RIDGE_ALPHA_PRESET = 1.0
LASSO_ALPHA_PRESET = 0.1
RIDGE_ALPHAS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
LASSO_ALPHAS = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# Course multicollinearity screen: absolute correlation at least 0.7.
CORR_THRESHOLD = 0.7

SCORING = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def cv_metrics(model, x_model, y_model, cv):
    scores = cross_validate(model, x_model, y_model, cv=cv, scoring=SCORING)
    return {
        "CV_R2": float(np.mean(scores["test_r2"])),
        "CV_RMSE": float(-np.mean(scores["test_rmse"])),
        "CV_MAE": float(-np.mean(scores["test_mae"])),
    }


def ridge_model(alpha):
    return Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
    )


def lasso_model(alpha):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=20000)),
        ]
    )


def vif_table(x_data):
    x_with_constant = sm.add_constant(x_data)
    values = [
        variance_inflation_factor(x_with_constant.values, i)
        for i in range(1, x_with_constant.shape[1])
    ]
    table = pd.DataFrame({"Variable": list(x_data.columns), "VIF": values})
    return table.sort_values("VIF", ascending=False)


def iqr_outliers(frame, columns):
    rows = []
    for column in columns:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = (frame[column] < low) | (frame[column] > high)
        rows.append(
            {
                "variable": column,
                "lower_fence": low,
                "upper_fence": high,
                "iqr": iqr,
                "n_flagged": int(flagged.sum()),
            }
        )
    return pd.DataFrame(rows)


def correlation_reduction(corr, predictors, vifs):
    """Drop one variable from each predictor pair with |r| >= 0.7.

    Keep the variable with the stronger absolute correlation with goals
    scored. If those correlations differ by less than 0.02, the two
    variables tell almost the same story about the target, so drop the
    one with the higher VIF.
    """
    drop = set()
    pairs = []
    target_corr = corr[TARGET].abs()
    vif_map = dict(zip(vifs["Variable"], vifs["VIF"]))
    for i, left in enumerate(predictors):
        for right in predictors[i + 1 :]:
            r_value = float(corr.loc[left, right])
            if abs(r_value) < CORR_THRESHOLD:
                continue
            if abs(target_corr[left] - target_corr[right]) < 0.02:
                weaker = left if vif_map[left] > vif_map[right] else right
            elif target_corr[left] < target_corr[right]:
                weaker = left
            else:
                weaker = right
            drop.add(weaker)
            pairs.append((left, right, r_value, weaker))
    reduced = [name for name in predictors if name not in drop]
    return reduced, pairs


def standardised_ols(y_data, x_data):
    scaled = StandardScaler().fit_transform(x_data)
    return sm.OLS(y_data, sm.add_constant(scaled)).fit()


def main():
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    df = pd.read_csv(DATA_PATH)
    x_data = df[PREDICTORS]
    y_data = df[TARGET]
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ----- EDA -----
    summary = pd.DataFrame(
        {
            "mean": df[PREDICTORS + [TARGET]].mean(),
            "median": df[PREDICTORS + [TARGET]].median(),
            "std": df[PREDICTORS + [TARGET]].std(),
            "IQR": (
                df[PREDICTORS + [TARGET]].quantile(0.75)
                - df[PREDICTORS + [TARGET]].quantile(0.25)
            ),
            "min": df[PREDICTORS + [TARGET]].min(),
            "max": df[PREDICTORS + [TARGET]].max(),
            "skew": df[PREDICTORS + [TARGET]].skew(),
        }
    ).round(4)
    summary.to_csv(FIG_DIR / "summary_statistics.csv")

    outliers = iqr_outliers(df, [TARGET] + PREDICTORS)
    outliers.to_csv(FIG_DIR / "iqr_outliers.csv", index=False)

    zero_form = int((x_data.sum(axis=1) == 0).sum())

    columns = [TARGET] + PREDICTORS
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
    for ax, column in zip(axes.ravel(), columns):
        ax.hist(df[column], bins=15, color="#3d5a80", edgecolor="white")
        ax.set_title(LABELS[column], fontsize=9)
    fig.suptitle("Objective 2.2 — distributions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histograms.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
    for ax, column in zip(axes.ravel(), columns):
        ax.boxplot(df[column], vert=True, widths=0.55)
        ax.set_title(LABELS[column], fontsize=9)
        ax.set_xticks([])
    fig.suptitle("Objective 2.2 — boxplots")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots.png", dpi=160)
    plt.close(fig)

    corr = df[PREDICTORS + [TARGET]].corr()
    corr.round(4).to_csv(FIG_DIR / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(8.4, 7))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    tick_labels = [LABELS[name] for name in corr.columns]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation of predictors and goals scored")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    pearson_rows = []
    for name in PREDICTORS:
        r_value, p_value = stats.pearsonr(df[name], y_data)
        pearson_rows.append({"Variable": name, "r": r_value, "p_value": p_value})
    pearson = pd.DataFrame(pearson_rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
    pearson.to_csv(FIG_DIR / "pearson_tests.csv", index=False)

    vifs = vif_table(x_data)
    vifs.to_csv(FIG_DIR / "vif.csv", index=False)
    reduced_predictors, dropped_pairs = correlation_reduction(corr, PREDICTORS, vifs)
    reduced_vifs = vif_table(df[reduced_predictors])
    reduced_vifs.to_csv(FIG_DIR / "vif_reduced.csv", index=False)

    # ----- Inferential test: host nation vs other teams -----
    host = y_data[df["X4_host_nation_flag"] == 1]
    other = y_data[df["X4_host_nation_flag"] == 0]
    t_stat, p_value = stats.ttest_ind(host, other, equal_var=False)
    pooled_sd = np.sqrt(
        ((len(host) - 1) * host.std(ddof=1) ** 2 + (len(other) - 1) * other.std(ddof=1) ** 2)
        / (len(host) + len(other) - 2)
    )
    cohens_d = (host.mean() - other.mean()) / pooled_sd

    # ----- Full and reduced OLS -----
    full_ols = sm.OLS(y_data, sm.add_constant(x_data)).fit()
    reduced_ols = sm.OLS(y_data, sm.add_constant(df[reduced_predictors])).fit()
    full_std = standardised_ols(y_data, x_data)
    coef_table = pd.DataFrame(
        {
            "Variable": PREDICTORS,
            "coef": full_ols.params[PREDICTORS].values,
            "std_err": full_ols.bse[PREDICTORS].values,
            "p_value": full_ols.pvalues[PREDICTORS].values,
            "std_coef": full_std.params.values[1:],
            "std_p_value": full_std.pvalues.values[1:],
        }
    )
    coef_table.to_csv(FIG_DIR / "ols_coefficients.csv", index=False)

    # ----- Penalty search on the same 5 folds -----
    ridge_search = []
    for alpha in RIDGE_ALPHAS:
        row = cv_metrics(ridge_model(alpha), x_data, y_data, kf)
        row["alpha"] = alpha
        ridge_search.append(row)
    ridge_search = pd.DataFrame(ridge_search).sort_values("CV_RMSE")
    best_ridge_alpha = float(ridge_search.iloc[0]["alpha"])

    lasso_search = []
    for alpha in LASSO_ALPHAS:
        row = cv_metrics(lasso_model(alpha), x_data, y_data, kf)
        row["alpha"] = alpha
        lasso_search.append(row)
    lasso_search = pd.DataFrame(lasso_search).sort_values("CV_RMSE")
    best_lasso_alpha = float(lasso_search.iloc[0]["alpha"])
    pd.concat(
        [
            ridge_search.assign(model="Ridge"),
            lasso_search.assign(model="Lasso"),
        ]
    ).to_csv(FIG_DIR / "alpha_search.csv", index=False)

    # Yeo-Johnson on the continuous predictors only. The host flag stays binary.
    continuous = [name for name in PREDICTORS if name != "X4_host_nation_flag"]
    yeo = Pipeline(
        [
            (
                "transform",
                ColumnTransformer(
                    [
                        ("yj", PowerTransformer(method="yeo-johnson"), continuous),
                        ("host", "passthrough", ["X4_host_nation_flag"]),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yeo_scores = cv_metrics(yeo, x_data, y_data, kf)
    yeo_warnings = len(caught)

    # ----- Main comparison, same folds as Objective 2.1 -----
    comparison_specs = [
        ("Mean baseline", DummyRegressor(strategy="mean"), x_data),
        ("Full OLS", LinearRegression(), x_data),
        ("Reduced OLS", LinearRegression(), df[reduced_predictors]),
        ("Ridge (pre-set α = 1)", ridge_model(RIDGE_ALPHA_PRESET), x_data),
        ("Lasso (pre-set α = 0.1)", lasso_model(LASSO_ALPHA_PRESET), x_data),
        (f"Ridge (tuned α = {best_ridge_alpha:g})", ridge_model(best_ridge_alpha), x_data),
        (f"Lasso (tuned α = {best_lasso_alpha:g})", lasso_model(best_lasso_alpha), x_data),
        ("OLS after Yeo-Johnson", yeo, x_data),
    ]
    comparison_rows = []
    for name, model, x_model in comparison_specs:
        row = cv_metrics(model, x_model, y_data, kf)
        row["Model"] = name
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values("CV_RMSE")
    comparison.to_csv(FIG_DIR / "model_comparison.csv", index=False)

    # Four-model chart for the report: full, reduced, best ridge, best lasso, plus baseline.
    report_names = [
        "Mean baseline",
        "Full OLS",
        "Reduced OLS",
        f"Ridge (tuned α = {best_ridge_alpha:g})",
        f"Lasso (tuned α = {best_lasso_alpha:g})",
    ]
    report_chart = comparison[comparison["Model"].isin(report_names)].sort_values(
        "CV_RMSE", ascending=False
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [
        "#0f6e56" if "Ridge (tuned" in name else "#8d99ae" if name == "Mean baseline" else "#3d5a80"
        for name in report_chart["Model"]
    ]
    ax.barh(report_chart["Model"], report_chart["CV_RMSE"], color=colors)
    ax.set_xlabel("Cross-validated RMSE (goals). Axis starts at 1.30 so small gaps stay visible.")
    ax.set_title("Model comparison — lower error is better")
    for i, rmse in enumerate(report_chart["CV_RMSE"]):
        ax.text(rmse + 0.0008, i, f"{rmse:.3f}", va="center", fontsize=9)
    ax.set_xlim(1.30, report_chart["CV_RMSE"].max() + 0.012)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison_rmse.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Penalty path: shows the search, not only the winner.
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ordered = ridge_search.sort_values("alpha")
    baseline_rmse = float(comparison.loc[comparison["Model"] == "Mean baseline", "CV_RMSE"].iloc[0])
    ax.plot(ordered["alpha"], ordered["CV_RMSE"], marker="o", color="#0f6e56", label="Ridge")
    ax.axhline(baseline_rmse, color="#8d99ae", linestyle="--", label="Mean baseline")
    ax.set_xscale("log")
    ax.set_ylim(1.35, 1.40)
    ax.set_xlabel("Ridge penalty (α)")
    ax.set_ylabel("Cross-validated RMSE (goals)")
    ax.set_title("Penalty search — the gain over the baseline stays under 0.03 goals")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ridge_alpha_path.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Story figure: which characteristics are associated with goals scored.
    story = coef_table.sort_values("std_coef")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#b42318" if value < 0 else "#0f6e56" for value in story["std_coef"]]
    story_labels = []
    for name, p_val in zip(story["Variable"], story["p_value"]):
        mark = ""
        if p_val < 0.05:
            mark = " *"
        elif p_val < 0.10:
            mark = " †"
        story_labels.append(LABELS[name] + mark)
    ax.barh(story_labels, story["std_coef"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Change in goals for a 1 SD higher predictor   (* p < 0.05,  † p < 0.10)")
    ax.set_title("What is associated with goals scored")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "standardized_coefficients.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ----- Diagnostics on full OLS, matching Objective 2.1 -----
    fitted = full_ols.fittedvalues
    resid = full_ols.resid
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(fitted, resid, alpha=0.65, color="#3d5a80", edgecolor="none")
    ax.axhline(0, color="#b42318", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted goals scored")
    ax.set_ylabel("Residual")
    ax.set_title("Full OLS — residuals against fitted values")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals_vs_fitted.png", dpi=160)
    plt.close(fig)

    fig = qqplot(resid, line="45", fit=True)
    plt.title("Full OLS — normal Q-Q plot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "qq_plot.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(y_data, fitted, alpha=0.65, color="#3d5a80", edgecolor="none")
    limit = max(y_data.max(), fitted.max())
    ax.plot([0, limit], [0, limit], color="#b42318", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual goals scored")
    ax.set_ylabel("Fitted goals scored")
    ax.set_title("Full OLS — actual against fitted (training sample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    shapiro_w, shapiro_p = stats.shapiro(resid)
    breusch = sm.stats.diagnostic.het_breuschpagan(resid, full_ols.model.exog)
    cooks = full_ols.get_influence().cooks_distance[0]
    cook_cut = 4 / len(df)
    high_cook = cooks > cook_cut
    stable = sm.OLS(
        y_data.loc[~high_cook],
        sm.add_constant(x_data.loc[~high_cook]),
    ).fit()

    # Lasso at the moderate penalty that removes the collinear pair.
    lasso_fit = lasso_model(0.05).fit(x_data, y_data)
    lasso_coef = pd.Series(lasso_fit.named_steps["model"].coef_, index=PREDICTORS)

    # Sensitivity: two rows from the same match are not independent.
    grouped_cv = list(GroupKFold(n_splits=5).split(x_data, y_data, df["match_id"]))
    grouped_rows = []
    for name, model, x_model in [
        ("Full OLS", LinearRegression(), x_data),
        ("Reduced OLS", LinearRegression(), df[reduced_predictors]),
        ("Ridge α = 1", ridge_model(RIDGE_ALPHA_PRESET), x_data),
        (f"Ridge α = {best_ridge_alpha:g}", ridge_model(best_ridge_alpha), x_data),
    ]:
        row = cv_metrics(model, x_model, y_data, grouped_cv)
        row["Model"] = name
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows).sort_values("CV_RMSE")
    grouped.to_csv(FIG_DIR / "grouped_cv.csv", index=False)

    # ----- Console record used to write the report -----
    print("n", len(df), "matches", df["match_id"].nunique(), "zero-form rows", zero_form)
    print("\nSUMMARY\n", summary)
    print("\nIQR\n", outliers.round(3).to_string(index=False))
    print("\nPEARSON\n", pearson.round(4).to_string(index=False))
    print("\nVIF\n", vifs.round(3).to_string(index=False))
    print("dropped pairs", dropped_pairs)
    print("reduced predictors", reduced_predictors)
    print("reduced VIF\n", reduced_vifs.round(3).to_string(index=False))
    print(
        f"Welch t={t_stat:.4f} p={p_value:.4f} d={cohens_d:.4f} "
        f"host n={len(host)} mean={host.mean():.4f} sd={host.std(ddof=1):.4f} "
        f"other n={len(other)} mean={other.mean():.4f} sd={other.std(ddof=1):.4f}"
    )
    print(
        f"Full OLS R2={full_ols.rsquared:.4f} adj={full_ols.rsquared_adj:.4f} "
        f"F={full_ols.fvalue:.4f} p={full_ols.f_pvalue:.6f}"
    )
    print(
        f"Reduced OLS R2={reduced_ols.rsquared:.4f} adj={reduced_ols.rsquared_adj:.4f} "
        f"F={reduced_ols.fvalue:.4f} p={reduced_ols.f_pvalue:.6f}"
    )
    print("\nCOEFS\n", coef_table.round(4).to_string(index=False))
    print("\nRIDGE SEARCH\n", ridge_search.round(4).to_string(index=False))
    print("\nLASSO SEARCH\n", lasso_search.round(4).to_string(index=False))
    print("best ridge", best_ridge_alpha, "best lasso", best_lasso_alpha)
    print(f"Yeo-Johnson warnings={yeo_warnings}", {k: round(v, 4) for k, v in yeo_scores.items()})
    print("\nCOMPARISON\n", comparison.round(4).to_string(index=False))
    print(
        f"Shapiro W={shapiro_w:.4f} p={shapiro_p:.6g} "
        f"BP lm={breusch[0]:.4f} p={breusch[1]:.4f}"
    )
    print(
        f"Cook max={cooks.max():.4f} n>{cook_cut:.4f} = {int(high_cook.sum())}"
    )
    print("stable std-relevant raw coefs")
    for name in ["X5_team_draw_rate", "X7_opponent_loss_rate", "X8_opponent_clean_sheet_rate"]:
        print(
            name,
            f"full {full_ols.params[name]:.3f} p={full_ols.pvalues[name]:.4f}",
            f"stable {stable.params[name]:.3f} p={stable.pvalues[name]:.4f}",
        )
    print("lasso 0.05\n", lasso_coef.round(4))
    print("\nGROUPED\n", grouped.round(4).to_string(index=False))
    draw = df.groupby(
        pd.cut(df["X5_team_draw_rate"], [-0.01, 0, 0.5, 1.01], include_lowest=True),
        observed=True,
    )[TARGET].agg(["mean", "count"])
    print("\nDRAW BINS\n", draw.round(3))


if __name__ == "__main__":
    main()
