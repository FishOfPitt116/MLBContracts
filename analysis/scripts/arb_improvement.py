"""
Phase 0 analysis for the arb model improvement plan.
Validates assumptions behind each proposed change before implementation.
Outputs graphs to analysis/graphs/arb_improvement/ and prints structured results.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

from models.arb import config
from models.arb.features import load_and_filter_data, get_features_and_target, add_arb_year_column
from models.arb.model import ArbModel
from models.preprocessing import normalize_service_time, load_contracts

GRAPH_DIR = "analysis/graphs/arb_improvement"
PITCHER_POSITIONS = config.PITCHER_POSITIONS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_train_test_split(df, player_type):
    X, y = get_features_and_target(df, player_type=player_type)
    X_bins = X.copy()
    X_bins["strat_bin"] = pd.cut(
        X_bins["service_time"], bins=[0, 4, 5, 10], labels=["1", "2", "3"]
    )
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=X_bins["strat_bin"],
    )


def load_arb_data_with_tiers():
    """Load filtered single-year arb contracts with normalized ST and arb_year."""
    df = load_contracts()
    mask = (
        (df["contract_type"] == "arb")
        & (df["duration"] == 1)
        & (df["service_time"].notna())
    )
    df = df[mask].copy()
    df["service_time_norm"] = df["service_time"].apply(normalize_service_time)
    df["arb_year"] = df["service_time_norm"].apply(config.get_tier_from_service_time)
    df["player_type"] = df["position"].apply(
        lambda p: "pitcher" if p in PITCHER_POSITIONS else "batter"
    )
    return df


def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Analysis 1: MAPE Distribution Baseline
# ---------------------------------------------------------------------------

def analysis_1_mape_baseline():
    section("Analysis 1: MAPE Distribution Baseline")

    results = {}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("MAPE Distribution by Player Type and Arb Tier (Current Model)", fontsize=14)

    for row_idx, player_type in enumerate(["pitcher", "batter"]):
        df = load_and_filter_data(player_type=player_type)
        X_train, X_test, y_train, y_test = get_train_test_split(df, player_type)
        model = ArbModel.load(player_type=player_type)
        preds = model.predict(X_test)

        mape_all = np.abs(y_test.values - preds) / y_test.values
        X_test_tiers = add_arb_year_column(X_test)

        results[player_type] = {"overall": {}, "by_tier": {}}

        print(f"\n{player_type.upper()} — Overall MAPE percentiles:")
        row = {}
        for p in [50, 75, 90, 95]:
            val = np.percentile(mape_all, p) * 100
            row[p] = val
            print(f"  {p}th pct: {val:.1f}%")
        results[player_type]["overall"] = row

        # Overall histogram
        axes[row_idx, 0].hist(mape_all * 100, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[row_idx, 0].axvline(20, color="red", linestyle="--", label="20% target")
        axes[row_idx, 0].axvline(np.percentile(mape_all * 100, 90), color="orange",
                                  linestyle="--", label=f"90th pct ({np.percentile(mape_all*100,90):.0f}%)")
        axes[row_idx, 0].set_title(f"{player_type.capitalize()} — All Tiers (n={len(mape_all)})")
        axes[row_idx, 0].set_xlabel("MAPE (%)")
        axes[row_idx, 0].legend(fontsize=8)

        print(f"\n{player_type.upper()} — MAPE by tier:")
        for col_idx, tier in enumerate([1, 2, 3]):
            mask = X_test_tiers["arb_year"] == tier
            if mask.sum() == 0:
                continue
            mape_tier = mape_all[mask.values]
            tier_pcts = {}
            for p in [50, 75, 90, 95]:
                val = np.percentile(mape_tier, p) * 100
                tier_pcts[p] = val
            results[player_type]["by_tier"][tier] = {
                "n": int(mask.sum()),
                "percentiles": tier_pcts,
                "pct_within_20": float(np.mean(mape_tier <= 0.20) * 100),
                "pct_within_15": float(np.mean(mape_tier <= 0.15) * 100),
            }
            print(f"  Year {tier} (n={mask.sum()}): "
                  f"50th={tier_pcts[50]:.1f}%  75th={tier_pcts[75]:.1f}%  "
                  f"90th={tier_pcts[90]:.1f}%  "
                  f"| within 20%: {results[player_type]['by_tier'][tier]['pct_within_20']:.1f}%")

            ax = axes[row_idx, col_idx + 1]
            ax.hist(mape_tier * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(20, color="red", linestyle="--", label="20% target")
            ax.axvline(np.percentile(mape_tier * 100, 90), color="orange",
                       linestyle="--", label=f"90th ({np.percentile(mape_tier*100,90):.0f}%)")
            ax.set_title(f"{player_type.capitalize()} Year {tier} (n={mask.sum()})")
            ax.set_xlabel("MAPE (%)")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/1_mape_baseline.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {GRAPH_DIR}/1_mape_baseline.png")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Residual Analysis
# ---------------------------------------------------------------------------

def analysis_2_residuals():
    section("Analysis 2: Residual Analysis")

    all_results = {}

    for player_type in ["pitcher", "batter"]:
        df = load_and_filter_data(player_type=player_type)
        X_train, X_test, y_train, y_test = get_train_test_split(df, player_type)
        model = ArbModel.load(player_type=player_type)
        preds = model.predict(X_test)

        residuals = preds - y_test.values
        abs_residuals = np.abs(residuals)

        X_test_tiers = add_arb_year_column(X_test).reset_index(drop=True)
        y_vals = y_test.values

        print(f"\n{player_type.upper()} residual correlations:")

        corrs = {}

        # contract_year
        r_yr, p_yr = stats.pearsonr(X_test["contract_year"], residuals)
        corrs["contract_year"] = (r_yr, p_yr)
        print(f"  |residual| ~ contract_year:  r={r_yr:+.3f}  p={p_yr:.4f}")

        # WAR momentum — drop NaNs before correlating
        if player_type == "pitcher":
            raw_momentum = X_test["pit_war_1y"] - X_test["pit_war_3y"] / 3
        else:
            raw_momentum = X_test["bat_war_1y"] - X_test["bat_war_3y"] / 3
        momentum = raw_momentum.reset_index(drop=True)
        valid_mom = ~momentum.isna()
        if valid_mom.sum() > 10:
            r_mom, p_mom = stats.pearsonr(momentum[valid_mom], residuals[valid_mom])
        else:
            r_mom, p_mom = float("nan"), float("nan")
        corrs["war_momentum"] = (r_mom, p_mom)
        print(f"  residual ~ WAR momentum:      r={r_mom:+.3f}  p={p_mom:.4f}  (n={valid_mom.sum()})")

        # service_time x WAR interaction — drop NaNs
        if player_type == "pitcher":
            raw_interaction = X_test["service_time"] * X_test["pit_war_3y"]
        else:
            raw_interaction = X_test["service_time"] * X_test["bat_war_3y"]
        interaction = raw_interaction.reset_index(drop=True)
        valid_int = ~interaction.isna()
        if valid_int.sum() > 10:
            r_int, p_int = stats.pearsonr(interaction[valid_int], residuals[valid_int])
        else:
            r_int, p_int = float("nan"), float("nan")
        corrs["st_war_interaction"] = (r_int, p_int)
        print(f"  residual ~ ST×WAR interaction: r={r_int:+.3f}  p={p_int:.4f}  (n={valid_int.sum()})")

        # heteroskedasticity: |residual| vs predicted value
        r_het, p_het = stats.pearsonr(preds, abs_residuals)
        corrs["heteroskedasticity"] = (r_het, p_het)
        print(f"  |residual| ~ predicted value:  r={r_het:+.3f}  p={p_het:.4f}  "
              f"({'significant heteroskedasticity' if p_het < 0.05 else 'no clear pattern'})")

        all_results[player_type] = corrs

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f"{player_type.capitalize()} — Residual Analysis", fontsize=14)

        # contract_year vs residual
        years = X_test["contract_year"].values
        axes[0, 0].scatter(years, residuals, alpha=0.3, s=15, color="steelblue")
        yr_means = pd.Series(residuals).groupby(years).mean()
        axes[0, 0].plot(yr_means.index, yr_means.values, "r-o", markersize=4, label="year mean")
        axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0, 0].set_xlabel("Contract Year")
        axes[0, 0].set_ylabel("Residual ($M)")
        axes[0, 0].set_title(f"Residual vs Year (r={r_yr:+.3f})")
        axes[0, 0].legend(fontsize=8)

        # WAR momentum vs residual (use valid rows only)
        axes[0, 1].scatter(momentum[valid_mom], residuals[valid_mom], alpha=0.3, s=15, color="steelblue")
        axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
        if valid_mom.sum() > 10 and not np.isnan(r_mom):
            m, b = np.polyfit(momentum[valid_mom], residuals[valid_mom], 1)
            x_line = np.linspace(momentum[valid_mom].min(), momentum[valid_mom].max(), 100)
            axes[0, 1].plot(x_line, m * x_line + b, "r-", linewidth=1.5)
        axes[0, 1].set_xlabel("WAR Momentum (1y WAR − 3y WAR/3)")
        axes[0, 1].set_ylabel("Residual ($M)")
        axes[0, 1].set_title(f"Residual vs WAR Momentum (r={r_mom:+.3f})")

        # Predicted vs |residual| (heteroskedasticity)
        axes[1, 0].scatter(preds, abs_residuals, alpha=0.3, s=15, color="steelblue")
        m2, b2 = np.polyfit(preds, abs_residuals, 1)
        x_line2 = np.linspace(preds.min(), preds.max(), 100)
        axes[1, 0].plot(x_line2, m2 * x_line2 + b2, "r-", linewidth=1.5)
        axes[1, 0].set_xlabel("Predicted Salary ($M)")
        axes[1, 0].set_ylabel("|Residual| ($M)")
        axes[1, 0].set_title(f"|Residual| vs Predicted (r={r_het:+.3f})")

        # Residual by tier
        tier_residuals = []
        tier_labels = []
        for tier in [1, 2, 3]:
            mask = X_test_tiers["arb_year"] == tier
            tier_residuals.append(abs_residuals[mask.values])
            tier_labels.append(f"Year {tier}\n(n={mask.sum()})")
        bp = axes[1, 1].boxplot(tier_residuals, labels=tier_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        axes[1, 1].set_ylabel("|Residual| ($M)")
        axes[1, 1].set_title("|Residual| by Arb Tier")
        axes[1, 1].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="$1M threshold")
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{GRAPH_DIR}/2_residuals_{player_type}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  → Saved: {GRAPH_DIR}/2_residuals_{player_type}.png")

    return all_results


# ---------------------------------------------------------------------------
# Analysis 3: Within-Tier Variance
# ---------------------------------------------------------------------------

def analysis_3_within_tier_variance():
    section("Analysis 3: Within-Tier Variance (tests tier-specific model hypothesis)")

    df = load_arb_data_with_tiers()

    results = {}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Salary Distributions: Full Dataset vs. Within Each Tier", fontsize=13)

    for row_idx, player_type in enumerate(["pitcher", "batter"]):
        subset = df[df["player_type"] == player_type]["value"]

        overall_cv = subset.std() / subset.mean()
        print(f"\n{player_type.upper()}:")
        print(f"  Full dataset:  mean=${subset.mean():.2f}M  std=${subset.std():.2f}M  "
              f"CV={overall_cv:.3f}  n={len(subset)}")

        results[player_type] = {"overall_cv": overall_cv, "tiers": {}}

        axes[row_idx, 0].hist(subset, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[row_idx, 0].set_title(f"{player_type.capitalize()} — All Tiers (CV={overall_cv:.2f})")
        axes[row_idx, 0].set_xlabel("Salary ($M)")

        for col_idx, tier in enumerate([1, 2, 3]):
            tier_vals = df[
                (df["player_type"] == player_type) & (df["arb_year"] == tier)
            ]["value"]

            if len(tier_vals) == 0:
                continue

            tier_cv = tier_vals.std() / tier_vals.mean()
            cv_reduction = (overall_cv - tier_cv) / overall_cv * 100
            results[player_type]["tiers"][tier] = {
                "n": len(tier_vals),
                "mean": float(tier_vals.mean()),
                "std": float(tier_vals.std()),
                "cv": float(tier_cv),
                "cv_reduction_pct": float(cv_reduction),
            }
            print(f"  Year {tier} (n={len(tier_vals)}):  mean=${tier_vals.mean():.2f}M  "
                  f"std=${tier_vals.std():.2f}M  CV={tier_cv:.3f}  "
                  f"(CV reduction vs full: {cv_reduction:.1f}%)")

            ax = axes[row_idx, col_idx + 1]
            ax.hist(tier_vals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax.set_title(f"{player_type.capitalize()} Year {tier} (n={len(tier_vals)}, CV={tier_cv:.2f})")
            ax.set_xlabel("Salary ($M)")

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/3_within_tier_variance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {GRAPH_DIR}/3_within_tier_variance.png")

    # Decision rule from plan: proceed with tier models if CV drops >30%
    print("\nDecision (threshold: CV drop > 30%):")
    for pt in ["pitcher", "batter"]:
        drops = [v["cv_reduction_pct"] for v in results[pt]["tiers"].values()]
        avg_drop = np.mean(drops)
        print(f"  {pt}: avg CV drop = {avg_drop:.1f}% → "
              f"{'PROCEED with tier models' if avg_drop > 30 else 'Low priority'}")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Market Anchor Correlation
# ---------------------------------------------------------------------------

def analysis_4_market_anchors():
    section("Analysis 4: Market Anchor Correlation (tests market-reset hypothesis)")

    df = load_arb_data_with_tiers()

    # Build prior-year tier stats (time-aware, no leakage)
    year_tier_stats = (
        df.groupby(["contract_year", "arb_year", "player_type"])["value"]
        .agg(
            market_prior_mean="mean",
            market_prior_p75=lambda x: x.quantile(0.75),
            market_prior_p90=lambda x: x.quantile(0.90),
            market_prior_max="max",
        )
        .reset_index()
    )
    year_tier_stats["contract_year"] += 1  # shift: year Y stats become anchor for Y+1

    df_anchored = df.merge(year_tier_stats, on=["contract_year", "arb_year", "player_type"], how="left")
    df_anchored = df_anchored.dropna(subset=["market_prior_p90"])

    anchor_cols = ["market_prior_mean", "market_prior_p75", "market_prior_p90", "market_prior_max"]
    results = {}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Market Anchor Correlation with Current Salary", fontsize=13)

    for row_idx, player_type in enumerate(["pitcher", "batter"]):
        subset = df_anchored[df_anchored["player_type"] == player_type].copy()
        print(f"\n{player_type.upper()} (n={len(subset)}):")

        # Full correlations
        print("  Raw correlation with value:")
        for col in anchor_cols:
            r, p = stats.pearsonr(subset[col], subset["value"])
            print(f"    {col}: r={r:+.3f}  p={p:.4e}")

        # Partial correlation: residuals of WAR-only model vs anchor
        if player_type == "pitcher":
            war_col = "pit_war_3y"
        else:
            war_col = "bat_war_3y"

        war_subset = subset.dropna(subset=[war_col])
        if len(war_subset) > 50:
            war_model = LinearRegression().fit(war_subset[[war_col]], war_subset["value"])
            war_residuals = war_subset["value"].values - war_model.predict(war_subset[[war_col]])

            print(f"  Partial correlation (after controlling for {war_col}):")
            results[player_type] = {}
            for col in anchor_cols:
                aligned = war_subset[col].values
                valid = ~np.isnan(aligned)
                r_partial, p_partial = stats.pearsonr(aligned[valid], war_residuals[valid])
                results[player_type][col] = (r_partial, p_partial)
                print(f"    {col}: r={r_partial:+.3f}  p={p_partial:.4e}  "
                      f"{'SIGNIFICANT' if abs(r_partial) >= 0.20 and p_partial < 0.05 else ''}")

        # Plot prior-year p90 over time per tier
        ax = axes[row_idx, 0]
        for tier in [1, 2, 3]:
            tier_data = (
                df_anchored[
                    (df_anchored["player_type"] == player_type) &
                    (df_anchored["arb_year"] == tier)
                ]
                .groupby("contract_year")["value"].mean()
            )
            ax.plot(tier_data.index, tier_data.values, marker="o", markersize=4, label=f"Year {tier} mean")
        ax.set_title(f"{player_type.capitalize()} — Avg Salary by Year")
        ax.set_xlabel("Contract Year")
        ax.set_ylabel("Avg Salary ($M)")
        ax.legend(fontsize=8)

        # Scatter: prior p90 vs current salary per tier
        for col_idx, tier in enumerate([1, 2, 3]):
            tier_subset = subset[subset["arb_year"] == tier].dropna(subset=["market_prior_p90"])
            ax = axes[row_idx, col_idx + 1]
            ax.scatter(tier_subset["market_prior_p90"], tier_subset["value"],
                       alpha=0.3, s=12, color="steelblue")
            if len(tier_subset) > 5:
                m, b = np.polyfit(tier_subset["market_prior_p90"], tier_subset["value"], 1)
                xl = np.linspace(tier_subset["market_prior_p90"].min(), tier_subset["market_prior_p90"].max(), 100)
                ax.plot(xl, m * xl + b, "r-", linewidth=1.5)
            ax.set_xlabel("Prior Year P90 ($M)")
            ax.set_ylabel("Current Salary ($M)")
            r_plot, _ = stats.pearsonr(tier_subset["market_prior_p90"], tier_subset["value"])
            ax.set_title(f"{player_type.capitalize()} Year {tier} p90 (r={r_plot:.2f})")

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/4_market_anchors.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {GRAPH_DIR}/4_market_anchors.png")

    return results


# ---------------------------------------------------------------------------
# Analysis 5: SP vs. RP Salary Distributions
# ---------------------------------------------------------------------------

def analysis_5_sp_vs_rp():
    section("Analysis 5: SP vs. RP Salary Distributions (tests structural split hypothesis)")

    df = load_arb_data_with_tiers()
    pitchers = df[df["player_type"] == "pitcher"].copy()
    pitchers["is_starter"] = pitchers["position"].isin(["SP"]).astype(int)
    pitchers["role"] = pitchers["position"].apply(
        lambda p: "SP" if p == "SP" else ("CL" if p == "CL" else "RP")
    )

    results = {}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("SP vs. RP/CL Salary Distributions by Arb Tier", fontsize=13)

    print()
    for col_idx, tier in enumerate([1, 2, 3]):
        tier_df = pitchers[pitchers["arb_year"] == tier]
        results[tier] = {}

        for row_idx, role in enumerate(["SP", "RP", "CL"]):
            role_df = tier_df[tier_df["role"] == role]["value"]
            if len(role_df) < 5:
                results[tier][role] = {"n": len(role_df), "mean": None, "std": None}
                axes[row_idx, col_idx].set_title(f"Year {tier} {role} (n={len(role_df)} — too few)")
                continue

            results[tier][role] = {
                "n": int(len(role_df)),
                "mean": float(role_df.mean()),
                "std": float(role_df.std()),
                "cv": float(role_df.std() / role_df.mean()),
            }
            print(f"  Year {tier} {role} (n={len(role_df)}): "
                  f"mean=${role_df.mean():.2f}M  std=${role_df.std():.2f}M  "
                  f"CV={role_df.std()/role_df.mean():.3f}")

            axes[row_idx, col_idx].hist(role_df, bins=25, color="steelblue",
                                         edgecolor="white", alpha=0.8)
            axes[row_idx, col_idx].set_title(
                f"Year {tier} {role} (n={len(role_df)}, mean=${role_df.mean():.1f}M)")
            axes[row_idx, col_idx].set_xlabel("Salary ($M)")

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/5_sp_vs_rp.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {GRAPH_DIR}/5_sp_vs_rp.png")

    # KS test: are SP and RP distributions different?
    print("\nKolmogorov-Smirnov test (SP vs RP salary distribution):")
    for tier in [1, 2, 3]:
        sp_vals = pitchers[(pitchers["arb_year"] == tier) & (pitchers["role"] == "SP")]["value"]
        rp_vals = pitchers[(pitchers["arb_year"] == tier) & (pitchers["role"] == "RP")]["value"]
        if len(sp_vals) > 5 and len(rp_vals) > 5:
            ks_stat, ks_p = stats.ks_2samp(sp_vals, rp_vals)
            print(f"  Year {tier}: KS={ks_stat:.3f}  p={ks_p:.4f}  "
                  f"{'SIGNIFICANTLY DIFFERENT' if ks_p < 0.05 else 'not significantly different'}")

    return results


# ---------------------------------------------------------------------------
# Analysis 6: Temporal Generalization
# ---------------------------------------------------------------------------

def analysis_6_temporal_generalization():
    section("Analysis 6: Temporal Generalization (train 2011-2022, test 2023-2025)")

    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Temporal Generalization: Train 2011-2022, Test 2023-2025", fontsize=13)

    for row_idx, player_type in enumerate(["pitcher", "batter"]):
        df = load_and_filter_data(player_type=player_type)
        all_features, y_all = get_features_and_target(df, player_type=player_type)
        df_full = df.reset_index(drop=True)
        years = df_full["contract_year"].values

        train_mask = years <= 2022
        test_mask = years >= 2023

        X_train_t = all_features[train_mask]
        y_train_t = y_all[train_mask]
        X_test_t = all_features[test_mask]
        y_test_t = y_all[test_mask]

        print(f"\n{player_type.upper()}:")
        print(f"  Train: {train_mask.sum()} samples (2011-2022)")
        print(f"  Test:  {test_mask.sum()} samples (2023-2025)")

        if test_mask.sum() < 10:
            print("  Insufficient test data — skipping")
            continue

        # Train a fresh model on the temporal split
        from models.arb.model import ArbModel as _ArbModel
        temporal_model = _ArbModel(model_type="random_forest", player_type=player_type)
        temporal_model.train(X_train_t, y_train_t)
        preds_temporal = temporal_model.predict(X_test_t)

        # Compare with random-split model (already trained, use saved)
        model_random = ArbModel.load(player_type=player_type)
        # Get the random-split test set for same player type
        X_train_r, X_test_r, y_train_r, y_test_r = get_train_test_split(df, player_type)
        preds_random = model_random.predict(X_test_r)

        mae_temporal = np.mean(np.abs(y_test_t.values - preds_temporal))
        mae_random = np.mean(np.abs(y_test_r.values - preds_random))

        mape_temporal = np.abs(y_test_t.values - preds_temporal) / y_test_t.values
        mape_random = np.abs(y_test_r.values - preds_random) / y_test_r.values

        p90_temporal = np.percentile(mape_temporal, 90) * 100
        p90_random = np.percentile(mape_random, 90) * 100
        within20_temporal = np.mean(mape_temporal <= 0.20) * 100
        within20_random = np.mean(mape_random <= 0.20) * 100

        results[player_type] = {
            "temporal": {"mae": float(mae_temporal), "mape_p90": float(p90_temporal), "within_20": float(within20_temporal)},
            "random_split": {"mae": float(mae_random), "mape_p90": float(p90_random), "within_20": float(within20_random)},
        }

        print(f"  MAE    — temporal: ${mae_temporal:.3f}M   random split: ${mae_random:.3f}M")
        print(f"  P90 MAPE — temporal: {p90_temporal:.1f}%   random split: {p90_random:.1f}%")
        print(f"  Within 20% — temporal: {within20_temporal:.1f}%   random split: {within20_random:.1f}%")

        # Scatter: actual vs predicted, temporal test set
        ax = axes[row_idx, 0]
        ax.scatter(y_test_t.values, preds_temporal, alpha=0.4, s=15, color="steelblue")
        lims = [min(y_test_t.min(), preds_temporal.min()) - 0.5,
                max(y_test_t.max(), preds_temporal.max()) + 0.5]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual Salary ($M)")
        ax.set_ylabel("Predicted Salary ($M)")
        ax.set_title(f"{player_type.capitalize()} — Temporal Test 2023-2025\nMAE=${mae_temporal:.2f}M  P90={p90_temporal:.0f}%")

        # MAPE by year in temporal test
        ax2 = axes[row_idx, 1]
        test_years = years[test_mask]
        for yr in sorted(set(test_years)):
            yr_mask = test_years == yr
            yr_mape = np.mean(mape_temporal[yr_mask]) * 100
            ax2.bar(yr, yr_mape, color="steelblue", alpha=0.7)
        ax2.axhline(20, color="red", linestyle="--", label="20% target")
        ax2.set_xlabel("Contract Year")
        ax2.set_ylabel("Mean MAPE (%)")
        ax2.set_title(f"{player_type.capitalize()} — Mean MAPE by Year (Temporal)")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/6_temporal_generalization.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {GRAPH_DIR}/6_temporal_generalization.png")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(r1, r2, r3, r4, r5, r6):
    section("SUMMARY OF FINDINGS")

    print("\n--- Analysis 1: MAPE Baseline ---")
    for pt in ["pitcher", "batter"]:
        p90 = r1[pt]["overall"][90]
        w20 = np.mean([r1[pt]["by_tier"][t]["pct_within_20"]
                       for t in r1[pt]["by_tier"]]) if r1[pt]["by_tier"] else 0
        print(f"  {pt}: 90th-pct MAPE = {p90:.1f}%  |  avg % within ±20% = {w20:.1f}%")
        print(f"         Gap to target: need to cut 90th-pct MAPE from {p90:.1f}% → 20%")

    print("\n--- Analysis 2: Residuals ---")
    for pt in ["pitcher", "batter"]:
        corrs = r2[pt]
        print(f"  {pt}:")
        print(f"    contract_year correlation:   r={corrs['contract_year'][0]:+.3f}")
        print(f"    WAR momentum correlation:    r={corrs['war_momentum'][0]:+.3f}")
        print(f"    ST×WAR interaction:          r={corrs['st_war_interaction'][0]:+.3f}")
        print(f"    heteroskedasticity:          r={corrs['heteroskedasticity'][0]:+.3f}")

    print("\n--- Analysis 3: Within-Tier Variance ---")
    for pt in ["pitcher", "batter"]:
        drops = [v["cv_reduction_pct"] for v in r3[pt]["tiers"].values()]
        print(f"  {pt}: avg CV drop = {np.mean(drops):.1f}%  → "
              f"{'PROCEED with tier models' if np.mean(drops) > 30 else 'Lower priority'}")

    print("\n--- Analysis 4: Market Anchors ---")
    for pt in ["pitcher", "batter"]:
        if pt not in r4:
            continue
        for col, (r_val, p_val) in r4[pt].items():
            flag = " ← SIGNIFICANT" if abs(r_val) >= 0.20 and p_val < 0.05 else ""
            print(f"  {pt} {col}: partial r={r_val:+.3f}{flag}")

    print("\n--- Analysis 5: SP vs. RP ---")
    for tier in [1, 2, 3]:
        sp = r5.get(tier, {}).get("SP", {})
        rp = r5.get(tier, {}).get("RP", {})
        if sp.get("mean") and rp.get("mean"):
            diff = sp["mean"] - rp["mean"]
            print(f"  Year {tier}: SP mean=${sp['mean']:.2f}M  RP mean=${rp['mean']:.2f}M  "
                  f"diff=${diff:+.2f}M")

    print("\n--- Analysis 6: Temporal Generalization ---")
    for pt in ["pitcher", "batter"]:
        if pt not in r6:
            continue
        t = r6[pt]["temporal"]
        rs = r6[pt]["random_split"]
        degradation = t["mape_p90"] - rs["mape_p90"]
        print(f"  {pt}: P90 MAPE temporal={t['mape_p90']:.1f}%  random={rs['mape_p90']:.1f}%  "
              f"degradation={degradation:+.1f}pp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs(GRAPH_DIR, exist_ok=True)

    print("Running Phase 0: Pre-Implementation Analysis")
    print("All graphs → analysis/graphs/arb_improvement/")

    r1 = analysis_1_mape_baseline()
    r2 = analysis_2_residuals()
    r3 = analysis_3_within_tier_variance()
    r4 = analysis_4_market_anchors()
    r5 = analysis_5_sp_vs_rp()
    r6 = analysis_6_temporal_generalization()

    print_summary(r1, r2, r3, r4, r5, r6)

    print("\nDone. Review graphs and summary above before proceeding to implementation.")
