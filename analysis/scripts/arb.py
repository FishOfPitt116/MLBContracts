"""Arbitration contract analysis and feature exploration."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.contract_analysis import GRAPH_DIR
from models.preprocessing import load_contracts, normalize_service_time

# Load contracts with stats for analysis
CONTRACTS_WITH_STATS = None


def get_arb_data():
    """Load and filter arbitration contracts with stats."""
    global CONTRACTS_WITH_STATS
    if CONTRACTS_WITH_STATS is None:
        CONTRACTS_WITH_STATS = load_contracts()

    df = CONTRACTS_WITH_STATS.copy()

    # Filter to single-year arbitration contracts
    mask = (df["contract_type"] == "arb") & (df["duration"] == 1)
    df = df[mask].copy()

    # Filter out records without service time
    df = df[df["service_time"].notna()].copy()

    # Normalize service time
    df["service_time"] = df["service_time"].apply(normalize_service_time)

    # Determine arb year from service time (2-5)
    # Arb Year 2: ST < 4 (includes Super Two with ST 2-3)
    # Arb Year 3: 4 <= ST < 5
    # Arb Year 4: 5 <= ST < 6
    # Arb Year 5: ST >= 6 (very rare)
    df["arb_year"] = df["service_time"].apply(_service_time_to_arb_year)

    return df


def _service_time_to_arb_year(st):
    """Convert normalized service time to arbitration year (1-3)."""
    if pd.isna(st):
        return None
    if st < 4:
        return 1  # Arb Year 1 (includes Super Two ST 2-3 and regular first-timers ST 3-4)
    elif st < 5:
        return 2  # Arb Year 2
    else:
        return 3  # Arb Year 3 (final arb year before free agency)


def arbitration_service_time_vs_contract_value(best_fit=False):
    """Visualize relationship between arbitration service time and contract value."""
    df = get_arb_data()

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=df, x="service_time", y="value", hue="arb_year", palette="Set2", alpha=0.7
    )

    if best_fit:
        sns.regplot(
            data=df,
            x="service_time",
            y="value",
            scatter=False,
            color="blue",
            line_kws={"linestyle": "dotted"},
        )

    plt.title("Arbitration Service Time vs Contract Value", fontsize=16)
    plt.xlabel("Service Time (years)", fontsize=14)
    plt.ylabel("Contract Value (millions)", fontsize=14)
    plt.legend(title="Arb Year")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arbitration_service_time_vs_contract_value.png"))
    plt.close()


def feature_correlations_by_arb_year():
    """Analyze feature correlations with salary for each arb year tier, separated by player type."""
    df = get_arb_data()

    # Separate pitchers and position players
    pitcher_positions = ["SP", "RP", "CL", "P"]
    pitchers_df = df[df["position"].isin(pitcher_positions)].copy()
    position_players_df = df[~df["position"].isin(pitcher_positions)].copy()

    arb_years = [1, 2, 3]

    # --- Batting Stats (Position Players Only) ---
    bat_cols = [c for c in df.columns if c.startswith("bat_")]
    bat_feature_cols = ["age", "service_time", "contract_year"] + bat_cols

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, arb_year in enumerate(arb_years):
        ax = axes[idx]
        year_df = position_players_df[position_players_df["arb_year"] == arb_year]

        if len(year_df) < 10:
            ax.text(0.5, 0.5, f"Insufficient data for Year {arb_year}", ha="center")
            ax.set_title(f"Arb Year {arb_year} (n={len(year_df)})")
            continue

        # Calculate correlations with salary
        correlations = {}
        for col in bat_feature_cols:
            if col in year_df.columns:
                valid = year_df[[col, "value"]].dropna()
                if len(valid) > 5:
                    corr = valid[col].corr(valid["value"])
                    if not pd.isna(corr):
                        correlations[col] = corr

        # Get top 12 by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:12]

        if not sorted_corrs:
            ax.text(0.5, 0.5, f"No correlations for Year {arb_year}", ha="center")
            continue

        features, corrs = zip(*sorted_corrs)

        colors = ["green" if c > 0 else "red" for c in corrs]
        ax.barh(range(len(corrs)), corrs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Correlation with Salary")
        ax.set_title(f"Year {arb_year} (n={len(year_df)}, avg=${year_df['value'].mean():.2f}M)")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(-0.2, 0.8)
        ax.invert_yaxis()

    plt.suptitle("Batting Feature Correlations by Arb Year (Position Players Only)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_feature_correlations_batting.png"), bbox_inches="tight")
    plt.close()

    # --- Pitching Stats (Pitchers Only) ---
    pit_cols = [c for c in df.columns if c.startswith("pit_")]
    pit_feature_cols = ["age", "service_time", "contract_year"] + pit_cols

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, arb_year in enumerate(arb_years):
        ax = axes[idx]
        year_df = pitchers_df[pitchers_df["arb_year"] == arb_year]

        if len(year_df) < 10:
            ax.text(0.5, 0.5, f"Insufficient data for Year {arb_year}", ha="center")
            ax.set_title(f"Arb Year {arb_year} (n={len(year_df)})")
            continue

        # Calculate correlations with salary
        correlations = {}
        for col in pit_feature_cols:
            if col in year_df.columns:
                valid = year_df[[col, "value"]].dropna()
                if len(valid) > 5:
                    corr = valid[col].corr(valid["value"])
                    if not pd.isna(corr):
                        correlations[col] = corr

        # Get top 12 by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:12]

        if not sorted_corrs:
            ax.text(0.5, 0.5, f"No correlations for Year {arb_year}", ha="center")
            continue

        features, corrs = zip(*sorted_corrs)

        colors = ["green" if c > 0 else "red" for c in corrs]
        ax.barh(range(len(corrs)), corrs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Correlation with Salary")
        ax.set_title(f"Year {arb_year} (n={len(year_df)}, avg=${year_df['value'].mean():.2f}M)")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(-0.3, 0.9)
        ax.invert_yaxis()

    plt.suptitle("Pitching Feature Correlations by Arb Year (Pitchers Only)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_feature_correlations_pitching.png"), bbox_inches="tight")
    plt.close()

    # Print summary to console
    print("\n" + "=" * 60)
    print("Feature Correlations Summary by Arb Year")
    print("=" * 60)

    print("\n--- Position Players (Batting) ---")
    for arb_year in arb_years:
        year_df = position_players_df[position_players_df["arb_year"] == arb_year]
        print(f"\nArb Year {arb_year}: n={len(year_df)}, avg salary=${year_df['value'].mean():.2f}M")

        correlations = {}
        for col in bat_feature_cols:
            if col in year_df.columns:
                valid = year_df[[col, "value"]].dropna()
                if len(valid) > 5:
                    corr = valid[col].corr(valid["value"])
                    if not pd.isna(corr):
                        correlations[col] = corr

        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        for feat, corr in sorted_corrs:
            print(f"  {feat:<35} {corr:+.3f}")

    print("\n--- Pitchers (Pitching) ---")
    for arb_year in arb_years:
        year_df = pitchers_df[pitchers_df["arb_year"] == arb_year]
        print(f"\nArb Year {arb_year}: n={len(year_df)}, avg salary=${year_df['value'].mean():.2f}M")

        correlations = {}
        for col in pit_feature_cols:
            if col in year_df.columns:
                valid = year_df[[col, "value"]].dropna()
                if len(valid) > 5:
                    corr = valid[col].corr(valid["value"])
                    if not pd.isna(corr):
                        correlations[col] = corr

        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        for feat, corr in sorted_corrs:
            print(f"  {feat:<35} {corr:+.3f}")


def age_salary_interaction_analysis():
    """Analyze how age interacts with salary across arb years."""
    df = get_arb_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Age distribution by arb year
    ax1 = axes[0, 0]
    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 0:
            ax1.hist(year_df["age"], alpha=0.5, label=f"Year {arb_year}", bins=15)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Count")
    ax1.set_title("Age Distribution by Arb Year")
    ax1.legend()

    # Plot 2: Salary vs Age scatter by arb year
    ax2 = axes[0, 1]
    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 0:
            ax2.scatter(year_df["age"], year_df["value"], alpha=0.5, label=f"Year {arb_year}", s=20)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Salary (millions)")
    ax2.set_title("Salary vs Age by Arb Year")
    ax2.legend()

    # Plot 3: Box plot of salary by age bucket
    ax3 = axes[1, 0]
    df["age_bucket"] = pd.cut(df["age"], bins=[20, 25, 27, 30, 35, 45], labels=["21-25", "26-27", "28-30", "31-35", "36+"])
    valid_df = df.dropna(subset=["age_bucket", "value"])
    if len(valid_df) > 0:
        sns.boxplot(data=valid_df, x="age_bucket", y="value", ax=ax3, showfliers=False)
    ax3.set_xlabel("Age Bucket")
    ax3.set_ylabel("Salary (millions)")
    ax3.set_title("Salary Distribution by Age Bucket")

    # Plot 4: Correlation of WAR with salary for young vs old
    ax4 = axes[1, 1]
    df["age_group"] = df["age"].apply(lambda x: "Young (<27)" if x < 27 else "Older (27+)")

    # Calculate average WAR (batting or pitching)
    df["total_war_1y"] = df["bat_war_1y"].fillna(0) + df["pit_war_1y"].fillna(0)

    for group in ["Young (<27)", "Older (27+)"]:
        group_df = df[df["age_group"] == group]
        if len(group_df) > 10:
            ax4.scatter(
                group_df["total_war_1y"],
                group_df["value"],
                alpha=0.4,
                label=f"{group} (n={len(group_df)})",
                s=20,
            )
    ax4.set_xlabel("1-Year WAR (Bat + Pit)")
    ax4.set_ylabel("Salary (millions)")
    ax4.set_title("WAR vs Salary by Age Group")
    ax4.legend()

    plt.suptitle("Age-Salary Interaction Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_age_salary_interaction.png"), bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Age-Salary Interaction Summary")
    print("=" * 60)

    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 0:
            print(f"\nArb Year {arb_year}:")
            print(f"  Age range: {year_df['age'].min():.0f} - {year_df['age'].max():.0f}")
            print(f"  Mean age: {year_df['age'].mean():.1f}")
            corr = year_df[["age", "value"]].dropna().corr().iloc[0, 1]
            print(f"  Age-Salary correlation: {corr:+.3f}")


def salary_by_position_analysis():
    """Analyze salary distribution and feature importance by position."""
    df = get_arb_data()

    # Define position groups
    position_groups = {
        "SP": ["SP"],
        "RP": ["RP", "CL"],
        "C": ["C"],
        "IF": ["1B", "2B", "3B", "SS"],
        "OF": ["LF", "CF", "RF", "OF", "DH"],
    }

    def get_position_group(pos):
        if pd.isna(pos):
            return "Unknown"
        for group, positions in position_groups.items():
            if pos in positions:
                return group
        return "Unknown"

    df["position_group"] = df["position"].apply(get_position_group)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Box plot of salary by position
    ax1 = axes[0, 0]
    order = ["SP", "RP", "C", "IF", "OF"]
    valid_positions = [p for p in order if p in df["position_group"].unique()]
    sns.boxplot(data=df, x="position_group", y="value", order=valid_positions, ax=ax1, showfliers=False)
    ax1.set_xlabel("Position Group")
    ax1.set_ylabel("Salary (millions)")
    ax1.set_title("Salary Distribution by Position")

    # Plot 2: Count by position group and arb year
    ax2 = axes[0, 1]
    pivot = df.groupby(["position_group", "arb_year"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(valid_positions)
    pivot.plot(kind="bar", ax=ax2, width=0.8)
    ax2.set_xlabel("Position Group")
    ax2.set_ylabel("Count")
    ax2.set_title("Contract Count by Position and Arb Year")
    ax2.legend(title="Arb Year")
    ax2.tick_params(axis="x", rotation=0)

    # Plot 3: Mean salary by position and arb year
    ax3 = axes[1, 0]
    pivot_salary = df.groupby(["position_group", "arb_year"])["value"].mean().unstack(fill_value=0)
    pivot_salary = pivot_salary.reindex(valid_positions)
    pivot_salary.plot(kind="bar", ax=ax3, width=0.8)
    ax3.set_xlabel("Position Group")
    ax3.set_ylabel("Mean Salary (millions)")
    ax3.set_title("Mean Salary by Position and Arb Year")
    ax3.legend(title="Arb Year")
    ax3.tick_params(axis="x", rotation=0)

    # Plot 4: WAR correlation with salary by position
    ax4 = axes[1, 1]
    correlations = []
    for pos_group in valid_positions:
        pos_df = df[df["position_group"] == pos_group]
        if len(pos_df) > 20:
            # Use bat_war for position players, pit_war for pitchers
            if pos_group in ["SP", "RP"]:
                war_col = "pit_war_1y"
            else:
                war_col = "bat_war_1y"

            valid = pos_df[[war_col, "value"]].dropna()
            if len(valid) > 10:
                corr = valid[war_col].corr(valid["value"])
                correlations.append((pos_group, corr, len(valid)))

    if correlations:
        groups, corrs, counts = zip(*correlations)
        colors = ["blue" if "P" in g else "green" for g in groups]
        bars = ax4.bar(groups, corrs, color=colors, alpha=0.7)
        ax4.set_xlabel("Position Group")
        ax4.set_ylabel("WAR-Salary Correlation")
        ax4.set_title("WAR Correlation with Salary by Position")
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Add count labels
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={count}", ha="center", fontsize=8)

    plt.suptitle("Salary Analysis by Position", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_salary_by_position.png"), bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Position Analysis Summary")
    print("=" * 60)

    for pos_group in valid_positions:
        pos_df = df[df["position_group"] == pos_group]
        print(f"\n{pos_group} (n={len(pos_df)}):")
        print(f"  Mean salary: ${pos_df['value'].mean():.2f}M")
        print(f"  Median salary: ${pos_df['value'].median():.2f}M")
        print(f"  Std salary: ${pos_df['value'].std():.2f}M")


def service_time_deep_dive():
    """Deep dive into service time effects within each arb year."""
    df = get_arb_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Service time vs salary with regression lines per arb year
    ax1 = axes[0, 0]
    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 10:
            ax1.scatter(year_df["service_time"], year_df["value"], alpha=0.3, label=f"Year {arb_year}", s=20)
            # Add regression line
            valid = year_df[["service_time", "value"]].dropna()
            if len(valid) > 5:
                z = np.polyfit(valid["service_time"], valid["value"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid["service_time"].min(), valid["service_time"].max(), 100)
                ax1.plot(x_line, p(x_line), linestyle="--", linewidth=2)

    ax1.set_xlabel("Service Time (years)")
    ax1.set_ylabel("Salary (millions)")
    ax1.set_title("Service Time vs Salary by Arb Year")
    ax1.legend()

    # Plot 2: Within-year service time variation
    ax2 = axes[0, 1]
    # Create fractional service time (days within year)
    df["service_days"] = (df["service_time"] % 1) * 172  # Convert back to days
    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 10:
            ax2.scatter(year_df["service_days"], year_df["value"], alpha=0.3, label=f"Year {arb_year}", s=20)

    ax2.set_xlabel("Days of Service (within year)")
    ax2.set_ylabel("Salary (millions)")
    ax2.set_title("Service Days vs Salary (Within-Year Effect)")
    ax2.legend()

    # Plot 3: Service time × WAR interaction
    ax3 = axes[1, 0]
    df["total_war_1y"] = df["bat_war_1y"].fillna(0) + df["pit_war_1y"].fillna(0)
    df["service_war_interaction"] = df["service_time"] * df["total_war_1y"]

    valid = df[["service_war_interaction", "value", "arb_year"]].dropna()
    for arb_year in [1, 2, 3]:
        year_df = valid[valid["arb_year"] == arb_year]
        if len(year_df) > 10:
            ax3.scatter(year_df["service_war_interaction"], year_df["value"], alpha=0.3, label=f"Year {arb_year}", s=20)

    ax3.set_xlabel("Service Time × WAR")
    ax3.set_ylabel("Salary (millions)")
    ax3.set_title("Service Time × WAR Interaction")
    ax3.legend()

    # Plot 4: Box plot of salary variance within each service year bucket
    # Note: Max ST in data is ~5.99, so only include buckets with data
    ax4 = axes[1, 1]
    df["service_bucket"] = pd.cut(
        df["service_time"],
        bins=[2, 3, 4, 5, 6],
        labels=["2-3", "3-4", "4-5", "5-6"],
    )
    valid_df = df.dropna(subset=["service_bucket", "value"])
    if len(valid_df) > 0:
        sns.boxplot(data=valid_df, x="service_bucket", y="value", ax=ax4, showfliers=False)
    ax4.set_xlabel("Service Time Bucket (years)")
    ax4.set_ylabel("Salary (millions)")
    ax4.set_title("Salary Distribution by Service Time Bucket")

    plt.suptitle("Service Time Deep Dive", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_service_time_analysis.png"), bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Service Time Analysis Summary")
    print("=" * 60)

    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        if len(year_df) > 5:
            print(f"\nArb Year {arb_year} (n={len(year_df)}):")
            print(f"  Service time range: {year_df['service_time'].min():.2f} - {year_df['service_time'].max():.2f}")

            # Correlation of service time with salary within this year
            valid = year_df[["service_time", "value"]].dropna()
            if len(valid) > 5:
                corr = valid["service_time"].corr(valid["value"])
                print(f"  Service time-salary corr: {corr:+.3f}")

            # Correlation of interaction term
            year_df_copy = year_df.copy()
            year_df_copy["total_war_1y"] = year_df_copy["bat_war_1y"].fillna(0) + year_df_copy["pit_war_1y"].fillna(0)
            year_df_copy["interaction"] = year_df_copy["service_time"] * year_df_copy["total_war_1y"]
            valid = year_df_copy[["interaction", "value"]].dropna()
            if len(valid) > 5:
                corr = valid["interaction"].corr(valid["value"])
                print(f"  Service×WAR interaction corr: {corr:+.3f}")


def salary_inflation_analysis():
    """Analyze year-over-year salary inflation for arbitration contracts."""
    df = get_arb_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean salary by contract year (overall)
    ax1 = axes[0, 0]
    yearly_overall = df.groupby("contract_year")["value"].mean()
    ax1.plot(yearly_overall.index, yearly_overall.values, marker="o", linewidth=2, color="blue")
    ax1.fill_between(yearly_overall.index, yearly_overall.values, alpha=0.3)
    ax1.set_xlabel("Contract Year")
    ax1.set_ylabel("Mean Salary (millions)")
    ax1.set_title("Overall Mean Arb Salary by Year")
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(yearly_overall.index, yearly_overall.values, 1)
    p = np.poly1d(z)
    ax1.plot(yearly_overall.index, p(yearly_overall.index), "--", color="red", alpha=0.7, label=f"Trend: {z[0]*100:.1f}%/yr")
    ax1.legend()

    # Plot 2: Mean salary by contract year per arb tier
    ax2 = axes[0, 1]
    colors = {1: "green", 2: "orange", 3: "red"}
    for tier in [1, 2, 3]:
        tier_df = df[df["arb_year"] == tier]
        yearly = tier_df.groupby("contract_year")["value"].mean()
        ax2.plot(yearly.index, yearly.values, marker="o", linewidth=2, color=colors[tier], label=f"Year {tier}", alpha=0.8)

    ax2.set_xlabel("Contract Year")
    ax2.set_ylabel("Mean Salary (millions)")
    ax2.set_title("Mean Arb Salary by Year (by Tier)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Year-over-year percentage change
    ax3 = axes[1, 0]
    yearly_pct_change = yearly_overall.pct_change() * 100
    colors_bar = ["green" if x >= 0 else "red" for x in yearly_pct_change.values[1:]]
    ax3.bar(yearly_pct_change.index[1:], yearly_pct_change.values[1:], color=colors_bar, alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Contract Year")
    ax3.set_ylabel("Year-over-Year Change (%)")
    ax3.set_title("YoY Salary Change (%)")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Contract count by year per tier
    ax4 = axes[1, 1]
    for tier in [1, 2, 3]:
        tier_df = df[df["arb_year"] == tier]
        yearly_count = tier_df.groupby("contract_year").size()
        ax4.plot(yearly_count.index, yearly_count.values, marker="s", linewidth=2, color=colors[tier], label=f"Year {tier}", alpha=0.8)

    ax4.set_xlabel("Contract Year")
    ax4.set_ylabel("Number of Contracts")
    ax4.set_title("Contract Count by Year (by Tier)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Arbitration Salary Inflation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "arb_salary_inflation.png"), bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Salary Inflation Analysis")
    print("=" * 60)

    # Calculate CAGR for each tier
    for tier in [1, 2, 3]:
        tier_df = df[df["arb_year"] == tier]
        yearly = tier_df.groupby("contract_year")["value"].agg(["count", "mean"])

        # Use years with at least 10 contracts
        valid = yearly[yearly["count"] >= 10]
        if len(valid) >= 2:
            first_year = valid.index.min()
            last_year = valid.index.max()
            first_val = valid.loc[first_year, "mean"]
            last_val = valid.loc[last_year, "mean"]
            years_diff = last_year - first_year
            cagr = ((last_val / first_val) ** (1 / years_diff) - 1) * 100
            print(f"\nArb Year {tier}:")
            print(f"  Period: {int(first_year)} - {int(last_year)}")
            print(f"  Start avg: ${first_val:.2f}M -> End avg: ${last_val:.2f}M")
            print(f"  CAGR: {cagr:+.1f}%/year")

    # Overall correlation
    corr = df[["contract_year", "value"]].corr().iloc[0, 1]
    print(f"\nContract year correlation with salary: {corr:+.3f}")
    print("(Weak correlation suggests inflation is not a major factor)")


def generate_analysis_summary():
    """Generate a markdown summary of analysis findings."""
    df = get_arb_data()

    summary = ["# Arbitration Model Analysis Summary\n"]
    summary.append("## Dataset Overview\n")
    summary.append(f"- Total arbitration contracts: {len(df)}\n")

    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        summary.append(f"- Arb Year {arb_year}: n={len(year_df)}, avg=${year_df['value'].mean():.2f}M, "
                      f"std=${year_df['value'].std():.2f}M\n")

    summary.append("\n## Key Feature Correlations\n")
    summary.append("\n### Consistently Strong Features (across all arb years):\n")

    # Find features with consistent correlations
    stat_cols = [c for c in df.columns if c.startswith("bat_") or c.startswith("pit_")]
    feature_cols = ["age", "service_time", "contract_year"] + stat_cols

    all_correlations = {}
    for arb_year in [1, 2, 3]:
        year_df = df[df["arb_year"] == arb_year]
        for col in feature_cols:
            if col in year_df.columns:
                valid = year_df[[col, "value"]].dropna()
                if len(valid) > 5:
                    corr = valid[col].corr(valid["value"])
                    if not pd.isna(corr):
                        if col not in all_correlations:
                            all_correlations[col] = []
                        all_correlations[col].append((arb_year, corr))

    # Features with consistent direction and magnitude
    consistent_features = []
    for feat, corrs in all_correlations.items():
        if len(corrs) >= 3:  # Present in at least 3 arb years
            values = [c[1] for c in corrs]
            avg_corr = np.mean(values)
            if abs(avg_corr) > 0.2:  # Reasonably strong
                consistent_features.append((feat, avg_corr, len(corrs)))

    consistent_features.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, avg_corr, count in consistent_features[:15]:
        summary.append(f"- `{feat}`: avg correlation = {avg_corr:+.3f} (in {count} arb years)\n")

    summary.append("\n## Recommended Feature Selection\n")
    summary.append("\n### Personal Features:\n")
    summary.append("- `age` - consistently correlated with salary\n")
    summary.append("- `service_time` - primary driver of arb salary tier\n")
    summary.append("- `contract_year` - captures inflation/market trends\n")
    summary.append("- `position` - different markets for pitchers vs position players\n")

    summary.append("\n### Batting Stats (for position players):\n")
    bat_features = [f for f, _, _ in consistent_features if f.startswith("bat_")][:5]
    for f in bat_features:
        summary.append(f"- `{f}`\n")

    summary.append("\n### Pitching Stats (for pitchers):\n")
    pit_features = [f for f, _, _ in consistent_features if f.startswith("pit_")][:5]
    for f in pit_features:
        summary.append(f"- `{f}`\n")

    summary.append("\n## Model Architecture Recommendations\n")
    summary.append("\n1. **Single unified model** - position can be a feature rather than separate models\n")
    summary.append("2. **RandomForest** preferred - handles mixed feature types well\n")
    summary.append("3. **Include interaction terms** - service_time × WAR shows promise\n")
    summary.append("4. **Tiered evaluation** - different tolerances per arb year\n")

    # Write to file
    summary_path = os.path.join("docs", "arb", "ANALYSIS_SUMMARY.md")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.writelines(summary)

    print(f"\nAnalysis summary written to {summary_path}")


def main():
    """Run all arbitration analysis functions."""
    print("=" * 60)
    print("Arbitration Contract Analysis")
    print("=" * 60)

    # Basic scatter plot
    arbitration_service_time_vs_contract_value()
    print("Arbitration service time vs contract value plot saved.")

    # Phase 1 comprehensive analysis
    feature_correlations_by_arb_year()
    print("\nFeature correlations plots saved (batting and pitching separately).")

    age_salary_interaction_analysis()
    print("\nAge-salary interaction analysis plot saved.")

    salary_by_position_analysis()
    print("\nSalary by position analysis plot saved.")

    service_time_deep_dive()
    print("\nService time deep dive plot saved.")

    salary_inflation_analysis()
    print("\nSalary inflation analysis plot saved.")

    # Note: Analysis summary (docs/arb/ANALYSIS_SUMMARY.md) is manually maintained
    # with additional sections like Super Two documentation, Rate Stats vs Counting Stats, etc.
    # Do not auto-generate to avoid overwriting curated content.

    print("\n" + "=" * 60)
    print("Arbitration analysis complete!")
    print("=" * 60)
