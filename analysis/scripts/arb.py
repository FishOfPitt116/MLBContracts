from matplotlib import pyplot as plt
import os
import seaborn as sns

from analysis.contract_analysis import CONTRACT_DATA, GRAPH_DIR, normalize_service_time


def arbitration_service_time_vs_contract_value(best_fit=False):
    """
    This function visualizes the relationship between arbitration service time and contract value.
    It uses a scatter plot to show the correlation between these two variables.
    """
    # Filter out records where service_time is -1 and type is not 'arb'
    filtered_data = CONTRACT_DATA[(CONTRACT_DATA["service_time"] != -1) & (CONTRACT_DATA["type"] == "arb")]

    # Update service time scale by dividing value after decimal point by 172 (1 year = 172 days)
    filtered_data["service_time"] = filtered_data["service_time"].apply(lambda x: normalize_service_time(x))

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=filtered_data,
        x="service_time",
        y="aav",
        palette="Set2",
        alpha=0.7
    )

    # Add a dotted best-fit line
    if best_fit:
        sns.regplot(
            data=filtered_data,
            x="service_time",
            y="aav",
            scatter=False,
            color="blue",
            line_kws={"linestyle": "dotted"}
        )

    # Customize the plot
    plt.title('Arbitration Service Time vs Contract Value', fontsize=16)
    plt.xlabel('Arbitration Service Time (yrs)', fontsize=14)
    plt.ylabel('Contract Value (in millions)', fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(GRAPH_DIR, "arbitration_service_time_vs_contract_value.png"))

def main():
    # Generate the arbitration service time vs contract value plot
    arbitration_service_time_vs_contract_value()
    print("Arbitration service time vs contract value plot saved.")