import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

GRAPH_DIR = os.path.join("analysis", "graphs")
CONTRACT_DATASET = "dataset/contracts_spotrac.csv"

CONTRACT_DATA = pd.read_csv(CONTRACT_DATASET)

CONTRACT_DATA["aav"] = CONTRACT_DATA["value"] / CONTRACT_DATA["duration"]

def contract_value_distribution():
    """
    This function visualizes the distribution of contract values by age and contract type.
    It uses a boxplot to show the distribution of average annual value (AAV) for different
    age groups and contract types.
    """
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=CONTRACT_DATA,
        x="age",
        y="aav",
        hue="type",
        palette="Set2",
        showfliers=False
    )

    # Customize the plot
    plt.title('Contract Value Distribution by Age and Contract Type', fontsize=16)
    plt.xlabel('Age at Signing', fontsize=14)
    plt.ylabel('Contract Value (in millions)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Contract Type', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(GRAPH_DIR, "contract_value_distribution.png"))

def arbitration_service_time_vs_contract_value(best_fit=False):
    """
    This function visualizes the relationship between arbitration service time and contract value.
    It uses a scatter plot to show the correlation between these two variables.
    """
    # Filter out records where service_time is -1 and type is not 'arb'
    filtered_data = CONTRACT_DATA[(CONTRACT_DATA["service_time"] != -1) & (CONTRACT_DATA["type"] == "arb")]

    # Update service time scale by dividing value after decimal point by 172 (1 year = 172 days)
    filtered_data["service_time"] = filtered_data["service_time"].apply(lambda x: _normalize_service_time(x))

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

def _normalize_service_time(st):
    years = int(st)
    days = round((st - years) * 1000)  # assuming the decimals are in 'days' out of 172
    scaled_days = days / 172
    return years + scaled_days

if __name__ == "__main__":
    # Create the graph directory if it doesn't exist
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Generate the contract value distribution plot
    contract_value_distribution()
    print("Contract value distribution plot saved.")

    # Generate the arbitration service time vs contract value plot
    arbitration_service_time_vs_contract_value()
    print("Arbitration service time vs contract value plot saved.")