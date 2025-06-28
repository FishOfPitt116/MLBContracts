import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from analysis.scripts import arb, free_agents, pre_arb

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

def normalize_service_time(st):
    years = int(st)
    days = round((st - years) * 1000)  # assuming the decimals are in 'days' out of 172
    scaled_days = days / 172
    return years + scaled_days

if __name__ == "__main__":
    # Create the graph directory if it doesn't exist
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Generate the general contract value distribution plot
    contract_value_distribution()
    print("General contract value distribution plot saved.")

    # Pre Arbitration Analysis
    pre_arb.main()

    # Arbitration Analysis
    arb.main()

    # Free Agents Analysis
    free_agents.main()
    