import os
import pandas as pd

from models import linear_regression, support_vector_regression
from runs import test_model_all_combo, find_best_model_combo

DATASET_DIR = "../dataset"

BATTING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "batting_stats.csv"))
CONTRACTS = pd.read_csv(os.path.join(DATASET_DIR, "contracts.csv"))
PITCHING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "pitching_stats.csv"))

def filter_by_position(pos):
    return CONTRACTS.query(f'position == \"{pos}\"')

def get_merged_dfs():
    batting_contracts = pd.DataFrame(columns=CONTRACTS.columns)
    pitching_contracts = pd.DataFrame(columns=CONTRACTS.columns)

    for position in CONTRACTS["position"].unique():
        pos_df = filter_by_position(position)
        if position in ["rhp-s", "lhp-s", "rhp-c", "lhp-c", "rhp", "lhp"]:
            pitching_contracts = pitching_contracts._append(pos_df, ignore_index=False)
        elif "lhp" in position or "rhp" in position:
            pass # shohei case
        else:
            batting_contracts = batting_contracts._append(pos_df, ignore_index=False)

    batting_data = pd.merge(batting_contracts, BATTING_STATS, left_on="Unnamed: 0", right_on="Index")
    pitching_data = pd.merge(pitching_contracts, PITCHING_STATS, left_on="Unnamed: 0", right_on="Index")

    return batting_data, pitching_data


# threshold = the percent of games a starter has to start to be considered a starter
def split_starters_relievers(data, threshold=0.85):
    sp = data[data["G"]*threshold <= data["GS"]]
    rp = data[data["G"]*threshold > data["GS"]]
    return sp, rp

if __name__ == "__main__":
    # generate batting and pitching data
    batting_data, pitching_data = get_merged_dfs()

    # split pitching dataset between starters and relievers
    starting_pitching_data, relief_pitching_data = split_starters_relievers(pitching_data)
    
    # train and test logistic regression on both types of pitchers and print results
    starter_results_linear_regression = test_model_all_combo(
        "starter", 
        starting_pitching_data, 
        linear_regression, 
        ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(starter_results_linear_regression)
    find_best_model_combo(starter_results_linear_regression, "r2")

    reliever_results_linear_regression = test_model_all_combo(
        "reliever",
        relief_pitching_data,
        linear_regression,
        ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(reliever_results_linear_regression)
    find_best_model_combo(reliever_results_linear_regression, "r2")

    starter_results_svr = test_model_all_combo(
        "starter", 
        starting_pitching_data, 
        support_vector_regression, 
        ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(starter_results_svr)
    find_best_model_combo(starter_results_svr, "r2")

    reliever_results_svr = test_model_all_combo(
        "reliever",
        relief_pitching_data,
        support_vector_regression,
        ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(reliever_results_svr)
    find_best_model_combo(reliever_results_svr, "r2")