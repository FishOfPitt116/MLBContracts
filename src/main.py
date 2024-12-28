import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models import linear_regression, support_vector_regression, lasso_regression
from runs import test_model_all_combo, test_model_all_combo_with_alpha, find_best_model_combo

DATASET_DIR = "dataset"

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

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

if __name__ == "__main__":
    # generate batting and pitching data
    batting_data, pitching_data = get_merged_dfs()

    # split pitching dataset between starters and relievers
    starting_pitching_data, relief_pitching_data = split_starters_relievers(pitching_data)

    # normalize dataframes
    batting_data, batting_scaler = normalize_dataframe(batting_data)
    starting_pitching_data, starting_pitching_scaler = normalize_dataframe(starting_pitching_data)
    relief_pitching_data, relief_pitching_scaler = normalize_dataframe(relief_pitching_data)
    
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

    position_results_linear_regression = test_model_all_combo(
        "position",
        batting_data,
        linear_regression,
        ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
        ["mse", "r2"]
    )
    print(position_results_linear_regression)
    find_best_model_combo(position_results_linear_regression, "r2")

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

    position_results_svr = test_model_all_combo(
        "position",
        batting_data,
        support_vector_regression,
        ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
        ["mse", "r2"]
    )
    print(position_results_svr)
    find_best_model_combo(position_results_svr, "r2")

    starter_results_lasso = test_model_all_combo_with_alpha(
        "starter", 
        starting_pitching_data, 
        lasso_regression, 
        ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(starter_results_lasso)
    find_best_model_combo(starter_results_lasso, "r2")

    reliever_results_lasso = test_model_all_combo_with_alpha(
        "reliever",
        relief_pitching_data,
        lasso_regression,
        ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
        ["mse", "r2"]
    )
    print(reliever_results_lasso)
    find_best_model_combo(reliever_results_lasso, "r2")

    position_results_lasso = test_model_all_combo_with_alpha(
        "position",
        batting_data,
        lasso_regression,
        ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
        ["mse", "r2"]
    )
    print(position_results_lasso)
    find_best_model_combo(position_results_lasso, "r2")