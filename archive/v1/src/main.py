import os
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models import linear_regression, support_vector_regression, lasso_regression, write_scaler_to_file
from runs import test_model_all_combo, test_model_all_combo_with_alpha, find_best_model_combo

DATASET_DIR = "dataset"

BATTING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "batting_stats.csv"))
CONTRACTS = pd.read_csv(os.path.join(DATASET_DIR, "contracts.csv"))
PITCHING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "pitching_stats.csv"))

def filter_by_position(pos: str) -> pd.DataFrame:
    return CONTRACTS.query(f'position == \"{pos}\"')

def get_merged_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    batting_contracts = pd.DataFrame(columns=CONTRACTS.columns)
    pitching_contracts = pd.DataFrame(columns=CONTRACTS.columns)

    for position in CONTRACTS["position"].unique():
        pos_df = filter_by_position(position)
        if position in ["rhp-s", "lhp-s", "rhp-c", "lhp-c", "rhp", "lhp"]:
            if pitching_contracts.empty and not pos_df.empty:
                pitching_contracts = pos_df
            elif not pos_df.empty:
                pitching_contracts = pd.concat([pitching_contracts, pos_df], ignore_index=True)
        elif "lhp" in position or "rhp" in position:
            pass # shohei case
        else:
            if batting_contracts.empty and not pos_df.empty:
                batting_contracts = pos_df
            elif not pos_df.empty:
                batting_contracts = pd.concat([batting_contracts, pos_df], ignore_index=True)

    batting_data = pd.merge(batting_contracts, BATTING_STATS, left_on="Unnamed: 0", right_on="Index")
    pitching_data = pd.merge(pitching_contracts, PITCHING_STATS, left_on="Unnamed: 0", right_on="Index")

    return batting_data, pitching_data

# threshold = the percent of games a starter has to start to be considered a starter
def split_starters_relievers(data: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sp = data[data["G"]*threshold <= data["GS"]]
    rp = data[data["G"]*threshold > data["GS"]]
    return sp, rp

def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    df = sanitize_dataframe(df)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(df)
    return df, scaler

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

if __name__ == "__main__":
    # generate batting and pitching data
    batting_data, pitching_data = get_merged_dfs()

    # split pitching dataset between starters and relievers
    starting_pitching_data, relief_pitching_data = split_starters_relievers(pitching_data)

    # normalize dataframes
    if not os.path.exists("scalers"):
        os.mkdir("scalers")
    batting_data, batting_scaler = normalize_dataframe(batting_data)
    write_scaler_to_file(batting_scaler, os.path.join("scalers", "batting_scaler.pkl"))
    starting_pitching_data, starting_pitching_scaler = normalize_dataframe(starting_pitching_data)
    write_scaler_to_file(starting_pitching_scaler, os.path.join("scalers", "starting_pitching_scaler.pkl"))
    relief_pitching_data, relief_pitching_scaler = normalize_dataframe(relief_pitching_data)
    write_scaler_to_file(relief_pitching_scaler, os.path.join("scalers", "relief_pitching_scaler.pkl"))


    
    # train and test logistic regression on both types of pitchers and print results
    starter_results_linear_regression = test_model_all_combo(
        "starter", 
        starting_pitching_data,
        linear_regression, 
        ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
        ["mse", "r2", "model"]
    )

    reliever_results_linear_regression = test_model_all_combo(
        "reliever",
        relief_pitching_data,
        linear_regression,
        ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
        ["mse", "r2", "model"]
    )

    position_results_linear_regression = test_model_all_combo(
        "position",
        batting_data,
        linear_regression,
        ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
        ["mse", "r2", "model"]
    )

    starter_results_svr = test_model_all_combo(
        "starter", 
        starting_pitching_data,
        support_vector_regression, 
        ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
        ["mse", "r2", "model"]
    )

    reliever_results_svr = test_model_all_combo(
        "reliever",
        relief_pitching_data,
        support_vector_regression,
        ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
        ["mse", "r2", "model"]
    )

    position_results_svr = test_model_all_combo(
        "position",
        batting_data,
        support_vector_regression,
        ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
        ["mse", "r2", "model"]
    )

    # NOTE: Lasso models are now not built, as their R2 and MSE are considerably worse than the other models
    # starter_results_lasso = test_model_all_combo_with_alpha(
    #     "starter", 
    #     starting_pitching_data, 
    #     lasso_regression, 
    #     ["GS", "age", "service time", "W-L%", "ERA", "WHIP", "SO"],
    #     ["mse", "r2", "model"]
    # )

    # reliever_results_lasso = test_model_all_combo_with_alpha(
    #     "reliever",
    #     relief_pitching_data,
    #     lasso_regression,
    #     ["G", "age", "service time", "SV", "ERA", "WHIP", "SO"],
    #     ["mse", "r2", "model"]
    # )

    # position_results_lasso = test_model_all_combo_with_alpha(
    #     "position",
    #     batting_data,
    #     lasso_regression,
    #     ["G", "age", "service time", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"],
    #     ["mse", "r2", "model"]
    # )