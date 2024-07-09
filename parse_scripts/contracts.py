import os
import pandas as pd

DIRECTORY = "../data/contract_data"
FINAL_DATASET = "../dataset/contracts.csv"

contracts_df = pd.DataFrame(columns=["first_name", "last_name", "team", "year", "position", "age", "service time", "agent", "value"])

min_salaries = {
    "2020" : 0.5635,
    "2021" : 0.5705,
    "2022" : 0.7,
    "2023" : 0.72,
    "2024" : 0.74
}

for filename in os.listdir(DIRECTORY):
    filename_split = filename.split(".")[0].split()
    team = filename_split[1]
    year = f"20{filename_split[2]}"
    filepath = os.path.join(DIRECTORY, filename)
    df = pd.read_csv(filepath)
    if int(year) >= 2020:
        i = 8
        while pd.notna(df.iloc[i, 1]):
            print(df.iloc[i, 0])
            if df.iloc[i, 12] != "forfeited":
                contracts_df.loc[len(contracts_df)] = {
                    "first_name" : df.iloc[i, 0].split(", ")[1],
                    "last_name" : df.iloc[i, 0].split(", ")[0],
                    "team" : team,
                    "year" : year,
                    "position" : df.iloc[i, 1],
                    "age" : df.iloc[i, 5],
                    "service time" : df.iloc[i, 6],
                    "agent" : df.iloc[i, 8],
                    "value" : int(df.iloc[i, 12][1:].replace(",", "")) / pow(10, 6) if pd.notna(df.iloc[i, 12]) else min_salaries[year]
                }
            i += 1
    else:
        print("old spreadsheet format")
contracts_df.to_csv(FINAL_DATASET)