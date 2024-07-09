import os
import pandas as pd
import re

DIRECTORY = "../data/stats_data"
FINAL_BATTING_DATASET = "../dataset/batting_stats.csv"
FINAL_PITCHING_DATASET = "../dataset/pitching_stats.csv"

contracts_df = pd.read_csv("../dataset/contracts.csv")

batting_stats_df = pd.DataFrame(columns=['Index', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'TB', 'GDP', 'HBP', 'SH', 'SF', 'IBB'])
pitching_stats_df = pd.DataFrame(columns=['Index', 'W', 'L', 'W-L%', 'ERA', 'G', 'GS', 'GF', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'SO', 'HBP', 'BK', 'WP', 'BF', 'ERA+', 'FIP', 'WHIP', 'H9', 'HR9', 'BB9', 'SO9', 'SO/W'])

for filename in os.listdir(DIRECTORY):
    filename_split = filename.split(" ")
    if len(filename_split) == 1:
        continue
    year = int(filename_split[0])
    type = filename_split[len(filename_split)-1].split(".")[0]
    # type is either "Batting" or "Pitching"
    filepath = os.path.join(DIRECTORY, filename)
    df = pd.read_csv(filepath, delimiter=";")
    for index, row in df.iterrows():
        name = row["Name"]
        name_split = name.split("�")
        first_name = name_split[0]
        last_name = re.sub("[#*]", "", name_split[1])
        print(first_name, last_name)
        q = contracts_df.query(f"(first_name==\"{first_name}\") and (last_name==\"{last_name}\") and (year=={year+1})")
        temp = "v"
        while len(name_split) > 2 or len(q) < 1:
            print(f"Player has inconsistent or unidentifiable name: {name_split}")
            temp = input("Type the player's first name (or s to skip): ")
            if temp == "s":
                break
            first_name = temp
            last_name = input("Type the player's last name: ")
            q = contracts_df.query(f"(first_name=='{first_name}') and (last_name=='{last_name}') and (year=={year+1})")
            name_split = [first_name, last_name]
        print(q)
        row["Index"] = q.iloc[0, 0] if temp != "s" else -1
        if type == "Batting":
            batting_stats_df.loc[len(batting_stats_df)] = {
                x : row[x] for x in batting_stats_df.columns
            }
            print(batting_stats_df)
        else:
            pitching_stats_df.loc[len(pitching_stats_df)] = {
                x : row[x] for x in pitching_stats_df.columns
            }
            print(pitching_stats_df)
batting_stats_df.to_csv(FINAL_BATTING_DATASET)
pitching_stats_df.to_csv(FINAL_PITCHING_DATASET)