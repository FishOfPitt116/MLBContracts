# from sklearn import svm

# class Example:
#     def __init__(self, a1, a2, a3):
#         self.attr1 = a1
#         self.attr2 = a2
#         self.attr3 = a3

#     def data_to_list(self):
#         return [self.attr1, self.attr2, self.attr3]

# e1 = Example(0, 0, 0)
# e2 = Example(2, 2, 2)
# X = [e1.data_to_list(), e2.data_to_list()]
# y = [0.5, 2.5]
# regression = svm.SVR()
# regression.fit(X, y)
# print(regression.predict([[1, 1, 1]]))

# def contract(position, age, service_time, stats):
#     pass

import os
import pandas as pd

DATASET_DIR = "../dataset"

BATTING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "batting_stats.csv"))
CONTRACTS = pd.read_csv(os.path.join(DATASET_DIR, "contracts.csv"))
PITCHING_STATS = pd.read_csv(os.path.join(DATASET_DIR, "pitching_stats.csv"))

BATTING_CONTRACTS = pd.DataFrame(columns=CONTRACTS.columns)
PITCHING_CONTRACTS = pd.DataFrame(columns=CONTRACTS.columns)

def filter_by_position(pos):
    return CONTRACTS.query(f'position == \"{pos}\"')

if __name__ == "__main__":
    for position in CONTRACTS["position"].unique():
        pos_df = filter_by_position(position)
        if position in ["rhp-s", "lhp-s", "rhp-c", "lhp-c", "rhp", "lhp"]:
            PITCHING_CONTRACTS = PITCHING_CONTRACTS._append(pos_df, ignore_index=False)
        elif "lhp" in position or "rhp" in position:
            pass
        else:
            BATTING_CONTRACTS = BATTING_CONTRACTS._append(pos_df, ignore_index=False)
    batting_data = pd.merge(BATTING_CONTRACTS, BATTING_STATS, left_on="Unnamed: 0", right_on="Index")
    pitching_data = pd.merge(PITCHING_CONTRACTS, PITCHING_STATS, left_on="Unnamed: 0", right_on="Index")
    print(batting_data.columns)
    print(pitching_data.columns)