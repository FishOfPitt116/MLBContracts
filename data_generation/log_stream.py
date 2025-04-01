import pandas as pd

class LogStream:
    def __init__(self, name: str):
        self.name = name

    def write(self, message: str):
        print(f"[{self.name}]: {message}")

    def player_mapping(self, first_name: str, last_name: str, id_df: pd.DataFrame, index: int, iloc: bool = False, low_confidence: bool = False):
        self.write("START PLAYER MAPPING")
        self.write(f"Spotrac Name: {first_name} {last_name} ")
        self.write(f"Fangraphs Name: {id_df['name_first'][index] if not iloc else id_df['name_first'].iloc[index]} {id_df['name_last'][index] if not iloc else id_df['name_last'].iloc[index]}")
        if 0 in id_df.index:
            self.write("LOW CONFIDENCE MATCH")
        self.write(f"Fangraphs ID: {id_df['key_fangraphs'][index] if not iloc else id_df['key_fangraphs'].iloc[index]}")
        self.write("END PLAYER MAPPING")