from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from models import read_model_from_file, read_scaler_from_file
import pandas as pd

model_path = 'best_model/position_support_vector_regression.pkl'
position_results_svr: SVR = read_model_from_file(model_path)

scaler_path = 'scalers/batting_scaler.pkl'
batting_scaler: MinMaxScaler = read_scaler_from_file(scaler_path)

print(position_results_svr.feature_names_in_)

juan_soto = pd.DataFrame({
    "year": [2025],
    "service time": [6.134],
    "value": [51.875],
    "Index" : [-1],
    "G": [157],
    "PA" : [713],
    "AB": [576],
    "R" : [128],
    "H": [166],
    "2B": [31],
    "3B": [4],
    "HR": [41],
    "RBI": [109],
    "SB": [7],
    "CS": [4],
    "BB": [129],
    "SO": [119],
    "BA": [0.288],
    "OBP": [0.419],
    "SLG": [0.569],
    "OPS": [0.989],
    "OPS+": [178],
    "TB": [328],
    "GDP": [10],
    "HBP": [4],
    "SH": [0],
    "SF": [4],
    "IBB": [2]
})
juan_soto_normalized = pd.DataFrame(batting_scaler.transform(juan_soto), columns=juan_soto.columns)
# current issue: position_results_svr is a dataframe where it should be the model
prediction = position_results_svr.predict(juan_soto_normalized[position_results_svr.feature_names_in_])
juan_soto_normalized['value'] = prediction
print(f"Juan Soto Predicted Value by SVR: {batting_scaler.inverse_transform(juan_soto_normalized)[0][juan_soto_normalized.columns.get_loc('value')]}")