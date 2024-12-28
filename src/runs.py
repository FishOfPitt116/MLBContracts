from models import read_model_from_file, write_model_to_file
import os
import pandas as pd

def test_model_all_combo(label, data, model, predictors, metrics, override=False):
    path = f"model_results/{label}_{model.__name__}_norm.csv"
    best_model_path = f"best_model/{label}_{model.__name__}.pkl"
    if not os.path.exists("best_model"):
        os.mkdir("best_model")
    if not override and os.path.exists(path):
        return read_model_from_file(best_model_path)
    results = pd.DataFrame(columns=predictors + metrics)
    n = len(predictors)
    for j in range(pow(2, n)):
        record = {}
        run_predictors = []
        for k in range(n):
            if ((1 << k)) & j != 0:
                run_predictors.append(predictors[k])
                record[predictors[k]] = 1
            else:
                record[predictors[k]] = 0
        if len(run_predictors) == 0:
            continue
        stats = model(data, run_predictors, "value")
        for metric in metrics:
            record[metric] = stats[metric]
        results = results._append(record, ignore_index=True)
    best_model_index = find_best_model_combo(results, "r2")
    results.drop(columns=["model"]).to_csv(path)
    write_model_to_file(results.iloc[best_model_index]["model"], best_model_path)
    return results.iloc[best_model_index]["model"]

def test_model_all_combo_with_alpha(label, data, model, predictors, metrics, override=False):
    path = f"model_results/{label}_{model.__name__}_norm.csv"
    best_model_path = f"best_model/{label}_{model.__name__}.pkl"
    if not override and os.path.exists(path):
        return read_model_from_file(best_model_path)
    results = pd.DataFrame(columns=predictors + metrics)
    n = len(predictors)
    for alpha in range(1, 10):
        for j in range(pow(2, n)):
            record = {}
            run_predictors = []
            for k in range(n):
                if ((1 << k)) & j != 0:
                    run_predictors.append(predictors[k])
                    record[predictors[k]] = 1
                else:
                    record[predictors[k]] = 0
            if len(run_predictors) == 0:
                continue
            record['alpha'] = alpha*0.1
            stats = model(data, run_predictors, "value", alpha=alpha*0.1)
            for metric in metrics:
                record[metric] = stats[metric]
            results = results._append(record, ignore_index=True)
    best_model_index = find_best_model_combo(results, "r2")
    results.drop(columns=["model"]).to_csv(path)
    write_model_to_file(results.iloc[best_model_index]["model"], best_model_path)
    return results.iloc[best_model_index]["model"]

def find_best_model_combo(results, metric, high_val_better=True):
    if high_val_better:
        print(results.iloc[results[metric].idxmax()])
        return results[metric].idxmax()
    else:
        print(results.iloc[results[metric].idxmin()])
        return results[metric].idxmin()