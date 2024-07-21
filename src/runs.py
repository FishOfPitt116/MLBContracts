import pandas as pd

def test_model_all_combo(label, data, model, predictors, metrics):
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
    results.to_csv(f"../model_results/{label}_{model.__name__}.csv")
    return results

def find_best_model_combo(results, metric, high_val_better=True):
    if high_val_better:
        print(results.iloc[results[metric].idxmax()])
    else:
        print(results.iloc[results[metric].idxmin()])