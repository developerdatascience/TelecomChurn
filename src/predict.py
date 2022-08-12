import pandas as pd
import numpy as np
import joblib
import os


def predict(test_data_path: str, model_type: str, model_path: str):
    df = pd.read_csv(test_data_path)
    test_idx = df["customerID"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path,f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /=5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["customerID", "Churn"])
    return sub

if __name__=="__main__":
    submission = predict(test_data_path = "inputs/test_data.csv",
                        model_type="randomforest",
                        model_path="models/")

    submission.to_csv(f"output/rf_submission.csv", index=False)


    