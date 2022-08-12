import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    df = pd.read_csv("inputs/Processed_data.csv")
    df["kfold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df["Churn"].values)):
        print("train: ", len(train_idx), "valid: ", len(val_idx))
        df.loc[val_idx, "kfold"] = fold
    
    df.to_csv("inputs/train_folds.csv", index=False)