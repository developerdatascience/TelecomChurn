import os
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher



TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


FOLD_MAPPING = {
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3, 4, 5],
    2: [0, 1, 3, 4, 5],
    3: [0, 1, 2, 4, 5],
    4: [0, 1, 2, 3, 5],
    5: [0, 1, 3, 3, 4]
}


if __name__=="__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)

    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    ytrain = train_df.Churn.values
    yvalid = valid_df.Churn.values

    train_df = train_df.drop(["customerID", "Churn", "kfold"], axis=1)
    valid_df = valid_df.drop(["customerID", "Churn", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = dict()
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        test_df.loc[:, c] = test_df.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist()+
                valid_df[c].values.tolist()+
                test_df[c].values.tolist()
                )
        train_df.loc[:, c] = lbl.transform(train_df[c].values)
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values)
        label_encoders[c] = lbl

    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]

    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")






    




