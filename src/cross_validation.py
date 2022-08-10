import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from math import sqrt

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("inputs/Processed_data.csv")
print("========Splitting the data in train and test==============")
X = df.drop(["Churn", "customerID"], axis=1)
y = df['Churn'].values

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train: ",X_train.shape, "X_test: ",X_cv.shape, "y_train: ",y_train.shape, "y_test: ",y_cv.shape)

err = []
y_pred_tot = []
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
i=1

for train_index, test_index in kf.split(X,y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.07, n_estimators=1000, 
                        min_child_weight=0.01, colsample_bytree=0.5, random_state=0)
    lgbm.fit(X_train, y_train, eval_set=[(X_cv, y_cv)], eval_metric='auc', early_stopping_rounds=100, verbose=100)
    preds = lgbm.predict_proba(X_test)[:,-1]

    print("ROC_AUC SCORE: ", roc_auc_score(y_test, preds))
    err.append(roc_auc_score(y_test, preds))
    p = lgbm.predict_proba(X_test)[:,-1]
    print(f'-------------Fold{i} completed-----------------')
    i=i+1
    y_pred_tot.append(p)

err_avg = np.mean(err,0)
print("Average Error: ", err_avg)






