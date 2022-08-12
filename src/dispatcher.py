from termios import N_SLIP
from sklearn import ensemble
from lightgbm import LGBMClassifier


MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "gbm": ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.075, max_features=2, max_depth=2, random_state=0),
    "lgbm": LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.07, n_estimators=1000, 
                        min_child_weight=0.01, colsample_bytree=0.5, random_state=0)
}
