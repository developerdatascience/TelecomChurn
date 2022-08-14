from sklearn import ensemble


MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "gbm": ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, max_features=2, max_depth=2, random_state=0)
}