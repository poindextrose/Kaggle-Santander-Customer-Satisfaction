from sklearn.grid_search import RandomizedSearchCV
import xgboost.sklearn as xgb
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
from scipy import stats
import numpy as np
import time
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

n_jobs = 1
n_iter = 1

client = MongoClient("mongodb://optimizer:bOQ0QxKl1oKX@ds015760.mlab.com:15760/santander")
db = client.santander
scores = db.xgb_fs_scores

kfolds = 10

# scipy.stats are seeded with np.random.seed
np.random.seed = int(time.time())

feature_importances = np.load(open("gbc_feature_importances.npy", "rb"))
X = np.load(open("X_all.npy", "rb"))[:, feature_importances > 0]
y = np.load(open("y_all.npy", "rb"))
X_test = np.load(open("X_test_all.npy", "rb"))[:, feature_importances > 0]
feature_importances = feature_importances[feature_importances > 0]

def score(X, y):
    score = feature_importances
    pvalues = np.ones(len(score)) * 0.01 # fake this
    return score, pvalues

parameters = {
    'selector__percentile': stats.randint(10, 100),
    'xgb__max_depth': stats.randint(2, 15),
    'xgb__learning_rate': stats.expon(1e-3, 0.1),
    'xgb__n_estimators': stats.randint(10,500),
    'xgb__gamma': stats.uniform(0, 1),
    'xgb__min_child_weight': stats.uniform(0, 4),
    'xgb__max_delta_step': stats.uniform(0,4),
    'xgb__subsample': stats.uniform(0.5, 0.5),
    'xgb__colsample_bytree': stats.uniform(0.5, 0.5),
    'xgb__colsample_bylevel': stats.uniform(0.5, 0.5),
    'xgb__reg_lambda': stats.uniform(0.5, 1),
 }

selector = SelectPercentile(score, percentile=50)
pipeline = Pipeline(steps=[('selector', selector), ('xgb', xgb.XGBClassifier(scale_pos_weight=25.3))])
search = RandomizedSearchCV(pipeline,
                         parameters,
                         n_iter=n_iter,
                         cv=kfolds,
                         scoring='roc_auc',
                         n_jobs=n_jobs,
                         verbose=2,
                         refit=False,
                         error_score=0)

def run_iteration():
    search.fit(X, y)
    grid_scores = [dict(score._asdict()) for score in search.grid_scores_]
    for score in grid_scores:
        score['cv_scores'] = score.pop('cv_validation_scores').tolist()
        score['mean_score'] = score.pop('mean_validation_score')
        score['params'] = score.pop('parameters')
    scores.insert_many(grid_scores)
    print(scores)

try:
    while True:
        run_iteration()
except KeyboardInterrupt:
    print('interrupted!')
