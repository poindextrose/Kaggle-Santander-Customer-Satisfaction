from sklearn.grid_search import RandomizedSearchCV
import xgboost.sklearn as xgb
from pymongo import MongoClient
from scipy import stats
import numpy as np
import time
# import warnings
# warnings.simplefilter('ignore', DeprecationWarning)

n_jobs = 1
n_iter = 1

client = MongoClient("mongodb://optimizer:bOQ0QxKl1oKX@ds015760.mlab.com:15760/santander")
db = client.santander
scores = db.xgb_scores

kfolds = 10

# scipy.stats are seeded with np.random.seed
np.random.seed = int(time.time())

X = np.load(open("X_all.npy", "rb"))
y = np.load(open("y_all.npy", "rb"))
X_unknown = np.load(open("X_test_all.npy", "rb"))

parameters = {
    'max_depth': stats.randint(2, 15),
    'learning_rate': stats.expon(1e-3, 0.1),
    'n_estimators': stats.randint(10,500),
    'gamma': stats.uniform(0, 1),
    'min_child_weight': stats.uniform(0, 4),
    'max_delta_step': stats.uniform(0,4),
    'subsample': stats.uniform(0.5, 0.5),
    'colsample_bytree': stats.uniform(0.5, 0.5),
    'colsample_bylevel': stats.uniform(0.5, 0.5),
    # 'reg_alpha': stats.uniform(0, 0.1),
    'reg_lambda': stats.uniform(0.5, 1),
}
clf = RandomizedSearchCV(xgb.XGBClassifier(scale_pos_weight=25.3),
                         parameters,
                         n_iter=n_iter,
                         cv=kfolds,
                         scoring='roc_auc',
                         n_jobs=n_jobs,
                         verbose=2,
                         refit=False,
                         error_score=0)

def run_iteration():
    clf.fit(X, y)
    grid_scores = [dict(score._asdict()) for score in clf.grid_scores_]
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
