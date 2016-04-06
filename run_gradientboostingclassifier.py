import sys

if len(sys.argv) != 3:
    print("Pass number of parallel jobs and iterations as arguments")
    print("Example: python " + sys.argv[0] + " 8 10")
    print("Uses 8 cores and runs 10 experiments")
    sys.exit();

from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from pymongo import MongoClient
from scipy import stats
import numpy as np
import time
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

n_jobs = int(sys.argv[1])
n_iter = int(sys.argv[2])

client = MongoClient("mongodb://optimizer:bOQ0QxKl1oKX@ds015760.mlab.com:15760/santander")
db = client.santander
scores = db.gbc_scores

kfolds = 10

# scipy.stats are seeded with np.random.seed
np.random.seed = int(time.time())

X = np.load(open("X_all.npy", "rb"))
y = np.load(open("y_all.npy", "rb"))
X_unknown = np.load(open("X_test_all.npy", "rb"))

parameters = {'loss': ['deviance', 'exponential'],
              'learning_rate': stats.expon(1e-3, 0.1),
              'n_estimators': stats.randint(10,500),
              'max_depth': stats.randint(2, 15),
              'min_samples_split': stats.randint(2, 6),
              'min_samples_leaf': stats.randint(1, 3),
              'min_weight_fraction_leaf': stats.uniform(0., 0.5),
              'subsample': stats.uniform(0.5, 0.5),
              'max_features': ["sqrt", "log2", None]
             }
clf = RandomizedSearchCV(GradientBoostingClassifier(),
                         parameters,
                         n_iter=n_iter,
                         cv=kfolds,
                         scoring='roc_auc',
                         n_jobs=n_jobs,
                         verbose=1,
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
