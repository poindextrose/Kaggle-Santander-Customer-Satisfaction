import sys

if len(sys.argv) != 3:
    print("Pass number of parallel jobs and iterations as arguments")
    print("Example: python " + sys.argv[0] + " 8 10")
    print("Uses 8 cores and runs 10 experiments")
    sys.exit();

from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
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
scores = db.gbc_fs_2_scores

kfolds = 10

# scipy.stats are seeded with np.random.seed
np.random.seed = int(time.time())

feature_importances = np.load(open("gbc_feature_importances.npy", "rb"))
X = np.load(open("X_all.npy", "rb"))[:, feature_importances > 0]
y = np.load(open("y_all.npy", "rb"))
feature_importances = feature_importances[feature_importances > 0]

def score(X, y):
    score = feature_importances
    pvalues = np.ones(len(score)) * 0.01 # fake this
    return score, pvalues

parameters = {'selector__percentile': stats.randint(24, 100),
              'gbc__loss': ['deviance', 'exponential'],
              'gbc__learning_rate': stats.uniform(0.005, 0.15),
              'gbc__n_estimators': stats.randint(60,1000),
              'gbc__max_depth': stats.randint(4, 50),
              'gbc__min_samples_split': stats.randint(2, 10),
              'gbc__min_samples_leaf': stats.randint(1, 6),
              'gbc__min_weight_fraction_leaf': stats.uniform(0., 0.09),
              'gbc__subsample': stats.uniform(0.25, 0.75),
              'gbc__max_features': ["sqrt", "log2", None]
             }

selector = SelectPercentile(score)
pipeline = Pipeline(steps=[('selector', selector), ('gbc', GradientBoostingClassifier())])
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
