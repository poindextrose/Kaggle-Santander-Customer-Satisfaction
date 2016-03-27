import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

training_features = result1.loc[training_indices].drop(['class', 'group', 'guess'], axis=1)
training_class_vals = result1.loc[training_indices, 'class'].values
if len(training_features.columns.values) == 0:
    result1 = result1.copy()
else:
    selector = SelectFwe(f_classif, alpha=0.05)
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    result1 = result1[mask_cols]

# Perform classification with a k-nearest neighbor classifier
knnc2 = KNeighborsClassifier(n_neighbors=2)
knnc2.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result2 = result1.copy()
result2['knnc2-classification'] = knnc2.predict(result2.drop('class', axis=1).values)
