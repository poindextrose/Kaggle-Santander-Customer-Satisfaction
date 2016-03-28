from tpot import TPOT
from sklearn.cross_validation import train_test_split
import numpy as np
from datetime import datetime

out_filename = "tpot santander " + str(datetime.now()) + ".py"

X_all = np.load(open("X_all.npy", "rb"))
y_all = np.load(open("y_all.npy", "rb"))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                            train_size=0.80, test_size=0.20)

tpot = TPOT(generations=100, verbosity=2)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
tpot.export(out_filename)
