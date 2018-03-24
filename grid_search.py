from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

# Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.15)

param_grid = {'k': [5, 10, 20, 30, 40, 50],
              'bsl_options': {'method': ['als'],
                              'n_epochs': [5, 10, 30, 60, 80],
                              'reg_u': [1, 5, 10, 15, 20],
                              'reg_i': [1, 5, 10, 15, 20]
                              },
              'sim_options': {'name': ['pearson_baseline', 'cosine'],
                              'user_based': [False, True]}
              }

# from surprise.model_selection import GridSearchCV
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator['rmse']
algo.fit(trainset)
test_pred = algo.test(testset)
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

