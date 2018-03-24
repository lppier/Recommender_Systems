from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.15)

param_grid = {'k': [20, 25, 30, 35, 40],
              'bsl_options': {'method': ['als'],
                              'n_epochs': [1, 3, 5, 10],
                              'reg_u': [15, 20, 25],
                              'reg_i': [25, 30, 35, 40]
                              },
              'sim_options': {'name': ['pearson_baseline', 'cosine'],
                              'user_based': [False]}
              }

gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator['rmse']
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
algo.fit(trainset)
test_pred = algo.test(testset)
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

