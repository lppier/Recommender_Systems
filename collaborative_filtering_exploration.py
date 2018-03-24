from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset  UserID::MovieID::Rating::Timestamp
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.15)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

# we can now query for specific predicions
uid = str(196)  # raw user id
iid = str(302)  # raw item id

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

# run the trained model against the testset
test_pred = algo.test(testset)

# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

# if you wanted to evaluate on the trainset
print("User-based Model : Training Set")
train_pred = algo.test(trainset.build_testset())
accuracy.rmse(train_pred)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)

# run the trained model against the testset
test_pred = algo.test(testset)

# get RMSE
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

# if you wanted to evaluate on the trainset
print("Item-based Model : Training Set")
train_pred = algo.test(trainset.build_testset())
accuracy.rmse(train_pred)

# Testing ALS baseline
bsl_options = {'method': 'als',
               'n_epochs': 50,
               'reg_u': 5,
               'reg_i': 1
               }

# Testing SGD regressor
# bsl_options = {'method': 'sgd',
#                'learning_rate': .008,
#                }


algo = KNNWithMeans(k=50, bsl_options=bsl_options, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)
test_pred = algo.test(testset)
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
#
# param_grid = {'k': [10, 20],
#               'sim_options': {'name': ['msd', 'cosine'],
#                               'min_support': [1, 5],
#                               'user_based': [False]}
#               }

# from surprise.model_selection import GridSearchCV
# gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
# gs.fit(data)