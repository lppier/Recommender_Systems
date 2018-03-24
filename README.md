# Exploration of Recommender Systems (Using Surprise Python Package)

## Nearest Neighbours - Top Recommendations

**Nearest Neighbours of Item - nearest_neighbours.py**

This code demonstrates how to get the 10 most related items (nearest neighbours) to the movie item "Clockwork Orange". 
Uses item-item approach.
    
## User and Item Based Collaborative Filtering

**User and Item based collaborative filtering - collaborative_filtering_exploration.py**

Algorithms used: http://surprise.readthedocs.io/en/stable/knn_inspired.html

This code demonstrates how collaborative filtering for user-based and item-based methods can be done
in Python. 

Advantages of item-based filtering over user-based filtering : 

1) **Scales Better** : User-based filtering does not scale well as user likes/interests may change frequently. Hence, 
the recommendation needs to be re-trained frequently. 

2) **Computationally Cheaper** : In many cases, there are way more users than items. It makes sense to use item-based
filtering in this case. 

A famous example of item-based filtering is **Amazon's** recommendation engine. 

https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf

## Utilizing Grid Search

**Grid Search on KNNMeans Algorithm - grid_search.py**

An extensive grid search to find the best hyper-parameters for KNNMeans on this dataset. **Warning**: Takes a long time!

Results:
````
{'k': 35, 'bsl_options': {'n_epochs': 1, 'reg_i': 40, 'method': 'als', 'reg_u': 25}, 'sim_options': {'user_based': False, 'name': 'pearson_baseline'}}
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
Item-based Model : Test Set
RMSE: 0.9129
````