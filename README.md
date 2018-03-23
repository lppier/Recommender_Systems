# Exploration of Recommender Systems Using Surprise Python Package

**User and Item based collaborative filtering - collaborative_filtering_exploration.py**

This code demonstrates how collaborative filtering for user-based and item-based methods can be done
in Python. 

Advantages of item-based filtering over user-based filtering : 

1) Scales Better : User-based filtering does not scale well as user likes/interests may change frequently. Hence, 
the recommendation needs to be re-trained frequently. 

2) Computationally Cheaper : In many cases, there are way more users than items. It makes sense to use item-based
filtering in this case. 

A famous example of item-based filtering is Amazon's recommendation engine. 

https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
