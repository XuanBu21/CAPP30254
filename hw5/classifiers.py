'''
CAPP30254 HW5
Xuan Bu
Build Classifiers
'''

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,\
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import *
from sklearn.grid_search import ParameterGrid


# Constants
LARGE_GRID = {'KNN': {'n_neighbors': [5, 10, 20, 50], 'metric':\
                ['euclidean', 'manhattan', 'minkowski'],\
                'weights': ['uniform', 'distance']},\
        'DT': {'max_depth': [5, 20, 50, 100], 'criterion': ['gini',\
                'entropy'], 'min_samples_leaf': [2, 10]},\
        'ET': {'n_estimators': [5, 10, 20, 50], 'max_depth': [5, 20, 50, 100],\
               'criterion': ['gini', 'entropy']},\
        'LR': {'penalty': ['l1', 'l2'], 'C': [10**-2, 10**-1, 1, 10, 10**2]},\
        'SVM': {'C': [10**-2, 10**-1, 1, 10, 10**2]},\
        'RF': {'n_estimators': [1, 10, 100], 'max_depth': [5, 20, 50, 100],\
              'min_samples_split': [2, 5, 10]},\
        'AB': {'n_estimators': [1, 10, 100], 'algorithm': ['SAMME', 'SAMME.R']},\
        'GB': {'n_estimators': [1, 10, 100], 'learning_rate' : [0.01, 0.1, 0.5],\
               'max_depth': [1, 5, 50, 100]},\
        'BAG': {'n_estimators': [5, 10, 20], 'max_samples': [0.3, 0.5, 0.7]}}

TEST_GRID = {'KNN': {'n_neighbors': [5], 'metric': ['manhattan'], 'weights': ['uniform']},\
        'DT': {'max_depth': [5], 'criterion': ['gini'], 'min_samples_leaf': [2]},\
        'ET': {'n_estimators': [5], 'max_depth': [5], 'criterion': ['gini']},\
        'LR': {'penalty': ['l2'], 'C': [1]},\
        'SVM': {'C': [1]},\
        'RF': {'n_estimators': [1], 'max_depth': [5], 'min_samples_split': [2]},\
        'AB': {'n_estimators': [1], 'algorithm': ['SAMME', 'SAMME.R']},\
        'GB': {'n_estimators': [1], 'learning_rate' : [0.1], 'max_depth': [5]},\
        'BAG': {'n_estimators': [5], 'max_samples': [0.5]}}


CLASSIFIERS = {'LR': LogisticRegression(), 'KNN': KNeighborsClassifier(),\
               'DT': DecisionTreeClassifier(), 'SVM': LinearSVC(),\
               'RF': RandomForestClassifier(), 'AB':AdaBoostClassifier(),\
               'BAG': BaggingClassifier(), 'ET': ExtraTreesClassifier(),\
               'GB': GradientBoostingClassifier()}

