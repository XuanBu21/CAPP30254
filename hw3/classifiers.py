'''
CAPP30254 HW3
Xuan Bu
Build Classifiers
'''
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,\
                             RandomForestClassifier
from sklearn.metrics import *


# Constants
THRESHOLD = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
SEED = 21
GRID = {'KNN': {'n_neighbors': [5, 10, 20, 50], 'metric':\
                ['euclidean', 'manhattan', 'minkowski'],\
                'weights': ['uniform', 'distance']},\
        'DT': {'max_depth': [5, 20, 50, 100], 'criterion': ['gini',\
                'entropy'], 'min_samples_leaf': [2, 10]},\
        'LR': {'penalty': ['l1', 'l2'], 'C': [10**-2, 10**-1, 1, 10, 10**2]},\
        'SVM': {'C': [10**-2, 10**-1, 1, 10, 10**2]},\
        'RF': {'n_estimators': [5, 10, 20], 'max_depth': [5, 20, 50, 100]},\
        'AB': {'n_estimators': [1, 10, 100]},\
        'BAG': {'n_estimators': [5, 10, 20], 'max_samples': [0.3, 0.5, 0.7]}}



# knn
def build_knn(n_neighbors, metric, weights):
    '''
    Given different parameters, build a k-nearest neighbors model.
    Inputs:
        n_neighbors: (int) number of neighbors
        weight: (str) weight function used in prediction
        metric: (str) distance metric to use for the tree
    Returns:
        knn model        
    '''
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,\
                               weights=weights)


# decision tree
def build_dt(criterion, max_depth, min_samples_leaf):
    '''
    Given different parameters, build a decision tree model.
    Inputs:
        criterion: (str) function to measure the quality of a split
        max_depth: (int) maximum depth of decision tree
        min_samples_leaf: (int) minimum number of samples required
          to be at a leaf node
    Returns:
        decision tree model        
    '''
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,\
                        min_samples_leaf=min_samples_leaf, random_state=SEED)


# lr
def build_lr(penalty, c):
    '''
    Given different parameters, build a logistic regression model.
    Inputs:
        penalty: (str) used to specify the norm used in the penalization
        c: (float) inverse of regularization strength       
    Returns:
        logistic regression model
    '''
    return LogisticRegression(penalty=penalty, C=c, random_state=SEED)


# svm
def build_svm(c):
    '''
    Given different parameters, build a SVM model.
    Inputs:
        c: (float) strength of regularization        
    Returns:
        SVM model
    '''
    return LinearSVC(C=c, random_state=SEED)


# RF
def build_rf(n_estimators, max_depth):
    '''
    Given different parameters, build a random forest model.
    Inputs:
        n_estimators: the number of trees in the forest
        max_depth: (int) maximum depth of the tree
    Returns:
        random forest model
    '''
    return RandomForestClassifier(n_estimators=n_estimators,\
                        max_depth=max_depth, random_state=SEED)


    for n in GRID['RF']['n_estimators']:
        print()
        print('number of estimators is {}'.format(n))
        for depth in GRID['RF']['max_depth']:
            print('maximum depth is {}'.format(depth))
            rf = build_rf(n, depth)
            trail_threshold(rf, train_data, train_target, test_data, test_target)


# AB
def build_ab(n_estimators):
    '''
    Given different parameters, build a ada boosting model.
    Inputs:
        n_estimators: (int) maximum number of estimators at which
          boosting is terminated
    Returns:
        ada boosting model
    '''
    return AdaBoostClassifier(n_estimators=n_estimators, random_state=SEED)


# BAG
def build_bag(n_estimators, max_samples):
    '''
    Given different parameters, build a ada boosting model.
    Inputs:
        n_estimators: (int) number of base estimators in the ensemble
        max_samples: (int) number of samples to draw from X to train
          each base estimator
    Returns:
        bagging model
    '''
    return BaggingClassifier(n_estimators=n_estimators,\
                    max_samples=max_samples, random_state=SEED)

    for n in GRID['BAG']['n_estimators']:
        print()
        print('number of estimators is {}'.format(n))
        for m in GRID['BAG']['max_samples']:
            print('number of maximum samples is {}'.format(m))
            bag = build_bag(n, m)
            trail_threshold(bag, train_data, train_target, test_data, test_target)


# Set up different thresholds
def trail_threshold(model, train_data, train_target, test_data, test_target):
    '''
    Conduct experiments with different models under different threshold.
    Inputs:
        model: model
        train_data: (dataframe) training data of variables of interest
        train_target: (dataframe) training data of target variable
        test_data: (dataframe) testing data of variables of interest
        test_target: (dataframe) testing data of target variable
    '''
    model.fit(train_data, train_target)
    pred_scores = model.predict_proba(test_data)
    for threshold in THRESHOLD:
        pred_label = [1 if x[1] > threshold else 0 for x in pred_scores]
        print('(Threshold: {}), the total number of the'
                'predicted is {}, F1 is {:.2f}'.format(threshold,\
                sum(pred_label), f1_score(test_target, pred_label)))

