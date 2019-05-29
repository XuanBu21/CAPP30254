'''
CAPP30254 HW5
Xuan Bu
Improved Pipeline
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import time
from sklearn import preprocessing
from sklearn.metrics import *
from evaluation import *
from classifiers import *


#########################
# Step 1 Read/Load Data #
#########################

def read_data(filename, cols_to_drop=None):
    '''
    Load data from file.
    Input:
        filename: (str) name of the file loaded
        cols_to_drop: (list) list of unused columns
    Returns:
        dataframe of the file
    '''
    if cols_to_drop is None:
        return pd.read_csv(filename, index_col=0)
    return pd.read_csv(filename, index_col=0)\
             .drop(cols_to_drop, axis=1)


#######################
# Step 2 Explore Data #
#######################

def summary_continuous_vars(df, contin_vars):
    '''
    Given a dataframe and associated continuous variables,
    returns the summary statistics.
    Inputs:
        df: dataframe
        contin_vars: (list) list of continuous variables
    Returns:
        dataframe of the summary statistics
    '''
    return df[contin_vars].describe()


def summary_categorical_vars(df, cat_vars):
    '''
    Given a dataframe and associated categorical variables,
    returns the summary statistics.
    Inputs:
        df: dataframe
        cat_vars: (str) name of categorical variable
    Returns:
        dataframe of the summary statistics
    '''
    return df[cat_vars].value_counts().reset_index()\
                       .rename(columns={'index': 'count'})


def generate_graph(df, contin_vars):
    '''
    Given a dataframe and a certain variable, returns the histograph
    of the variable
    Inputs:
        df: dataframe
        contin_vars: (list) list of continuous variables
    '''
    for var in contin_vars:
        df.hist(column=var)
        plt.title('The Histograph of ' + var)
        plt.xlabel('Value')
        plt.ylabel('Frequence')
        plt.savefig('Hist of ' + var)
        plt.show()
        plt.close()


def generate_corr_graph(df):
    '''
    Given a dataframe then generates the graph of the correlations
    between the variables.
    Inputs:
        df: dataframe
    '''
    sns.heatmap(df.corr(), annot=True, fmt='.2f',\
                            annot_kws={'size': 11}, cmap="Blues")
    plt.title('The Heatmap of Correlation')
    plt.savefig('The Heatmap of Correlation')
    plt.show()
    plt.close()


def count_outliers(df, contin_vars):
    '''
    Given a dataframe and a certain variable, returns the number of
    the outliers of the variable.
    Inputs:
        df: dataframe
        contin_vars: (list) list of continuous variables
    Returns:
        (int) the number of the outliers of the variable
    '''
    outliers = {}
    for var in contin_vars:
        summary = summary_continuous_vars(df, contin_vars)
        var_mean, var_std = summary[var]['mean'], summary[var]['std']
        cut_off = var_std * 3
        lower, upper = var_mean - cut_off, var_mean + cut_off
        outlier = df[(df[var] > upper) | (df[var] < lower)]
        outliers[var] = [outlier.shape[0]]

    return pd.DataFrame(outliers)


#########################
# Step 3 Splitting Data #
#########################

# funtions for this step are in temporal_validation.py


######################################
# Step 4 Imputation/Pre-Process Data #
######################################

def fill_missing(train_data, test_data, var, var_type, filling_value=None):
    '''
    Given training set and tesing set, replace the missing
    values of the variable with median of training set.
    Inputs:
        train_data: (dataframe) training set
        test_data: (dataframe) testing set
        var: (str) name of variable
        var_type: (str) 'continuous' or 'categorical'
        filling_value: (str) value to fill missing value
    '''
    if var_type == 'continuous':
        median = train_data[var].median()
        train_data[var].fillna(median, inplace=True)
        test_data[var].fillna(median, inplace=True)
        return None
    if var_type == 'categorical':
        train_data[var].fillna(filling_value, inplace=True)
        test_data[var].fillna(filling_value, inplace=True)
        return None


def normalize_features(df, contin_vars):
    '''
    Given a dataframe, then normalize continuous features by
    scaling each continuous feature.
    Inputs:
        df: a dataframe
    Returns:
        a dataframe with scaled feature
    '''
    scaler = preprocessing.MinMaxScaler()
    df[contin_vars] = scaler.fit_transform(df[contin_vars])


############################
# Step 5 Generate Features #
############################

def discretize_continuous_var(df, var, bins, labels):
    '''
    Given a dataframe and a certain continuous variable, then
    convert it to discrete variable.
    Inputs:
        df: dataframe
        var: (str) a certain variable
        bins: (int) number of discrete intervals
        labels: (list) the categories for the returned bins
    Returns:
        dataframe with discretized variables
    '''
    discretized_var = 'discretized_' + var
    df[discretized_var] = pd.cut(df[var], bins=bins, labels=labels)
    return df


def create_dummies(train_data, test_data, cols):
    '''
    Given a dataframe and a set of categorical variables, then
    create corresponding dummy variables.
    Inputs:
        train_data: (dataframe) training set
        test_data: (dataframe) testing set
        cols: (list) a set of categorical variables
    Returns:
        training set with dummies and testing set with dummies
    '''
    for col in cols:
        values = list(train_data[col].unique())
        for val in values:
            train_data['{}_{}'.format(col, val)] = \
                train_data[col].apply(lambda x: 1 if x == val else 0)
            if val in list(test_data[col].unique()):
                test_data['{}_{}'.format(col, val)] = \
                    test_data[col].apply(lambda x: 1 if x == val else 0)
            else:
                test_data['{}_{}'.format(col, val)] = 0
    train_data = train_data.drop(cols, axis=1)
    test_data = test_data.drop(cols, axis=1)

    return train_data, test_data


###########################
# Step 6 Build Classifier #
###########################

def build_classifier(train_data, test_data, train_target, test_target, ks):
    '''
    Modeling the training data to build the classifier with different models.
    Inputs:
        classifier: (object) a kind of classifier
        train_data: (dataframe) of variables of interest
        train_target: (dataframe) of target variable
        ks: (list) of integers
    '''
    for idx, clf in CLASSIFIERS.items():
        print('Model is : {}'.format(idx))
        paras = LARGE_GRID[idx]
        grid = ParameterGrid(paras)
        for p in grid:
            print('Parameter is {}'.format(p))
            model = clf.set_params(**p)
            model.fit(train_data, train_target)
            if idx == 'SVM':
                pred_scores = model.decision_function(test_data)
            else:
                pred_scores = model.predict_proba(test_data)[:,1]
            for k in ks:
                print('k is {}'.format(k))
                print('Recall is {}'.format(recall_at_k(test_target, pred_scores, k)))
                print('Precision is {}'.format(precision_at_k(test_target, pred_scores, k)))
                print()


##############################
# Step 7 Evaluate Classifier #
##############################

def evaluate_classifier(best_grid, train_data, test_data, train_target, test_target, ks):
    '''
    Given a dataframe, build classifiers with training data set, then 
    evaluate classifiers with testing data set.
    Inputs:
        best_grid: (dict) parameter grid that gives the best results
          on the hold out data
        train_data: (dataframe) training data of variables of interest
        train_target: (dataframe) training data of target variable
        test_data: (dataframe) testing data of variables of interest
        test_target: (dataframe) testing data of target variable
        ks: (list) of integers
    '''
    models = collections.defaultdict(dict)
    selected_clf = list(best_grid.keys())
    clfs = {n: clf for n, clf in CLASSIFIERS.items() if n in selected_clf}
    for name, clf in clfs.items():
        para = best_grid[name]
        models[name]['best_paras'] = para
        train_start = time.time()
        model = clf.set_params(**para)
        model.fit(train_data, train_target)
        
        models[name]['train_time'] = time.time() - train_start
        if name == 'SVM':
            pred_scores = model.decision_function(test_data)
        else:
            pred_scores = model.predict_proba(test_data)[:,1]

    # plot ROC, precision, and recall
        generate_precision_recall_curve(pred_scores, test_target, name)
        generate_ROC_graph(pred_scores, test_target, name)

    # evaluate classifiers at different percent of projects predicted true
        for k in ks:
            a_at_k = 'a_at_' + str(k)
            p_at_k = 'p_at_' + str(k)
            r_at_k = 'r_at_' + str(k)
            roc_at_k = 'roc_at_' + str(k)
            models[name][a_at_k] = accuracy_at_k(test_target, pred_scores, k)
            models[name][p_at_k] = precision_at_k(test_target, pred_scores, k)
            models[name][r_at_k] = recall_at_k(test_target, pred_scores, k)
            models[name][roc_at_k] = roc_auc_at_k(test_target, pred_scores, k)

    # convert dictionary to dataframe
    df = pd.DataFrame([(k,k1,v1) for k,v in models.items() for k1,v1 in v.items()],\
                                        columns = ['Classifier','Index','Val'])
    df = df.pivot(index='Classifier', columns='Index', values='Val')\
           .reset_index()
    return df


def best_model(df, criteria):
    '''
    Find the best model based on certain criterion
    Inputs:
        df: (dataframe) performances of different models
        criterion: (list) of criteria to evaluate models
    Returns:
        (list) of dataframe with the rank based on criterion
    '''
    dfs = []
    for c in criteria:
        cols = ['Classifier'] + [c]
        temp_df = df.sort_values(c, ascending=False)[cols]
        dfs.append(temp_df)

    return dfs

