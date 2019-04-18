'''
CAPP30254 HW2
Xuan Bu
Functions for the pipeline
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# Constants
TARGET = 'SeriousDlqin2yrs'
SEED = 21
COL_TO_DROP = 'zipcode'
CONTIN_VARS = ['RevolvingUtilizationOfUnsecuredLines', 'age',\
               'DebtRatio', 'NumberOfTime30-59DaysPastDueNotWorse',\
               'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',\
               'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',\
               'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
CLASSIFIERS = [LogisticRegression(), KNeighborsClassifier(),\
                                            DecisionTreeClassifier()]
CLASSIFIERS_NAMES = ['Logistic Regression', 'K-Nearest Neighbors',\
                                                        'Decision Tree']


#########################
# Step 1 Read/Load Data #
#########################

def read_data(filename):
    '''
    Load data from file.
    Input:
        filename: (str) name of the file loaded
    Returns:
        dataframe of the file
    '''
    return pd.read_csv(filename, index_col=0)\
             .drop([COL_TO_DROP], axis=1)


#######################
# Step 2 Explore Data #
#######################

def summary_continuous_vars(df):
    '''
    Given a dataframe and associated continuous variables,
    returns the summary statistics.
    Inputs:
        df: dataframe
    Returns:
        dataframe of the summary statistics
    '''
    return df[CONTIN_VARS].describe()


def summary_categorical_vars(df):
    '''
    Given a dataframe and associated categorical variables,
    returns the summary statistics.
    Inputs:
        df: dataframe
    Returns:
        dataframe of the summary statistics
    '''
    return df[TARGET].value_counts().reset_index()\
                       .rename(columns={'index': 'count'})


def generate_graph(df):
    '''
    Given a dataframe and a certain variable, returns the histograph
    of the variable
    Inputs:
        df: dataframe
    '''
    for var in CONTIN_VARS:
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
    Returns:
        dataframe of the graph of the correlations
    '''
    return df.corr().style.background_gradient(cmap='Blues').set_precision(4)


def count_outliers(df):
    '''
    Given a dataframe and a certain variable, returns the number of
    the outliers of the variable.
    Inputs:
        df: dataframe
    Returns:
        (int) the number of the outliers of the variable
    '''
    outliers = {}
    for var in CONTIN_VARS:
        summary = summary_continuous_vars(df)
        var_mean, var_std = summary[var]['mean'], summary[var]['std']
        cut_off = var_std * 3
        lower, upper = var_mean - cut_off, var_mean + cut_off
        outlier = df[(df[var] > upper) | (df[var] < lower)]
        outliers[var] = [outlier.shape[0]]

    return pd.DataFrame(outliers)


###########################
# Step 3 Pre-Process Data #
###########################

def fill_missing_with_median(df, var):
    '''
    Given a dataframe and a certain variable, replace the missing
    values of the variable with median.
    Inputs:
        df: dataframe
        var: (str) a certain variable
    Returns:
        dataframe with no missing values
    '''
    median = df[var].median()
    return df[var].fillna(median, inplace=True)


############################
# Step 4 Generate Features #
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


def create_binary_var(df, variables):
    '''
    Given a dataframe and a set of categorical variables, then
    create corresponding dummy variables.
    Inputs:
        df: dataframe
        variables: (list) a set of categorical variables
    Returns:
        dataframe with dummy variables
    '''
    return pd.get_dummies(df, columns=variables)


###########################
# Step 5 Build Classifier #
###########################

def split_data(df, predictors):
    '''
    Given a dataframe then split it into testing data and training data.
    Inputs:
        df: dataframe
        predictors: (dataframe) a set of independent variables
    Returns:
        dataframes of training data and testing data
    '''
    train_data, test_data, train_target, test_target = train_test_split(\
                df[predictors], df[TARGET], test_size=.1, random_state=SEED)
    return train_data, test_data, train_target, test_target


def build_classifier(classifier, train_data, train_target):
    '''
    Modeling the training data to build the classifier with different models.
    Inputs:
        classifier: (object) a kind of classifier
        train_data: (dataframe) of variables of interest
        train_target: (dataframe) of target variable
    Returns:
        (object) a classifier modeled with training data set
    '''
    return classifier.fit(train_data, train_target)


##############################
# Step 6 Evaluate Classifier #
##############################

def evaluate_classifier(df):
    '''
    Given a dataframe, build classifiers with training data set, then 
    evaluate classifiers with testing data set.
    Inputs:
        df: dataframe
    '''
    predictors = df.columns.difference([TARGET])
    train_data, test_data, train_target, test_target =\
                                            split_data(df, predictors)
    for idx, classifier in enumerate(CLASSIFIERS):
        model = build_classifier(classifier, train_data, train_target)
        pred_test_target = model.predict(test_data)
        accuracy = metrics.accuracy_score(test_target, pred_test_target)
        precision = metrics.precision_score(test_target, pred_test_target)
        print('The accuracy score of ' + \
                    CLASSIFIERS_NAMES[idx] + ' is {}'.format(accuracy))
        print('The precision score of ' + \
                    CLASSIFIERS_NAMES[idx] + ' is {}'.format(precision))

