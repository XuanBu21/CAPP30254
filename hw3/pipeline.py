'''
CAPP30254 HW3
Xuan Bu
Improved Pipeline
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,\
                             RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest, chi2


# Constants
SEED = 21
CLASSIFIERS = {'LR': LogisticRegression(), 'KNN': KNeighborsClassifier(),\
               'DT': DecisionTreeClassifier(), 'SVM': LinearSVC(),\
               'RF': RandomForestClassifier(), 'AB':AdaBoostClassifier(),\
               'BAG': BaggingClassifier()}
THRESHOLD = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


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


###########################
# Step 3 Pre-Process Data #
###########################

def fill_missing(df, var, var_type, filling_value=None):
    '''
    Given a dataframe and a certain variable, replace the missing
    values of the variable with median.
    Inputs:
        df: dataframe
        var: (str) name of variable
        var_type: (str) 'continuous' or 'categorical'
        filling_value: (str) value to fill missing value
    Returns:
        dataframe with no missing values
    '''
    if var_type == 'continuous':
        median = df[var].median()
        return df[var].fillna(median, inplace=True)
    if var_type == 'categorical':
        return df[var].fillna(filling_value, inplace=True)


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

def select_feature(train_data, train_target, k):
    '''
    Select the best k features
    Inputs:
        train_data: (dataframe) training data of variables of interest
        train_target: (dataframe) training data of target variable
    Returns:
        (dataframe) train data with k best features
    '''
    selector = SelectKBest(chi2, k=k)
    selector.fit(train_data, train_target)

    return train_data[train_data.columns[selector.get_support(indices=True)]]


def split_data(df, predictors, target):
    '''
    Given a dataframe then split it into testing data and training data.
    Inputs:
        df: dataframe
        predictors: (dataframe) a set of independent variables
        target: (str) target columns
    Returns:
        dataframes of training data and testing data
    '''
    train_data, test_data, train_target, test_target = train_test_split(\
                df[predictors], df[target], test_size=.2, random_state=SEED)
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


def select_best_grid(model, grid, train_data, train_target):
    '''
    Conduct experiments with different models under different threshold.
    Inputs:
        model: model
        grid: (dict) dictionary with parameters names (string) as
          keys and lists of parameter settings to try as values
        train_data: (dataframe) training data of variables of interest
        train_target: (dataframe) training data of target variable
    Returns:
        (dict) parameter grid that gives the best results on the hold out data
    '''
    grid_search = GridSearchCV(model, grid, cv=5)
    grid_search.fit(train_data, train_target)
    return grid_search.best_params_


def temporal_validation(df, predictors, target, time_var, time_stamp):
    '''
    Given a dataframe then use temporal validation to split it into
    testing data and training data.
    Inputs:
        df: dataframe
        predictors: (dataframe) a set of independent variables
        target: (str) target columns
        time_var: (str) name of the column representing time
        time_stamp: (object) time stamp to split data
    Returns:
        dataframes of training data and testing data
    '''
    x_set = df[predictors]
    y_set = df[target]
    train_data = x_set[df[time_var] <= time_stamp]
    test_data = x_set[df[time_var] > time_stamp]
    train_target = y_set[df[time_var] <= time_stamp]
    test_target = y_set[df[time_var] > time_stamp]
    return train_data, test_data, train_target, test_target


##############################
# Step 6 Evaluate Classifier #
##############################

def evaluate_classifier(best_grid, train_data, test_data, train_target, test_target):
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
    '''
    models = collections.defaultdict(dict)
    for name, clf in CLASSIFIERS.items():
        para = best_grid[name]
        models[name]['best_paras'] = para
        train_start = time.time()
        model = clf.set_params(**para)
        model.fit(train_data, train_target)
        
        models[name]['train_time'] = time.time() - train_start
        if name == 'SVM':
            pred_scores = model.decision_function(test_data)
        else:
            pred_scores = model.predict_proba(test_data)

    # plot ROC, precision, and recall
        generate_precision_recall_curve(pred_scores, test_target, name)
        generate_ROC_graph(pred_scores, test_target, name)

    # evaluate performances of classifiers under different thresholds
        for threshold in THRESHOLD:
            if name == 'SVM':
                pred_label = [1 if x > threshold else 0 for x in pred_scores]
            else:
                pred_label = [1 if x[1] > threshold else 0 for x in pred_scores]
            p_at_threshold = 'p_at_threshold_' + str(threshold)
            r_at_threshold = 'r_at_threshold_' + str(threshold)
            roc_at_threshold = 'roc_at_threshold_' + str(threshold)
            models[name][p_at_threshold] = precision_score(test_target, pred_label)
            models[name][r_at_threshold] = recall_score(test_target, pred_label)
            models[name][roc_at_threshold] = roc_auc_score(test_target, pred_label)
    
    # convert dictionary to dataframe
    df = pd.DataFrame([(k,k1,v1) for k,v in models.items() for k1,v1 in v.items()],\
                                        columns = ['Classifier','Index','Val'])
    df = df.pivot(index='Classifier', columns='Index', values='Val')\
           .reset_index()
    return df


def generate_precision_recall_curve(pred_scores, test_target, model):
    '''
    Plot the graph of the tradeoff between precision and recall under
    different thresholds.
    Inputs:
        pred_scores: (array) the 
        test_target: (dataframe) testing data of target variable
        model: (str) name of the classifier
    '''
    if model == 'SVM':
        precision, recall, thresholds = \
                    precision_recall_curve(test_target, pred_scores)
    else:
        precision, recall, thresholds = \
                    precision_recall_curve(test_target, pred_scores[:,1])
    plt.plot(recall, precision, marker='.')
    plt.title('The Tradeoff between Precision and Recall of '+str(model))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('The Tradeoff between Precision and Recall of '+str(model))
    plt.show()
    plt.close()


def generate_ROC_graph(pred_scores, test_target, model):
    '''
    Plot the graph of Receiver Operating Characteristic.
    Inputs:
        pred_scores: (array) the 
        test_target: (dataframe) testing data of target variable
        model: (str) name of the classifier
    '''
    if model == 'SVM':
        fpr, tpr, threshold = roc_curve(test_target, pred_scores)
    else:
        fpr, tpr, threshold = roc_curve(test_target, pred_scores[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of '+str(model))
    plt.legend(loc = 'lower right')
    plt.savefig('ROC Graph of '+str(model))
    plt.show()
    plt.close()


def best_model(df, criterion):
    '''
    Find the best model based on certain criterion
    Inputs:
        df: (dataframe) performances of different models
        criterion: (str) criterion to evaluate models
    Returns:
        (dataframe) of the rank based on criterion
    '''
    return df.sort_values(criterion, ascending=False)

