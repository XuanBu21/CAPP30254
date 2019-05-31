'''
CAPP30254 HW4
Xuan Bu
Pipeline for Unsupervised Learning
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn import preprocessing
from sklearn.cluster import KMeans
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


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


def generate_scatter_graph(df, cols, target):
    '''
    Given a dataframe and a set of columns, then generates the scatter
    graph of the possible combinations of columns.
    Inputs:
        df: dataframe
        cols: (list) of columns
        target: (str) target varable
    '''
    groups = df.groupby(target)
    for c in combinations(cols, 2):
        fig, ax = plt.subplots()
        for idx, group in groups:
            ax.scatter(df[c[0]], df[c[1]], label=idx)
        ax.legend()
        title = 'scatter plot: {} vs {}'.format(c[0], c[1])
        plt.title(title)
        plt.xlabel(c[0])
        plt.ylabel(c[1])
        plt.savefig(title)
        plt.show()
        plt.close()

### Source: https://github.com/dssg/MLforPublicPolicy/tree/master/labs/2019


######################################
# Step 3 Imputation/Pre-Process Data #
######################################

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


def normalize_features(df, contin_vars):
    '''
    Given a dataframe, then normalize continuous features by
    scaling each continuous feature.
    Inputs:
        df: a dataframe
        contin_vars: (list) list of continuous variables
    Returns:
        a dataframe with scaled feature
    '''
    scaler = preprocessing.MinMaxScaler()
    df[contin_vars] = scaler.fit_transform(df[contin_vars])


##########################################
# Step 4 Implement Unsupervised Learning #
##########################################

def run_kmeans(df, features, cluster):
    '''
    Implement k-means clustering algorithm.
    Inputs:
        df: a dataframe
        features: (list) of features
        cluster: (int) number of clusters to form
    Returns:
        a dataframe being clustered
    '''
    pd.options.mode.chained_assignment = None 
    data = df[features]
    kmeans = KMeans(n_clusters=cluster).fit(data)
    df['clusters'] = kmeans.labels_
    return df


########################
# Step 5 Summary Stats #
########################

def plot_decision_tree(df, features, target, cluster):
    '''
    Plot decision tree.
    Inputs:
        df: a dataframe
        features: (list) of features
        target: (str) target variable
        cluster: (int) number of clusters
    Returns:
        a decision tree graph
    '''
    dt = DecisionTreeClassifier()
    dt.fit(df[features], df[target])
    dot_data = tree.export_graphviz(dt, feature_names=features,\
                class_names=True, filled=True, rounded=True, out_file=None)
    file = 'DT with {} cluster'.format(cluster)
    graph = graphviz.Source(dot_data, filename=file, format='png')
    graph.view()
    return graph



###############################
# Step 6: Functions for Users #
###############################

def merge_clusters(df, new_cluster, clusters):
    '''
    Given a dataframe, then merge several clusters into one.
    Inputs:
        df: a dataframe
        clusters: (list) of clusters that are gonna be merged
        new_cluster: (str) label of merged cluster
    Returns:
        a dataframe with merged clusters
    '''
    df.loc[df['clusters'].isin(clusters), 'clusters'] = new_cluster
    return df


def split_cluster(df, features, cluster_label, cluster):
    '''
    Split a specific cluster into many (with a specific number of new clustering)
    Inputs:
        df: a dataframe
        features: (list) of features
        cluster_label: (str) specific cluster group
        cluster: (int) number of clusters to form
    Returns:
        a dataframe being reclustered
    '''
    data = df[df['clusters'] == cluster_label]
    return run_kmeans(data, features, cluster)


