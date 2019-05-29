'''
CAPP30254 HW5
Xuan Bu
Run Pipeline
'''

import pipeline as pl
import classifiers as clf
import evaluation as el
import temporal_validation as tv
import pandas as pd


### Step 1: Read Data
df = pl.read_data('projects_2012_2013.csv')


### Step 2: Explore Data
continuous_vars = ['total_price_including_optional_support', 'students_reached']
categorical_vars = ['teacher_prefix', 'school_metro', 'school_charter',\
                    'school_magnet', 'primary_focus_subject', 'primary_focus_area',\
                    'secondary_focus_subject', 'secondary_focus_area',\
                    'resource_type', 'poverty_level', 'grade_level',\
                    'eligible_double_your_impact_match']

pl.summary_continuous_vars(df, continuous_vars)
for cat in categorical_vars:
    print(pl.summary_categorical_vars(df, cat))
pl.generate_graph(df, continuous_vars)
pl.generate_corr_graph(df)
outliers = pl.count_outliers(df, continuous_vars)
outliers.sum().mean()


### Step 3: Generate Features
# label 1 for projects that didn't get fully funded within 60 days, otherwise 0
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days'] = (df['datefullyfunded'] - df['date_posted']).dt.days
df['NotFunded60days'] = [1 if x > 60 else 0 for x in df['days']]

features = ['total_price_including_optional_support', 'students_reached',\
            'teacher_prefix', 'school_metro', 'school_charter', 'school_magnet',\
            'primary_focus_subject', 'primary_focus_area', 'secondary_focus_subject',\
            'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level',\
            'eligible_double_your_impact_match']


### Step 4: Splitting Data
# First dataset
train1, test1, train_target1, test_target1 = \
    tv.splitting_data(df, features, 'NotFunded60days', 'date_posted', '2012-01-01', '2012-06-30', 6, 60)
# Second dataset
train2, test2, train_target2, test_target2 = \
    tv.splitting_data(df, features, 'NotFunded60days', 'date_posted', '2012-01-01', '2012-12-31', 6, 60)
# Third dataset
train3, test3, train_target3, test_target3 = \
    tv.splitting_data(df, features, 'NotFunded60days', 'date_posted', '2012-01-01', '2013-06-30', 6, 60)


### Step 5: Imputation/Pre-Process
datasets = [(train1, test1), (train2, test2), (train3, test3)]
# imputation with median of training set
for df in datasets:
    for cat in categorical_vars:
        pl.fill_missing(df[0], df[1], cat, 'categorical', 'Unknonw')
    for ct in continuous_vars:
        pl.fill_missing(df[0], df[1], ct, 'continuous')
# Normalize continuous variables by scaling
for ds in datasets:
    for d in ds:
        pl.normalize_features(d, continuous_vars)
# discretize 'total_price_including_optional_support' and 'students_reached'
for dataset in datasets:
    for df in dataset:
        for var in continuous_vars:
            labels_var = ['low', 'mediate', 'high']
            min_var = df[var].min()
            max_var = df[var].max()
            q1 = df[var].quantile(0.25)
            q3 = df[var].quantile(0.75)
            bins_var = [min_var, q1, q3, max_var]
            df = pl.discretize_continuous_var(df, var, bins_var, labels_var)
# create dummy variables for all datasets
train1, test1 = pl.create_dummies(train1, test1, list(train1.columns.difference(continuous_vars)))
train2, test2 = pl.create_dummies(train2, test2, list(train2.columns.difference(continuous_vars)))
train3, test3 = pl.create_dummies(train3, test3, list(train3.columns.difference(continuous_vars)))


### Step 6: Experiment with Different Classifiers of Different Parameters
ks = [10, 50]
pl.build_classifier(train1, test1, train_target1, test_target1, ks)


### Step 7: Evaluate Models with the Best Parameters by Temporal Validation
best_paras = {'LR': {'C': 10, 'penalty': 'l1'},\
            'DT': {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 2},\
            'GB': {'n_estimators': 10, 'learning_rate' : 0.5, 'max_depth': 50},\
            'RF': {'max_depth': 5, 'n_estimators': 5},\
            'AB': {'n_estimators': 1},\
            'BAG': {'max_samples': 0.5, 'n_estimators': 20},\
            'ET': {'n_estimators': 5, 'max_depth': 5, 'criterion': 'gini'}}
ks = [1, 2, 5, 10, 20, 30, 50]
# Performances of different models using dataset 1
df1 = pl.evaluate_classifier(best_paras, train1, test1, train_target1, test_target1, ks)
# Performances of different models using dataset 2
df2 = pl.evaluate_classifier(best_paras, train2, test2, train_target2, test_target2, ks)
# Performances of different models using dataset 3
df3 = pl.evaluate_classifier(best_paras, train3, test3, train_target3, test_target3, ks)


### Step 8: Find the Best Model for Different Dataset
# Set k = 20, that is, to identify 20% of posted projects that are
# at highest risk of not getting fully funded to intervene with.
# And comparing three datasets historically.
criteria_20 = ['a_at_20', 'p_at_20', 'r_at_20', 'roc_at_20', 'train_time']
df1_at_20 = pl.best_model(df1, criteria_20)
for df in df1_at_20:
    print(df)
df2_at_20 = pl.best_model(df2, criteria_20)
for df in df2_at_20:
    print(df)
df3_at_20 = pl.best_model(df3, criteria_20)
for df in df3_at_20:
    print(df)
# To identify 5% of posted projects that are at highest risk of not getting
# fully funded to intervene with, I set k = 5.
# Using dataset 2 as the selected dataset.
criteria_5 = ['a_at_5', 'p_at_5', 'r_at_5', 'roc_at_5', 'train_time']
df2_at_5 = pl.best_model(df2, criteria_5)
for df in df2_at_5:
    print(df)

