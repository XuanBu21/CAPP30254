'''
CAPP30254 HW3
Xuan Bu
Run Pipeline
'''
import pipeline as pl
import classifiers as clf
from sklearn import metrics
import pandas as pd


# Step 1: Read Data
cols_to_drop = ['teacher_prefix', 'teacher_acctid', 'schoolid',\
        'school_ncesid', 'school_latitude', 'school_longitude',\
        'school_city', 'school_state', 'school_district', 'school_county']
df = pl.read_data('../data/projects_2012_2013.csv', cols_to_drop)


# Step 2: Explore Data
continuous_vars = ['total_price_including_optional_support', 'students_reached']
categorical_vars = ['school_metro', 'school_charter', 'school_magnet',\
                    'primary_focus_subject', 'primary_focus_area',\
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


# Step 3: Pre-Process Data
# Before replacement
df.isnull().sum()
cat_vars_with_nan = ['school_metro', 'primary_focus_subject',\
                'primary_focus_area', 'secondary_focus_subject',\
                'secondary_focus_area', 'resource_type', 'grade_level']
contin_vars_with_nan = ['students_reached']
# After replacement
for cat in cat_vars_with_nan:
    pl.fill_missing(df, cat, 'categorical', 'Unknonw')
for ct in contin_vars_with_nan:
    pl.fill_missing(df, ct, 'continuous')
df.isnull().sum()


# Step 4: Generate Features
# discretize 'total_price_including_optional_support' and 'students_reached'
for var in continuous_vars:
    labels_var = ['low', 'mediate', 'high']
    min_var = df[var].min()
    max_var = df[var].max()
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    bins_var = [min_var, q1, q3, max_var]
    df = pl.discretize_continuous_var(df, var, bins_var, labels_var)

# creat dummy variables for projects that got fully funded within 60 days.
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days'] = (df['datefullyfunded'] - df['date_posted']).dt.days
df['60daysfunded'] = [1 if x <= 60 else 0 for x in df['days']]


# create dummy variables
cats = df.columns.difference(['total_price_including_optional_support',\
                        'students_reached', 'date_posted', 'datefullyfunded',\
                        'days', '60daysfunded'])
df = pl.create_binary_var(df, cats)
df.head()


# Step 5: Experiment with Different Classifiers of Different Parameters
predictors = df.columns.difference(['total_price_including_optional_support',\
                        'students_reached', 'date_posted', 'datefullyfunded',\
                        'days', '60daysfunded'])
train_data, test_data, train_target, test_target = \
                pl.temporal_validation(df, predictors,\
                                '60daysfunded', 'date_posted', '2013-01-01')
# number of features before feature selection
train_data.shape[1]
# number of features after feature selection
new_train_data = pl.select_feature(train_data, train_target, 20)
new_train_data.shape[1]
# select testing data with the same dimension as training data
new_test_data = test_data[new_train_data.columns]

# KNN with different parameters
print('Experiment of KNN')
for neigh in clf.GRID['KNN']['n_neighbors']:
    print()
    print('number of neighbors is {}'.format(neigh))
    for metric in clf.GRID['KNN']['metric']:
        print('metric is {}'.format(metric))
        for weight in clf.GRID['KNN']['weights']:
            print('weight is {}'.format(weight))
            knn = clf.build_knn(neigh, metric, weight)
            clf.trail_threshold(knn, new_train_data,\
                                train_target, new_test_data, test_target)
# Decision Tree with different parameters
print('Experiment of DT')
for crt in clf.GRID['DT']['criterion']:
    print()
    print('criterion is {}'.format(crt))
    for depth in clf.GRID['DT']['max_depth']:
        print('max depth is {}'.format(depth))
        for leaf in clf.GRID['DT']['min_samples_leaf']:
            print('min samples leaf is {}'.format(leaf))
            dt = clf.build_dt(crt, depth, leaf)
            clf.trail_threshold(dt, new_train_data,\
                                train_target, new_test_data, test_target)
# Logistic Regression with different parameters
print('Experiment of LR')
for p in clf.GRID['LR']['penalty']:
    print()
    print('penalty is {}'.format(p))
    for c in clf.GRID['LR']['C']:
        print('C is {}'.format(c))
        lr = clf.build_lr(p, c)
        clf.trail_threshold(lr, new_train_data,\
                                train_target, new_test_data, test_target)
# SVM with different parameters
print('Experiment of SVM')
for c in clf.GRID['SVM']['C']:
    print('C is {}'.format(c))
    svm = clf.build_svm(c)
    svm.fit(new_train_data, train_target)
    confidence_score = svm.decision_function(new_test_data)
    for threshold in clf.THRESHOLD:
        pred_label = [1 if x > threshold else 0 for x in confidence_score]
        print('(Threshold: {}), the total number of the predicted '
            'is {}, the accuracy is {:.2f}'.format(threshold,\
            sum(pred_label), metrics.f1_score(test_target, pred_label)))
# Random Forest with different parameters
print('Experiment of RF')
for n in clf.GRID['RF']['n_estimators']:
    print()
    print('number of estimators is {}'.format(n))
    for depth in clf.GRID['RF']['max_depth']:
        print('maximum depth is {}'.format(depth))
        rf = clf.build_rf(n, depth)
        clf.trail_threshold(rf, new_train_data,\
                                train_target, new_test_data, test_target)
# Boosting with different parameters
print('Experiment of AB')
for n in clf.GRID['AB']['n_estimators']:
    print()
    print('number of estimators is {}'.format(n))
    ab = clf.build_ab(n)
    clf.trail_threshold(ab, new_train_data,\
                                train_target, new_test_data, test_target)
# Bagging with different parameters
print('Experiment of BAG')
for n in clf.GRID['BAG']['n_estimators']:
    print()
    print('number of estimators is {}'.format(n))
    for m in clf.GRID['BAG']['max_samples']:
        print('number of maximum samples is {}'.format(m))
        bag = clf.build_bag(n, m)
        clf.trail_threshold(bag, new_train_data,\
                                train_target, new_test_data, test_target)


# Step 6: Select the Best Parameter Grid for Different Classifiers
best_paras = {}
new_train_data = pl.select_feature(train_data, train_target, 20)
for name, model in clf.CLASSIFIERS.items():
    best_paras[name] = pl.select_best_grid(model,\
                            clf.GRID[name], new_train_data, train_target)
print(best_paras)


# Step 7: Evaluate Models with the Best Parameters by Tempora l Validation
# First dataset is separated by the timestamp 2012-07-01.
train1, test1, train_target1, test_target1 = \
        pl.temporal_validation(df, predictors,\
                            '60daysfunded', 'date_posted', '2012-07-01')
# Second dataset is separated by the timestamp 2013-01-01.
train2, test2, train_target2, test_target2 = \
        pl.temporal_validation(df, predictors,\
                            '60daysfunded', 'date_posted', '2013-01-01')
# Third dataset is separated by the timestamp 2013-07-01.
train3, test3, train_target3, test_target3 = \
        pl.temporal_validation(df, predictors,\
                            '60daysfunded', 'date_posted', '2013-07-01')

# Find best parameter grid for dataset 1
best_paras_1 = {}
new_train1 = pl.select_feature(train1, train_target1, 20)
for name, model in clf.CLASSIFIERS.items():
    best_paras_1[name] = pl.select_best_grid(model,\
                                clf.GRID[name], new_train1, train_target1)
print(best_paras_1)
# Find best parameter grid for dataset 2
best_paras_2 = {}
new_train2 = pl.select_feature(train2, train_target2, 20)
for name, model in clf.CLASSIFIERS.items():
    best_paras_2[name] = pl.select_best_grid(model,
                                clf.GRID[name], new_train2, train_target2)
print(best_paras_2)
# Find best parameter grid for dataset 3
best_paras_3 = {}
new_train3 = pl.select_feature(train3, train_target3, 20)
for name, model in clf.CLASSIFIERS.items():
    best_paras_3[name] = pl.select_best_grid(model,\
                                clf.GRID[name], new_train3, train_target3)
print(best_paras_3)

# select testing data with the same dimension as training data
new_test1 = test1[new_train1.columns]
new_test2 = test2[new_train2.columns]
new_test3 = test3[new_train3.columns]

# Performances of different models using dataset 1
df1 = pl.evaluate_classifier(best_paras_1,\
                        new_train1, new_test1, train_target1, test_target1)
# Performances of different models using dataset 2
df2 = pl.evaluate_classifier(best_paras_2,\
                        new_train2, new_test2, train_target2, test_target2)
# Performances of different models using dataset 3
df3 = pl.evaluate_classifier(best_paras_3,\
                        new_train3, new_test3, train_target3, test_target3)


# Step 8: Find the Best Model for Different Dataset
print(df1)
print(df2)
print(df3)
pl.best_model(df2,\
        'p_at_threshold_0.5').loc[:, ['Classifier', 'p_at_threshold_0.5']]
pl.best_model(df2,\
        'r_at_threshold_0.5').loc[:, ['Classifier', 'r_at_threshold_0.5']]
pl.best_model(df2,\
        'roc_at_threshold_0.5').loc[:, ['Classifier', 'roc_at_threshold_0.5']]
pl.best_model(df2, 'train_time').loc[:, ['Classifier', 'train_time']]

pl.best_model(df1,\
        'p_at_threshold_0.05').loc[:, ['Classifier', 'p_at_threshold_0.05']]
pl.best_model(df1,\
        'r_at_threshold_0.05').loc[:, ['Classifier', 'r_at_threshold_0.05']]
pl.best_model(df1,\
        'roc_at_threshold_0.05').loc[:, ['Classifier', 'roc_at_threshold_0.05']]
pl.best_model(df2,\
        'p_at_threshold_0.05').loc[:, ['Classifier', 'p_at_threshold_0.05']]
pl.best_model(df2,\
        'r_at_threshold_0.05').loc[:, ['Classifier', 'r_at_threshold_0.05']]
pl.best_model(df2,\
        'roc_at_threshold_0.05').loc[:, ['Classifier', 'roc_at_threshold_0.05']]
pl.best_model(df3,\
        'p_at_threshold_0.05').loc[:, ['Classifier', 'p_at_threshold_0.05']]
pl.best_model(df3,\
        'r_at_threshold_0.05').loc[:, ['Classifier', 'r_at_threshold_0.05']]
pl.best_model(df3,\
        'roc_at_threshold_0.05').loc[:, ['Classifier', 'roc_at_threshold_0.05']]
pl.best_model(df3, 'train_time').loc[:, ['Classifier', 'train_time']]

