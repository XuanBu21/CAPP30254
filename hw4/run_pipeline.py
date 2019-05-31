'''
CAPP30254 HW4
Xuan Bu
Run Pipeline for Unsupervised Learning
'''

import pipeline as pl
import pandas as pd

### Step 1: Load Data
df = pl.read_data('projects_2012_2013.csv')



### Step 2: Explore Data
continuous_vars = ['total_price_including_optional_support', 'students_reached']
categorical_vars = ['teacher_prefix', 'school_metro', 'school_charter',\
                    'school_magnet', 'primary_focus_subject', 'primary_focus_area',\
                    'secondary_focus_subject', 'secondary_focus_area',\
                    'resource_type', 'poverty_level', 'grade_level',\
                    'eligible_double_your_impact_match']

# create target variable: label 1 for projects that didn't get fully funded within 60 days, otherwise 0
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
df['days'] = (df['datefullyfunded'] - df['date_posted']).dt.days
df['NotFunded60days'] = [1 if x > 60 else 0 for x in df['days']]
# generate scatter plot
pl.generate_scatter_graph(df, continuous_vars, 'NotFunded60days')
# do descriptive statistics of continuous variables
pl.summary_continuous_vars(df, continuous_vars)
# do descriptive statistics of categorical variables
for cat in categorical_vars:
    print(pl.summary_categorical_vars(df, cat))
# generate histograph of continuous variables
pl.generate_graph(df, continuous_vars)
# generate heatmap of correlations between variables
pl.generate_corr_graph(df)
# find outliers
pl.count_outliers(df, continuous_vars)



### Step 3: Imputation/Pre-Process
for cat in categorical_vars:
    pl.fill_missing(df, cat, 'categorical', 'Unknonw')
for ct in continuous_vars:
    pl.fill_missing(df, ct, 'continuous')
pl.normalize_features(df, continuous_vars)



### Step 4: Implement Kmeans
df = df.reset_index()
clusters = [2, 4, 6]
clustered_dfs = []
for c in clusters:
    print('Number of clusters is {}'.format(c))
    ndf = pl.run_kmeans(df, continuous_vars, c)
    pl.generate_scatter_graph(ndf, continuous_vars, 'clusters')
    clustered_dfs.append(ndf)



### Step 5: Summary Stats

df_cluster_2 = clustered_dfs[0]
df_cluster_3 = clustered_dfs[1]
df_cluster_4 = clustered_dfs[2]
# when cluster is 2
pl.plot_decision_tree(df_cluster_2, continuous_vars, 'clusters', 2)
# when cluster is 3
pl.plot_decision_tree(df_cluster_3, continuous_vars, 'clusters', 3)
# when cluster is 4
pl.plot_decision_tree(df_cluster_4, continuous_vars, 'clusters', 4)



### Step 6: Functions for Users
# merge several clusters into one
# use df_cluster_4 and merge cluster 2 and 3 into one cluster labeled as 5
# before merging
df_cluster_4['clusters'].unique()
# after merging
new_df_cluster_4 = pl.merge_clusters(df_cluster_4, 5, [2, 3])
new_df_cluster_4['clusters'].unique()

# recluster with a new k
# use df_cluster_2 and k = 5
k = 5
# before reclustering
df_cluster_2['clusters'].unique()
# after reclustering
new_df_cluster_2 = pl.run_kmeans(df_cluster_2, continuous_vars, 5)
new_df_cluster_2['clusters'].unique()

# split a specific cluster into many (with a specific number of new clustering)
# use df_cluster_3 and split cluster labeled 0 into many
# before splitting
df_cluster_3['clusters'].unique()
new_df_cluster_3 = pl.split_cluster(df_cluster_3, continuous_vars, 0, 5)
# after splitting
new_df_cluster_3['clusters'].unique()

