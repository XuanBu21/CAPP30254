'''
CAPP30254 HW2
Xuan Bu
Running the pipeline
'''

import hw2

# Step 1 Read/Load Data
df = hw2.read_data('~/Downloads/hw2/data/credit-data.csv')
df.head()


# Step 2 Explore Data
hw2.summary_continuous_vars(df)
hw2.summary_categorical_vars(df)
hw2.generate_graph(df)
hw2.generate_corr_graph(df)
outliers = hw2.count_outliers(df)
print(outliers)
outliers.sum().mean()


# Step 3 Pre-Process Data
hw2.fill_missing_with_median(df, 'MonthlyIncome')
hw2.fill_missing_with_median(df, 'NumberOfDependents')
df.isnull().sum()


# Step 4: Generate Features
# discretize 'MonthlyIncome'
labels_income = ['low', 'mediate', 'high']
min_income = df['MonthlyIncome'].min()
max_income = df['MonthlyIncome'].max()
q1 = df['MonthlyIncome'].quantile(0.25)
q3 = df['MonthlyIncome'].quantile(0.75)
bins_income = [min_income, q1, q3, max_income]
df = hw2.discretize_continuous_var(df, 'MonthlyIncome',\
                            bins_income, labels_income)
# discretize 'age'
labels_age = ['20-35', '35-50', '50-65', '65-80','80-95','95-110']
bins_age = range(20,111,15)
df = hw2.discretize_continuous_var(df, 'age', bins_age, labels_age)
df = hw2.create_binary_var(df,\
            ['discretized_age', 'discretized_MonthlyIncome'])
df.head()


# Step 5 Build Classifier
predictors = df.columns.difference([hw2.TARGET])
train_data, test_data, train_target, test_target =\
                                hw2.split_data(df, predictors)
for classifier in hw2.CLASSIFIERS:
    hw2.build_classifier(classifier, train_data, train_target)


# Step 6 Evaluate Classifier
hw2.evaluate_classifier(df)

