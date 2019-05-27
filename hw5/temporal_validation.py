'''
CAPP30254 HW5
Xuan Bu
Temporal Validation
'''

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


def generate_date(train_start_time, train_end_time, rolling_window, gap):
    '''
    Find the splitting timestamp for training data and testing data.
    Inputs:
        train_start_time: (str) start time of training data
        train_end_time: (str) second time stamp to split data
        rolling_window: (int) the duration of testing data
        gap: (int) the gap between training data and testing data
    Returns:
        timestamp for both training data and testing data
    '''
    train_start_date = datetime.strptime(train_start_time, '%Y-%m-%d')
    train_end_date = datetime.strptime(train_end_time, '%Y-%m-%d')
    test_start_date = train_end_date + relativedelta(days=+gap)
    test_end_date = test_start_date + relativedelta(months=+rolling_window)

    return train_start_date, train_end_date, test_start_date, test_end_date


def splitting_data(df, predictors, target, time_var, start, end, rolling_window, gap):
    '''
    Given a dataframe then use temporal validation to split it into
    testing data and training data.
    Inputs:
        df: dataframe
        predictors: (dataframe) a set of independent variables
        target: (str) target columns
        time_var: (str) name of the column representing time
        start: (str) start time of training data
        end: (str) second time stamp to split data
        rolling_window: (int) the duration of testing data
        gap: (int) the gap between training data and testing data
    Returns:
        dataframes of training data and testing data
    '''
    train_start, train_end, test_start, test_end = \
                            generate_date(start, end, rolling_window, gap)
    x_set = df[predictors]
    y_set = df[target]
    train_data = x_set[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    test_data = x_set[(test_start <= df[time_var]) & (df[time_var] <= test_end)]
    train_target = y_set[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    test_target = y_set[(test_start <= df[time_var]) & (df[time_var] <= test_end)]
    
    return train_data, test_data, train_target, test_target


