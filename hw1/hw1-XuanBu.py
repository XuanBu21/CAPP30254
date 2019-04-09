'''
CAPP30254 HW1
Xuan Bu
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from functools import reduce
import re


# import data
cri_17 = pd.read_csv('Crimes_-_2017.csv')
cri_18 = pd.read_csv('Crimes_-_2018.csv')
convert_community_tract = \
    pd.read_csv('2010 Tract to Community Area Equivalency File - Sheet1.csv')
convert_community_tract = convert_community_tract\
            .rename(columns={'CHGOCA': 'Community Area', 'TRACT': 'tract'})



#############
# Problem 1 #
#############
# count total number of crimes
cri_17['amount'] = cri_17['Primary Type'].count()
cri_17['year'] = 2017
cri_18['amount'] = cri_18['Primary Type'].count()
cri_18['year'] = 2018
cri_17_18 = pd.concat([cri_17, cri_18])
sns.pointplot(x="year", y="amount", data=cri_17_18)\
   .set_title('Change of the Amount of Crimes from 2017 to 2018')
plt.tight_layout()
plt.show()


# count crimes of each type
c17_count = cri_17.groupby('Primary Type').size()\
                  .reset_index().rename(columns={0 : 'count'})
c17_count['year'] = 2017
sns.barplot(x='count', y='Primary Type', data=c17_count)\
   .set_title('Amount of Crimes of Each Type in 2017')
plt.tight_layout()
plt.show()

c18_count = cri_18.groupby('Primary Type').size()\
                  .reset_index().rename(columns={0 : 'count'})
c18_count['year'] = 2018
sns.barplot(x='count', y='Primary Type', data=c18_count)\
   .set_title('Amount of Crimes of Each Type In 2018')
plt.tight_layout()
plt.show()


# compare the change of the crimes of each type from 2017 to 2018
crime_17_18 = pd.concat([c17_count, c18_count])
sns.pointplot(x="year", y="count", hue="Primary Type", data=crime_17_18)\
   .set_title('Change of the Number of Crimes of Each Type from 2017 to 2018')
plt.legend(loc='upper right', fontsize='6', fancybox=True)
plt.show()

cat_plot = sns.catplot(x='year', y='count', col='Primary Type', col_wrap=6,
                    data=crime_17_18, saturation=.5,
                    kind="point", ci=None, aspect=.6)
(cat_plot.set_axis_labels("", "count")
      .set_xticklabels(["2017", "2018"])
      .set_titles("{col_name}"))
plt.show()


# compare crimes of each type among different neighborhood
max_cri = cri_17.groupby(['Community Area', 'Primary Type'])['Primary Type']\
                .agg(['count']).sort_values(by='count', ascending=False)\
                .reset_index().drop_duplicates('Community Area', keep='first')
cri_type_neigh_17 = max_cri.groupby(['Primary Type']).size()\
                           .reset_index().rename(columns={0 : 'count'})
cri_type_neigh_17['year'] = 2017
sns.barplot(x='count', y='Primary Type', data=cri_type_neigh_17)\
   .set_title('Most Frequent Crime among Neighboorhood in 2017')
plt.tight_layout()
plt.show()

max_cri = cri_18.groupby(['Community Area', 'Primary Type'])['Primary Type']\
                .agg(['count']).sort_values(by='count', ascending=False)\
                .reset_index().drop_duplicates('Community Area', keep='first')
cri_type_neigh_18 = max_cri.groupby(['Primary Type']).size()\
                           .reset_index().rename(columns={0 : 'count'})
cri_type_neigh_18['year'] = 2018
sns.barplot(x='count', y='Primary Type', data=cri_type_neigh_18)\
   .set_title('Most Frequent Crime among Neighboorhood in 2018')
plt.tight_layout()
plt.show()

cri_17_18_neigh = pd.concat([cri_type_neigh_17, cri_type_neigh_18])
sns.pointplot(x="year", y="count", hue="Primary Type", data=cri_17_18_neigh)\
   .set_title('Change of the Type of the Most Frequent Crime'
                        'among Neighboorhood from 2017 to 2018')
plt.legend(loc='upper right', fontsize='6', fancybox=True)
plt.show()

# top 10 neighborhood of the most frequent crimes
top10_neigh_17 = cri_17.groupby(['Community Area']).size()\
                    .nlargest(10).reset_index().rename(columns={0 : 'count'})
top10_neigh_17['year'] = 2017
top10_neigh_18 = cri_17.groupby(['Community Area']).size()\
                    .nlargest(10).reset_index().rename(columns={0 : 'count'})
top10_neigh_18['year'] = 2018
top10_17_18 = pd.concat([top10_neigh_17, top10_neigh_18])
cat_plot = sns.catplot(x='year', y='count', col='Community Area', col_wrap=4,
                    data=top10_17_18, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
plt.show()



#############
# Problem 2 #
#############

# pull race data of 2017
race_raw_17 = requests.get('https://api.census.gov/data/2017/acs/acs5?'
    'get=B02001_002E,B02001_003E,B02001_005E,B02001_006E,B02001_007E,'
    'NAME&for=tract:*&in=state:17&in=county:031')
race_data_17 = race_raw_17.json()
race_df_17 = pd.DataFrame(race_data_17)
header_17 = race_df_17.iloc[0]
race_df_17 = race_df_17[1:]
race_df_17.columns = header_17
race_17 = race_df_17.rename(columns = {'B02001_002E': 'white',
                    'B02001_003E': 'black',
                    'B02001_005E': 'asian',
                    'B02001_006E': 'nativehawaiian',
                    'B02001_007E': 'other'})
race_17 = race_17.astype({'white': int, 'black': int, 'asian': int,
                    'nativehawaiian': int, 'other': int, 'tract': int})
race_17 = race_17[['tract', 'white', 'black', 'asian',
                    'nativehawaiian', 'other']]
race_17['total'] = race_17.iloc[:, 1:].sum(axis=1)
race_17.iloc[:, 1:] = race_17.iloc[:, 1:].div(race_17['total'], axis=0)


# pull education data of 2017
edu_raw_17 = requests.get('https://api.census.gov/data/2017/acs/acs5?'
    'get=B15003_001E,B15003_002E,NAME&for=tract:*&in=state:17&in=county:031')
edu_data_17 = edu_raw_17.json()
edu_df_17 = pd.DataFrame(edu_data_17)
header_17 = edu_df_17.iloc[0]
edu_df_17 = edu_df_17[1:]
edu_df_17.columns = header_17
edu_17 = edu_df_17.rename(columns = {'B15003_001E': 'total edu',
                                'B15003_002E': 'no schooling completed'})
edu_17 = edu_17.astype({'no schooling completed': int,
                        'total edu': int, 'tract': int})
edu_17 = edu_17[['tract', 'no schooling completed', 'total edu']]
edu_17['edu rate'] = edu_17.apply(lambda row: \
    (row['total edu']-row['no schooling completed'])/row['total edu'], axis=1)


# pull income data of 2017
income_raw_17 = requests.get('https://api.census.gov/data/2017/acs/acs5?'
        'get=B19001_001E,B19001_002E,B19001_003E,B19001_004E,B19001_005E,'
        'B19001_006E,B19001_007E,B19001_008E,B19001_009E,B19001_010E,'
        'B19001_011E,B19001_012E,B19001_013E,B19001_014E,B19001_015E,'
        'B19001_016E,B19001_017E,NAME&for=tract:*&in=state:17&in=county:031')
income_data_17 = income_raw_17.json()
income_df_17 = pd.DataFrame(income_data_17)
header_17 = income_df_17.iloc[0]
income_df_17 = income_df_17[1:]
income_df_17.columns = header_17
income_17 = income_df_17.rename(\
    columns = {'B19001_002E': '< 10000', 'B19001_003E': '10000 to 14999',
        'B19001_004E': '15000 to 19999', 'B19001_005E': '15000 to 19999',
        'B19001_006E': '25000 to 29999', 'B19001_007E': '30000 to 34999',
        'B19001_008E': '35000 to 39999', 'B19001_009E': '40000 to 44999',
        'B19001_010E': '45000 to 49999', 'B19001_011E': '50000 to 59999', 
        'B19001_012E': '60000 to 74999', 'B19001_013E': '75000 to 99999',
        'B19001_014E': '100000 to 124999', 'B19001_015E': '125000 to 149999',
        'B19001_016E': '150000 to 199999', 'B19001_017E': '200000 or more',
        'B19001_001E': 'total income'})
income_17 = income_17.astype({'< 10000': int, '10000 to 14999': int,
    '15000 to 19999': int, '15000 to 19999': int,'25000 to 29999': int,
    '30000 to 34999': int, '35000 to 39999': int, '40000 to 44999': int,
    '45000 to 49999': int, '50000 to 59999': int, '60000 to 74999': int,
    '75000 to 99999': int, '100000 to 124999': int, '125000 to 149999': int,
    '150000 to 199999': int, '200000 or more': int,
    'total income': int, 'tract': int})
income_17['< 50000'] = income_17.iloc[:, 1:10].sum(axis=1)
income_17['50000 to 100000'] = income_17.iloc[:, 10:13].sum(axis=1)
income_17['> 100000'] = income_17.iloc[:, 13:17].sum(axis=1)
income_17 = income_17[['tract', 'total income', '< 50000',\
                                     '50000 to 100000', '> 100000']]
income_17.iloc[:, 2:] = income_17.iloc[:, 2:]\
                            .div(income_17['total income'], axis=0)

# pull race data of 2018
race_raw_18 = requests.get('https://api.census.gov/data/2018/pdb/tract?get='
                        'NH_White_alone_ACS_12_16,NH_Blk_alone_ACS_12_16,NH'
                        '_AIAN_alone_ACS_12_16,NH_SOR_alone_ACS_12_16&for='
                        'tract:*&in=state:17%20county:031')
race_data_18 = race_raw_18.json()
race_df_18 = pd.DataFrame(race_data_18)
header_18 = race_df_18.iloc[0]
race_df_18 = race_df_18[1:]
race_df_18.columns = header_18
race_18 = race_df_18.rename(columns = {'NH_White_alone_ACS_12_16': 'white',
                    'NH_Blk_alone_ACS_12_16': 'black',
                    'NH_AIAN_alone_ACS_12_16': 'native',
                    'NH_SOR_alone_ACS_12_16': 'other'})
race_18 = race_18.astype({'white': int, 'black': int, 'native': int,
                            'other': int, 'tract': int})
race_18 = race_18[['tract', 'white', 'black', 'native', 'other']]
race_18['total'] = race_18.iloc[:, 1:].sum(axis=1)
race_18.iloc[:, 1:] = race_18.iloc[:, 1:].div(race_18['total'], axis=0)

# pull education data of 2018
edu_raw_18 = requests.get('https://api.census.gov/data/2018/pdb/tract?get'
                        '=Not_HS_Grad_ACS_12_16,Pop_25yrs_Over_ACS_12_16&'
                        'for=tract:*&in=state:17%20county:031')
edu_data_18 = edu_raw_18.json()
edu_df_18 = pd.DataFrame(edu_data_18)
header_18 = edu_df_18.iloc[0]
edu_df_18 = edu_df_18[1:]
edu_df_18.columns = header_18
edu_18 = edu_df_18.rename(columns = {'Not_HS_Grad_ACS_12_16': 'not HS',
                                    'Pop_25yrs_Over_ACS_12_16': 'total'})
edu_18 = edu_18.astype({'not HS': int, 'total': int, 'tract': int})
edu_18 = edu_18[['tract', 'not HS', 'total']]
edu_18['edu rate'] = edu_18.apply(lambda \
                    row: (row['total']-row['not HS'])/row['total'], axis=1)

# pull income data of 2018
income_raw_18 = requests.get('https://api.census.gov/data/2018/pdb/tract?get'
                        '=Aggregate_HH_INC_ACS_12_16&for=tract:*&in=state:'
                        '17%20county:031')
income_data_18 = income_raw_18.json()
income_df_18 = pd.DataFrame(income_data_18)
header_18 = income_df_18.iloc[0]
income_df_18 = income_df_18[1:]
income_df_18.columns = header_18
income_18 = income_df_18.rename(\
                        columns = {'Aggregate_HH_INC_ACS_12_16': 'income'})
income_18 = income_18.dropna()
income_18['income'] = income_18.apply(lambda row: \
            int(''.join(x for x in row['income'] if x.isdigit())), axis=1)
income_18 = income_18.astype({'income': int, 'tract': int})

# merge data
cri_17_t = pd.merge(cri_17, convert_community_tract, on='Community Area')
dfs_17 = [cri_17_t, race_17, edu_17, income_17]
df_final_17 = reduce(lambda \
                left, right: pd.merge(left, right, on='tract'), dfs_17)

cri_18_t = pd.merge(cri_18, convert_community_tract, on='Community Area')
dfs_18 = [cri_18_t, race_18, edu_18, income_18]
df_final_18 = reduce(lambda \
                left, right: pd.merge(left, right, on='tract'), dfs_18)


def describe_edu_in_block_by_crime(df, crime):
    '''
    Given a dataset and a specific crime type, calculate the rate
    of the uneducated people in certain area.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    return:
        (float) rate of educated people
    '''
    avg_edu_rate = df['edu rate'].mean()
    block = df[df['Primary Type'] == crime]
    edu_rate = block.groupby(['edu rate']).size().reset_index()\
                            .rename(columns={0: 'count'})
    uneducated_rate = (edu_rate[edu_rate['edu rate'] < \
                    avg_edu_rate]['count'].sum()) / block.shape[0]

    return uneducated_rate


def describe_race_in_block_by_crime_17(df, crime):
    '''
    Given a dataset of 2017 and a specific crime type, output a barplot
    of the distribution of races among blocks of certain type of crime.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    '''
    block = df[df['Primary Type'] == crime]
    block['black dom'] = block.apply(lambda \
                    row: 1 if row['black'] > 0.5 else 0, axis=1)
    block['white dom'] = block.apply(lambda \
                    row: 1 if row['white'] > 0.5 else 0, axis=1)
    block['asian dom'] = block.apply(lambda \
                    row: 1 if row['asian'] > 0.5 else 0, axis=1)
    block['native dom'] = block.apply(lambda \
            row: 1 if row['nativehawaiian'] > 0.5 else 0, axis=1)
    block['other dom'] = block.apply(lambda \
                    row: 1 if row['other'] > 0.5 else 0, axis=1)
    black_dom = block['black dom'].sum() / block.shape[0]
    white_dom = block['white dom'].sum() / block.shape[0]
    asian_dom = block['asian dom'].sum() / block.shape[0]
    native_dom = block['native dom'].sum() / block.shape[0]
    other_dom = block['other dom'].sum() / block.shape[0]
    d_race = {'race': ['white', 'black', 'asian', 'native', 'other'],
        'ratio': [white_dom, black_dom, asian_dom, native_dom, other_dom]}
    df_race = pd.DataFrame(d_race)
    sns.barplot(x='race', y='ratio', data=df_race)\
       .set_title('Distribution of Race Ratio among Blocks with Crimes of '\
                                                        + crime + ' 2017')
    plt.tight_layout()
    plt.show()


def describe_income_in_block_by_crime_17(df, crime):
    '''
    Given a dataset of 2017 and a specific crime type, output a barplot
    of the distribution of incomes among blocks of certain type of crime.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    '''
    block = df[df['Primary Type'] == crime]
    block['< 50000 dom'] = block.apply(lambda \
                    row: 1 if row['< 50000'] > 0.5 else 0, axis=1)
    block['50000-100000 dom'] = block.apply(lambda \
            row: 1 if row['50000 to 100000'] > 0.5 else 0, axis=1)
    block['> 100000 dom'] = block.apply(lambda \
                    row: 1 if row['> 100000'] > 0.5 else 0, axis=1)
    low_income_dom = block['< 50000 dom'].sum() / block.shape[0]
    mediate_income_dom = block['50000-100000 dom'].sum() / block.shape[0]
    high_income_dom = block['> 100000 dom'].sum() / block.shape[0]
    d_income = {'income': ['low', 'mediate', 'high'],
            'ratio': [low_income_dom, mediate_income_dom, high_income_dom]}
    df_income = pd.DataFrame(d_income)
    sns.barplot(x='income', y='ratio', data=df_income)\
       .set_title('Distribution of Income Ratio among Blocks with Crimes of '\
                                                            + crime + ' 2017')
    plt.tight_layout()
    plt.show()


def describe_race_in_block_by_crime_18(df, crime):
    '''
    Given a dataset of 2018 and a specific crime type, output a barplot
    of the distribution of races among blocks of certain type of crime.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    '''
    block = df[df['Primary Type'] == crime]
    block['black dom'] = block.apply(lambda \
                        row: 1 if row['black'] > 0.5 else 0, axis=1)
    block['white dom'] = block.apply(lambda \
                        row: 1 if row['white'] > 0.5 else 0, axis=1)
    block['native dom'] = block.apply(lambda \
                        row: 1 if row['native'] > 0.5 else 0, axis=1)
    block['other dom'] = block.apply(lambda \
                        row: 1 if row['other'] > 0.5 else 0, axis=1)
    black_dom = block['black dom'].sum() / block.shape[0]
    white_dom = block['white dom'].sum() / block.shape[0]
    native_dom = block['native dom'].sum() / block.shape[0]
    other_dom = block['other dom'].sum() / block.shape[0]
    d_race = {'race': ['white', 'black', 'native', 'other'],
            'ratio': [white_dom, black_dom, native_dom, other_dom]}
    df_race = pd.DataFrame(d_race)
    sns.barplot(x='race', y='ratio', data=df_race)\
       .set_title('Distribution of Race Ratio among Blocks with Crimes of '\
                                                        + crime + ' 2018')
    plt.tight_layout()
    plt.show()


def describe_income_in_block_by_crime_18(df, crime):
    '''
    Given a dataset of 2018 and a specific crime type, output a barplot
    of the distribution of incomes among blocks of certain type of crime.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    '''
    block = df[df['Primary Type'] == crime]
    avg_income = df['income'].mean()
    one_four = avg_income * 0.5
    three_four = avg_income * 1.5
    block['low income'] = block.apply(lambda \
                    row: 1 if row['income'] < one_four else 0, axis=1)
    block['mediate income'] = block.apply(lambda \
        row: 1 if one_four < row['income'] < three_four else 0, axis=1)
    block['high income'] = block.apply(lambda \
                    row: 1 if row['income'] > three_four else 0, axis=1)
    low_income_rate = block['low income'].sum() / block.shape[0]
    mediate_income_rate = block['mediate income'].sum() / block.shape[0]
    high_income_rate = block['high income'].sum() / block.shape[0]
    d_income = {'income': ['low', 'mediate', 'high'],
        'ratio': [low_income_rate, mediate_income_rate, high_income_rate]}
    df_income = pd.DataFrame(d_income)
    sns.barplot(x='income', y='ratio', data=df_income)\
       .set_title('Distribution of Income Ratio among Blocks with Crimes of '\
                                                        + crime + ' 2018')
    plt.tight_layout()
    plt.show()


# Question 1
# analyze education status
uneducated_rate_battery_17 = \
            describe_edu_in_block_by_crime(df_final_17, 'BATTERY')
# 0.39055717385293115
uneducated_rate_battery_18 = \
            describe_edu_in_block_by_crime(df_final_18, 'BATTERY')
# 0.5199007173792402

# analyze race distribution
describe_race_in_block_by_crime_17(df_final_17, 'BATTERY')
describe_race_in_block_by_crime_18(df_final_18, 'BATTERY')

# analyze income status
describe_income_in_block_by_crime_17(df_final_17, 'BATTERY')
describe_income_in_block_by_crime_18(df_final_18, 'BATTERY')

# Question 2
# analyze education status
uneducated_rate_homicide_17 = \
        describe_edu_in_block_by_crime(df_final_17, 'HOMICIDE')
# 0.42494939810376053
uneducated_rate_homicide_18 = \
        describe_edu_in_block_by_crime(df_final_18, 'HOMICIDE')
# 0.6294191919191919

# analyze race distribution
describe_race_in_block_by_crime_17(df_final_17, 'HOMICIDE')
describe_race_in_block_by_crime_18(df_final_18, 'HOMICIDE')

# analyze income status
describe_income_in_block_by_crime_17(df_final_17, 'HOMICIDE')
describe_income_in_block_by_crime_18(df_final_18, 'HOMICIDE')


# Question 4
# analyze education status
uneducated_rate_dp_17 = \
    describe_edu_in_block_by_crime(df_final_17, 'DECEPTIVE PRACTICE')
# 0.3069640669513088
uneducated_rate_dp_18 = \
    describe_edu_in_block_by_crime(df_final_18, 'DECEPTIVE PRACTICE')
# 0.3210836661644895
uneducated_rate_so_17 = \
    describe_edu_in_block_by_crime(df_final_17, 'SEX OFFENSE')
# 0.4198565442795256
uneducated_rate_so_18 = \
    describe_edu_in_block_by_crime(df_final_18, 'SEX OFFENSE')
# 0.44556930383702037

# analyze race distribution
describe_race_in_block_by_crime_17(df_final_17, 'DECEPTIVE PRACTICE')
describe_race_in_block_by_crime_18(df_final_18, 'DECEPTIVE PRACTICE')

describe_race_in_block_by_crime_17(df_final_17, 'SEX OFFENSE')
describe_race_in_block_by_crime_18(df_final_18, 'SEX OFFENSE')

# analyze income status
describe_income_in_block_by_crime_17(df_final_17, 'DECEPTIVE PRACTICE')
describe_income_in_block_by_crime_18(df_final_18, 'DECEPTIVE PRACTICE')

describe_income_in_block_by_crime_17(df_final_17, 'SEX OFFENSE')
describe_income_in_block_by_crime_18(df_final_18, 'SEX OFFENSE')



#############
# Problem 3 #
#############

def find_record(df, crime):
    '''
    Given a dataset and a certain type of crime,
    find the subset of certain periods.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    Return:
        filtered dataset(dataframe)
    '''
    pattern = r'07\/2[0-6]'
    filtered_df = df[df['Primary Type'] == crime]
    filtered_df['match'] = filtered_df.apply(lambda \
            row: 1 if re.search(pattern, row['Date']) else 0, axis=1)
    return filtered_df[filtered_df['match'] == 1]


def compute_change(df17, df18, crime):
    '''
    Given two datasets and a certain type of crime,
    compute the change of the crime in certain periods.
    Input:
        df(dataframe): processed dataset
        crime(str): certain type of crime
    Return:
        change of the crime in certain period (float)
    '''    
    filtered_df17 = find_record(df17, crime)
    filtered_df18 = find_record(df18, crime)
    prev = filtered_df17.shape[0]
    cur = filtered_df18.shape[0]
    return (cur - prev) / prev * 100


change_robbery = compute_change(cri_17, cri_18, 'ROBBERY')
change_battery = compute_change(cri_17, cri_18, 'BATTERY')
change_burglary = compute_change(cri_17, cri_18, 'BURGLARY')
change_mvt = compute_change(cri_17, cri_18, 'MOTOR VEHICLE THEFT')



#############
# Problem 4 #
#############
# Question 1
michigan_ave = cri_18[cri_18['Community Area'] == 33]
total = michigan_ave.shape[0]
prob = michigan_ave.groupby(['Primary Type']).size() / total * 100
prob.sort_values(ascending=False)

#Question 2
theft = cri_18[cri_18['Primary Type'] == 'THEFT']
area = theft[theft['Community Area'].isin([3, 26, 27])]
total = area.shape[0]
prob = area.groupby(['Community Area']).size() / total * 100

