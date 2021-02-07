
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw dataset:
df = pd.read_csv('at_home_data_group.csv')

# Check datatypes:
types = df.dtypes

    # Fix wrong datatypes:
df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
df['Private Vehicle'] = df['Private Vehicle'].astype(float)

# Eliminate irrelevant columns:
df.drop(['city_sales_rev'],axis=1, inplace = True)

# NUMERICAL VARIABLES:
    
# Get summary statistics
summary = df.describe()

# Study variable distribution
num_vars = []
num_vars = list(df.select_dtypes(exclude=['object']).columns)

# Generate frequency histograms
for i in num_vars:
    plt.hist(df[i])
    plt.title(f'{i} Frequency Histogram')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()
    
bins = []
for i in np.arange(0,9,0.5):
    bins.append(i)
plt.hist(df.PF, bins = bins)
plt.title('Pull Factor Frequency Histogram')
plt.xlabel('PF')
plt.ylabel('Frequency')

# CHEACK OUTLIERS:

# Generate boxplots
for i in num_vars:
    plt.boxplot(df[i])
    plt.title(f'{i} Boxplot')
    plt.ylabel(i)
    plt.show()

# Check inner boxplot fences and determine number of outliers
for i in num_vars:
    iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
    lb = df[i].quantile(0.25) - (1.5 * iqr)
    ub = df[i].quantile(0.75) + (1.5 * iqr)
    cond = df[df[i].between(lb,ub)]
    num_out = 219 - cond.shape[0]
    print(f'{i}(lower bound: {lb} ; upper bound" {ub}) / Number of Outliers: {num_out}')

# CATEGORICAL VARIABLES:

# Univariate analysis
cat_vars = []
cat_vars = list(df.select_dtypes(include=['object']).columns)

for i in cat_vars[6:9]:
    df[i].value_counts().plot(kind='bar')
    plt.title(f'Number of stores with {i} service')
    plt.ylabel('Count')
    plt.show()
    
for i in cat_vars:
    df[i].value_counts().plot(kind='bar')
    plt.title(f'{i} bar plot')
    plt.ylabel('Count')
    plt.show()
    
# Correlation Matrix:
corr = df.corr()
corr.to_excel('initial_corr(group).xlsx')
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True, annot_kws = {'size': 8}, fmt = '.2f', cbar = False)
plt.title('Correlation Matrix')
plt.xlabel('Numeric Features')
plt.ylabel('Numeric Features')

# Feature engineering:
    
df1 = df.copy(deep = True)

# Create Occupancy Rate measure:
df1['occupancy_rate'] = ''
for i in df1.index:
    df1.loc[i, 'occupancy_rate'] = df1.occupied_housing_units[i]/df1.housing_units[i]
    
# Population by Age:
# Group ages into three features:
    #1. children and young adolescents (ages < 19)
    #2. working age (ages 19 - 64)
    #3. elderly age (>65)

df1['child_teen_rate'] = ''
df1['working_age_rate'] = ''
df1['elderly_rate'] = ''

for i in df1.index:
    df1.loc[i, 'child_teen_rate'] = (df1.loc[i, 'pop_age_under_5'] + 
    df1.loc[i, 'pop_age_5-9'] + df1.loc[i, 'pop_age_10-14'] +
    df1.loc[i, 'pop_age_15-19'])/df.loc[i, 'population']

for i in df1.index:
    df1.loc[i, 'working_age_rate'] = (df1.loc[i, 'pop_age_20-24'] + 
    df1.loc[i, 'pop_age_25-29'] + df1.loc[i, 'pop_age_30-34'] +
    df1.loc[i, 'pop_age_35-39'] + df1.loc[i, 'pop_age_40-44'] +
    df1.loc[i, 'pop_age_45-49'] + df1.loc[i, 'pop_age_50-54'] +
    df1.loc[i, 'pop_age_55-59'] + df1.loc[i, 'pop_age_60-64'])/df.loc[i, 'population']
    
for i in df1.index:
    df1.loc[i, 'elderly_rate'] = (df1.loc[i, 'pop_age_65-69'] + 
    df1.loc[i, 'pop_age_70-74'] + df1.loc[i, 'pop_age_75-79'] + 
    df1.loc[i, 'pop_age_80-84'] + df1.loc[i, 'pop_age_85_plus'])/df.loc[i, 'population']
    
# Population by gender:
# Convert pop_male and pop_female into female_rate
df1['female_rate'] = ''
for i in df1.index:
    df1.loc[i, 'female_rate'] = df1.pop_female[i]/df1.population[i]

# Population by race:
# Convert pop_white, pop_black, pop_native_am, pop_asian, pop_hawaii, pop_other,
# pop_two_plus into white_pop_rate
df1['white_pop_rate'] = ''
for i in df1.index:
    df1.loc[i, 'white_pop_rate'] = df1.pop_white[i]/df1.population[i]
    
df_pop = df1[['pop_white', 'pop_black', 'pop_native_am', 'pop_asian', 'pop_hawaii', 'pop_other', 'pop_two_plus']]
df_pop.rename(columns={'pop_white':'white', 'pop_black':'black',
                       'pop_native_am': 'native american', 'pop_asian':'asian',
                       'pop_hawaii':'hawaiian', 'pop_other':'other',
                       'pop_two_plus':'two or more'}, inplace = True)
race_majority = df_pop.idxmax(axis = 1, skipna = True)
race_majority = pd.DataFrame(race_majority)
race_minority = df_pop.iloc[:,1:].idxmax(axis = 1, skipna = True) 
race_minority = pd.DataFrame(race_minority)   

df1['race_majority'] = race_majority[0]
df1['race_minority'] = race_minority[0]

# Households by marital status:
# Convert husb_wife, single_guard, single, sing_room into married_hh_rate
df1['married_hh_rate'] = ''
for i in df1.index:
    df1.loc[i, 'married_hh_rate'] = df1.husb_wife[i]/df1.occupied_housing_units[i]

# Households with kids:
# Convert hh_no_kids, hh_kids into hh_kids_rate
df1['hh_kids_rate'] = ''
for i in df1.index:
    df1.loc[i, 'hh_kids_rate'] = df1.hh_kids[i]/df1.occupied_housing_units[i]

# Households by owning or renting accomodation:
# Convert hh_mortgage, hh_owned, hh_rent into owned_hh_rate
df1['owned_hh_rate'] = ''
for i in df1.index:
    df1.loc[i, 'owned_hh_rate'] = df1.hh_owned[i]/df1.occupied_housing_units[i]

# Household vacancy rate:
# Convert hh_vacant
df1['hh_vacancy_rate'] = ''
for i in df1.index:
    df1.loc[i, 'hh_vacancy_rate'] = df1.hh_vacant[i]/df1.occupied_housing_units[i]

# Employment status:
# Convert full_time_work, part_time_work and no_earnings into income_earners_rate
df1['income_earners_rate'] = ''
for i in df1.index:
    df1.loc[i, 'income_earners_rate'] = (1 -  df1.no_earnings[i]/(df1.full_time_work[i]+df1.part_time_work[i]+df1.no_earnings[i]))
    
# Income brackets:
df1['low_income_hh_rate'] = ''
for i in df1.index:
    df1.loc[i, 'low_income_hh_rate'] = ((df1['hi_less_25k'][i]+df1['hi_25_44,9k'][i])/
    (df1['hi_less_25k'][i]+df1['hi_25_44,9k'][i]+df1['hi_45_59,9k'][i]+df1['hi_60_99,9k'][i]+
     df1['hi_100_149,9k'][i]+df1['hi_150_199,9k'][i]+df1['hi_200k+'][i]))

df1['middle_income_hh_rate'] = ''
for i in df1.index:
    df1.loc[i, 'middle_income_hh_rate'] = ((df1['hi_45_59,9k'][i]+df1['hi_60_99,9k'][i]+
     df1['hi_100_149,9k'][i])/
    (df1['hi_less_25k'][i]+df1['hi_25_44,9k'][i]+df1['hi_45_59,9k'][i]+df1['hi_60_99,9k'][i]+
     df1['hi_100_149,9k'][i]+df1['hi_150_199,9k'][i]+df1['hi_200k+'][i]))

df1['high_income_hh_rate'] = ''
for i in df1.index:
    df1.loc[i, 'high_income_hh_rate'] = ((df1['hi_150_199,9k'][i]+df1['hi_200k+'][i])/
    (df1['hi_less_25k'][i]+df1['hi_25_44,9k'][i]+df1['hi_45_59,9k'][i]+df1['hi_60_99,9k'][i]+
     df1['hi_100_149,9k'][i]+df1['hi_150_199,9k'][i]+df1['hi_200k+'][i]))
  
# Work from Home Rate:
df1['work_from_home_rate'] = ''

for i in df1.index:
    df1.loc[i, 'work_from_home_rate'] = (df1['Work from Home'][i]/(
        df1['Private Vehicle'][i]+df1['Public Transportation'][i]+
        df1['Taxi'][i]+df1['Motorcycle'][i]+df1['Bicycle, Walking or Other'][i]+df1['Work from Home'][i]))
    
# Higher education rate:
df1['higher_education_rate'] = ''

for i in df1.index:
    df1.loc[i, 'higher_education_rate'] = ((df1['Associate Degree'][i]+df1['Bachelor Degree'][i]+ df1['Master Degree'][i]+df1['Professional School Degree'][i]+df1['Doctorate Degree'][i])/(
        (df1['Less than High School'][i]+df1['High School Graduate'][i]+df1['Associate Degree'][i]+df1['Bachelor Degree'][i]+ df1['Master Degree'][i]+df1['Professional School Degree'][i]+df1['Doctorate Degree'][i])))

# Average yearly population growth:
df1['average_year_pop_growth_rate'] = ''

for i in df1.index:
    df1.loc[i, 'average_year_pop_growth_rate'] = ((df1['pop_growth_2011'][i]+df1['pop_growth_2012'][i]+df1['pop_growth_2013'][i]+df1['pop_growth_2014'][i]+df1['pop_growth_2015'][i]+df1['pop_growth_2016'][i]+df1['pop_growth_2017'][i]+df1['pop_growth_2018'][i]+df1['pop_growth_2019'][i])/9)*100                                                    

# Drop columns already contemplated on engineered features:
df2 = df1.copy(deep = True)

df2.drop(columns = ['occupied_housing_units', 'housing_units', 'pop_age_under_5',
                     'pop_age_5-9', 'pop_age_10-14', 'pop_age_15-19',
                     'pop_age_20-24', 'pop_age_25-29', 'pop_age_30-34', 'pop_age_35-39',
                     'pop_age_40-44', 'pop_age_45-49', 'pop_age_50-54', 'pop_age_55-59', 
                     'pop_age_60-64', 'pop_age_65-69', 'pop_age_70-74', 'pop_age_75-79',
                     'pop_age_80-84', 'pop_age_85_plus', 'pop_female', 'pop_white',
                     'husb_wife', 'hh_kids', 'hh_owned', 'hh_vacant', 'no_earnings',
                     'full_time_work', 'part_time_work', 'hi_less_25k', 'hi_25_44,9k',
                     'hi_45_59,9k', 'hi_60_99,9k', 'hi_100_149,9k','hi_150_199,9k', 'hi_200k+',
                     'ID', 'name', 'address', 'city', 'state', 'zip_code', 'latitude',
                     'longitude', 'population','land_area_in_sqmi','pop_male', 'pop_black',
                     'pop_native_am', 'pop_asian', 'pop_hawaii',
                     'pop_other', 'pop_two_plus', 'single_guard', 'single', 'sing_room',
                     'hh_no_kids', 'hh_mortgage', 'hh_rent', 'state_full', 'estab',
                     'sales_revenue_2012', 'num_emp', 'city_sales_rev_final', 'Pop State',
                     'PIC_City', 'PCI State', 'TAC',
                     'race_majority', 'county','Work from Home', 'Private Vehicle',
                     'Public Transportation', 'Taxi', 'Motorcycle', 'Bicycle, Walking or Other',
                     'Less than High School', 'High School Graduate', 'Associate Degree', 
                     'Bachelor Degree', 'Master Degree', 'Professional School Degree', 'Doctorate Degree',
                     'pop_2010','pop_2011','pop_2012','pop_2013','pop_2014','pop_2015','pop_2016','pop_2017',
                     'pop_2018','pop_2019','pop_growth_2011','pop_growth_2012','pop_growth_2013',
                     'pop_growth_2014','pop_growth_2015','pop_growth_2016','pop_growth_2017',
                     'pop_growth_2018','pop_growth_2019'],
                     inplace = True)

df2.columns
df2.dropna(inplace = True)
df2.dtypes
df2.drop(columns = ['elderly_rate'], inplace = True)
df2 = df2.astype({'occupancy_rate': float, 'child_teen_rate': float,
                  'working_age_rate': float,'female_rate': float,
                  'white_pop_rate': float, 'married_hh_rate': float, 'hh_kids_rate': float,
                  'owned_hh_rate': float, 'hh_vacancy_rate': float, 'income_earners_rate': float,
                  'low_income_hh_rate': float, 'middle_income_hh_rate': float,
                  'high_income_hh_rate': float, 'work_from_home_rate': float,
                  'higher_education_rate': float, 'average_year_pop_growth_rate':float})
df2.dtypes

c_matrix = df2.corr()
c_matrix.to_excel('final_correlation_matrix(group).xlsx')

df2.drop(columns = ['hh_vacancy_rate', 'high_income_hh_rate',
                    'low_income_hh_rate', 'hh_kids_rate', 'median_home_value',
                    'married_hh_rate'], inplace = True)


c_matrix1 = df2.corr()
c_matrix1.to_excel('final_correlation_matrix (group project).xlsx')

# Export data
df2.to_csv('group_data.csv')

# Once the exploratory data analysis is done we exported the data to combine
# the current data exported with the sales data provided by the proffesor.
# In that matter, we proceed to perform a VLOOKUP() function to match the sales 
# information with our data.

# Importing data including the sales variables:
at_home = pd.read_csv('final_group_data.csv')
data = pd.read_csv('final_group_data.csv')

# Value counts categorical data:
data.iloc[:,1].value_counts()
data.iloc[:,2].value_counts()
data.iloc[:,3].value_counts()
data.iloc[:,12].value_counts()

# Null values:
data.isnull().sum()

# Data modeling:
data.drop(columns = ['Unnamed: 0', 'ID', 'name', 'address', 'city', 
                     'state', 'zip_code', 'latitude', 'longitude'], 
          inplace = True)

# Identifing null values:
    # Null values for year 2020
ind = -1
list_ind_20 = []
for i in data.iloc[:, 21].isnull():
    ind = ind + 1
    if i == True:
        list_ind_20.append(ind)
                
    # Dataframe containing the stores with no sales by the year 2020
data_no_sales = pd.DataFrame()
for i in list_ind_20:
    data_no_sales.loc[i,'name'] = at_home.loc[i,'name']
    data_no_sales.loc[i,'address'] = at_home.loc[i,'address']
    data_no_sales.loc[i,'city'] = at_home.loc[i,'city']
    data_no_sales.loc[i,'state'] = at_home.loc[i,'state']
    
    # # Null values for year 2019 and backwards
ind = -1
list_ind_19 = []
for i in data.iloc[:, 20].isnull():
    ind = ind + 1
    if i == True:
        list_ind_19.append(ind)

list_new = []        
for i in list_ind_19:
    if i not in list_ind_20:
        list_new.append(i)

    #Dataframe containing the stores with no sales for 2019 and backwards
data_no_sales19 = pd.DataFrame()
for i in list_new:
    data_no_sales19.loc[i,'name'] = at_home.loc[i,'name']
    data_no_sales19.loc[i,'address'] = at_home.loc[i,'address']
    data_no_sales19.loc[i,'city'] = at_home.loc[i,'city']
    data_no_sales19.loc[i,'state'] = at_home.loc[i,'state']

# Droping the stores with no sales in 2019 and 2020:
data = data.drop(data.index[list_ind_19])
data = data.reset_index()
data = data.drop( 'index', axis = 1)

# Filling null values with zeros:
data['2017 Sales'] = data['2017 Sales'].fillna(0)
data['2018 Sales'] = data['2018 Sales'].fillna(0)
data.isnull().sum()

# Feature engineering:
data['17-18_sales_growth'] = np.nan
data['18-19_sales_growth'] = np.nan
data['19-20_sales_growth'] = np.nan

for i in range(201):
    val = data.loc[i, '2018 Sales'] - data.loc[i, '2017 Sales']
    per = val / data.loc[i, '2017 Sales']
    data.loc[i, '17-18_sales_growth']= per

for i in range(201):
    val = data.loc[i, '2019 Sales'] - data.loc[i, '2018 Sales']
    per = val / data.loc[i, '2018 Sales']
    data.loc[i, '18-19_sales_growth']= per

for i in range(201):
    val = data.loc[i, '2020 Sales'] - data.loc[i, '2019 Sales']
    per = val / data.loc[i, '2019 Sales']
    data.loc[i, '19-20_sales_growth']= per

data['17-18_sales_growth'] = data['17-18_sales_growth'].fillna(0)
for i in range(201):
    if data.loc[i, '17-18_sales_growth'] == float('inf'):
        data.loc[i, '17-18_sales_growth'] = 0

data['18-19_sales_growth'] = data['18-19_sales_growth'].fillna(0)
for i in range(201):
    if data.loc[i, '18-19_sales_growth'] == float('inf'):
        data.loc[i, '18-19_sales_growth'] = 0

data['19-20_sales_growth'] = data['19-20_sales_growth'].fillna(0)
for i in range(201):
    if data.loc[i, '19-20_sales_growth'] == float('inf'):
        data.loc[i, '19-20_sales_growth'] = 0

    # Counting the number of year the store has been open
data['count'] = np.nan

for i in range(201):
    c = 0
    if data.loc[i, '19-20_sales_growth'] != 0:
        c = c+1
        if data.loc[i, '18-19_sales_growth'] != 0:
            c = c+1
            if data.loc[i, '17-18_sales_growth'] != 0:
                c = c+1
    data.loc[i, 'count'] = c

    # Creating the target variable
data['avg_sales_growth'] = np.nan

for i in range(201):
    data.loc[i,'avg_sales_growth'] = (data.loc[i,'17-18_sales_growth'] +
                                      data.loc[i, '18-19_sales_growth'] +
                                      data.loc[i, '19-20_sales_growth'])/(
                                          data.loc[i, 'count'])

    # Droping unnecessary columns
data.drop(columns = ['2017 Sales', '2018 Sales','2019 Sales', '2020 Sales',
                     '17-18_sales_growth', '18-19_sales_growth',
                     '19-20_sales_growth', 'count'], inplace = True)

# Correlation matrix:
corr = data.corr()
corr.to_excel(r'Corr_matrix.xlsx', index = False)

# Encoding categorical variables:
    # Labeling encoding for service type variables
for i in range(201):
    if data.iloc[i,0] == 'No':
        data.iloc[i,0] = 0
    else:
        data.iloc[i,0] = 1

for i in range(201):
    if data.iloc[i,1] == 'No':
        data.iloc[i,1] = 0
    else:
        data.iloc[i,1] = 1

for i in range(201):
    if data.iloc[i,2] == 'No':
        data.iloc[i,2] = 0
    else:
        data.iloc[i,2] = 1

    # Dummy method for "race_minority" variable
dummies = pd.get_dummies(data.loc[:,'race_minority'])
dummies = dummies.rename(columns={"asian": "rm_asian", 
                                  "black": "rm_black",
                                  "native american": "rm_native_american",
                                  "other": "rm_other",
                                  "two or more": "rm_two_or_more"})

for i in dummies.columns:
    data[i] = dummies[i].values

# Reorganizing the columns:
data = data[['pick_up_in_store', 'pick_up_curbside', 'local_store_delivery',
             'population_density', 'median_household_income', 'PF', 'occupancy_rate',
             'child_teen_rate', 'working_age_rate', 'female_rate', 'white_pop_rate',
             'race_minority', 'rm_asian', 'rm_black','rm_native_american', 'rm_other',
             'rm_two_or_more', 'owned_hh_rate', 'income_earners_rate',
             'middle_income_hh_rate', 'work_from_home_rate', 'higher_education_rate',
             'average_year_pop_growth_rate', 'avg_sales_growth']]

data.drop(columns = ['race_minority'], inplace = True)

# Train and test dataset:
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:,3:11] = sc.fit_transform(X_train.iloc[:,3:11])
X_test.iloc[:,3:11] = sc.transform(X_test.iloc[:,3:11])
X_train.iloc[:,16:] = sc.fit_transform(X_train.iloc[:,16:])
X_test.iloc[:,16:] = sc.transform(X_test.iloc[:,16:])

# Variable importance:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 1)

model.fit(X_train, y_train)

importance = model.feature_importances_

    # Summarize feature importance
for i,v in enumerate(importance): print('Feature: %0d, Score: %.5f' % (i,v))

    # Plot feature importance
l = []
for x in range(len(importance)):
    l.append(x)

plt.bar(l, importance)
plt.xticks(range(len(l)), ('pick_up_in_store', 'pick_up_curbside', 'local_store_delivery',
       'population_density', 'median_household_income', 'PF', 'occupancy_rate',
       'child_teen_rate', 'working_age_rate', 'female_rate', 'white_pop_rate',
       'rm_asian', 'rm_black', 'rm_native_american', 'rm_other',
       'rm_two_or_more', 'owned_hh_rate', 'income_earners_rate',
       'middle_income_hh_rate', 'work_from_home_rate', 'higher_education_rate',
       'average_year_pop_growth_rate'), 
           rotation = 90, fontsize = 8)
plt.title("Variable Importance")
plt.xlabel("Variable")
plt.ylabel("Importance")
plt.show()

# Selected important variables discribe in the variable importance
data_vi = pd.DataFrame()
data_vi['median_household_income'] = data['median_household_income']
data_vi['child_teen_rate'] = data['child_teen_rate']
data_vi['rm_two_or_more'] = data['rm_two_or_more']
data_vi['owned_hh_rate'] = data['owned_hh_rate']
data_vi['higher_education_rate'] = data['higher_education_rate']
data_vi['avg_sales_growth'] = data['avg_sales_growth']

# Train and test dataset:
X = data_vi.iloc[:,:-1]
y = data_vi.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 1)

# Feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:,0:2] = sc.fit_transform(X_train.iloc[:,0:2])
X_test.iloc[:,0:2] = sc.transform(X_test.iloc[:,0:2])
X_train.iloc[:,3:] = sc.fit_transform(X_train.iloc[:,3:])
X_test.iloc[:,3:] = sc.transform(X_test.iloc[:,3:])

# Modeling:
    # Multilinear Regression
from sklearn.linear_model import LinearRegression
multi_regressor = LinearRegression()
multi_regressor.fit(X_train, y_train)

y_pred = multi_regressor.predict(X_test)

    # Performance metrics
        # Mean Squared Error(MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared = True)
##MSE = 0.1717

        # Root-Mean-Squared-Error(RMSE)
mean_squared_error(y_test, y_pred, squared = False)
##RMSE = 0.4144

        # Mean-Absolute-Error(MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
##MAE = 0.2837

        # R2 or Coefficient of Determination
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
##r2 = -0.0789 = -7.89%

    # Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

    # Performance metrics
        # Mean Squared Error(MSE)
mean_squared_error(y_test, y_pred, squared = True)
##MSE = 0.5264

        # Root-Mean-Squared-Error(RMSE)
mean_squared_error(y_test, y_pred, squared = False)
##RMSE = 0.7256

        # Mean-Absolute-Error(MAE)
mean_absolute_error(y_test, y_pred)
##MAE = 0.3696

        # R2 or Coefficient of Determination
r2_score(y_test, y_pred)
##r2 = -2.3077 = -230.77%

    # SVR
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

    # Performance metrics
        # Mean Squared Error(MSE)
mean_squared_error(y_test, y_pred, squared = True)
##MSE = 0.1615

        # Root-Mean-Squared-Error(RMSE)
mean_squared_error(y_test, y_pred, squared = False)
##RMSE = 0.4018

        # Mean-Absolute-Error(MAE)
mean_absolute_error(y_test, y_pred)
##MAE = 0.2208

        # R2 or Coefficient of Determination
r2_score(y_test, y_pred)
##r2 = -0.0146 = -1.46%

# Hyper parameter tunning:
from sklearn.model_selection import GridSearchCV

    # Define parameter grid
params = {'kernel':['rbf'],
          'gamma':[0.0001, 0.001, 0.01, 0.1],
          'C':[1, 10, 100, 1000]}

    # Instantiate a support vector regressor
svm = SVR()

    # Instantiate a GridSearchCV classifier with 10 fold cross-validation
grid = GridSearchCV(svm, params, cv = 10, verbose = 2)
grid.fit(X_train, y_train)

    # Generate predictions and calculate accuracy error
y_pred = grid.predict(X_test)
print('Best parameters: ', grid.best_params_)

    # Performance metrics:
        # Mean Squared Error(MSE)
mean_squared_error(y_test, y_pred, squared = True)
##MSE = 0.1642

        # Root-Mean-Squared-Error(RMSE)
mean_squared_error(y_test, y_pred, squared = False)
##RMSE = 0.4053

        # Mean-Absolute-Error(MAE)
mean_absolute_error(y_test, y_pred)
##MAE = 0.2069

        # R2 or Coefficient of Determination
r2_score(y_test, y_pred)
##r2 = -0.0319 = -3.19%


