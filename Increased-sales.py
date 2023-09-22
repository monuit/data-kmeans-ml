# %% [markdown]
# ### Note: 
#     
#     4 different analysis has been applied to sales data in order to possibily increase the quantity %. 
#     
#     Each analysis will generate an excel file and in that excel file you will see-
#     'increase_percentage_quantity' as last column. That column will guide you how much quantity should be
#     increased in order to increase the sales. Check the following:
#     
# **Analysis 1:** ==> You can increase the 'UNIT' value for the highest repeated value either [Cases, pallets, eaches].<br><br>
#                 e.g: the below analysis-1 shows **Cases** has more importance than 'Pallets' and 'Eaches'.
#                      so, you can increase the quantity percentage of **Cases** to increase the sales.
# 
# 
# **Anslysis 2:** ==> You can increase in the amount of time spent for **closedwork** and **workinprocess** to increase 
#                     sales percentage accordingly.
#                     
# 
# **Analysis 3:** ==> You can increase the values in columns Unit, workingprocess and closed work to increase the sales percentage.
# 
# 
# **Analysis 4:** ==> Rather than selecting the random columns, let machine learning to decides the best column and increase the quantity accordingly. This technique will help you to increase the quantity values in each possible column to increase the sales. (Note: the amount of quantity to be increased in each possible column is given in excel file 'Percentage_of_Quantity_to_be_increased_with_best_columns.csv' )

# %% [markdown]
# ## Libs

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# %% [markdown]
# ## Load data

# %%
df = pd.read_excel('Working table sales data dump for year 2022.xlsx')

# %%
org_data = df

# %%
## data samples

df.sample(frac=1).head(10)

# %% [markdown]
# ## Pre-processing data

# %%
# total records

print(f"Total records in Sales Data: ==> ", df.shape[0])

# %%
## duplicated records

print(f"Total duplicated values in Sales Data: ==> ", df.duplicated().sum())

# %%
## missing values

print(f"Missing values in Sales-Data:\n")

df.isnull().sum()

# %%
# filtering sales data

def filtering(df):
    return df.rename(columns={col:col.lower().replace(" ", "_") for col in df.columns.values.tolist()})

# %%
df = filtering(df)

# %%
# filtered columns

print(f"Filtered Columns: \n")
df.columns.tolist()

# %%
## handling missing values

df.fillna(0, inplace=True)

# %% [markdown]
# ## Analysis: 1 ===> Increase quantity % based on the UNIT [Each, Cases, or Pallet]

# %% [markdown]
# ### Create a new column that calculates the total number of cases per order

# %%
df["total_cases"] = df["workquantity"] / df["unit"].apply(lambda x: 12 if x == "Eaches" else 1)
print(f"New column created!")

# %% [markdown]
# ### Calculate the total number of eaches per order

# %%
total_eaches = df.loc[df["unit"] == "Each", "workquantity"].sum()

# %% [markdown]
# ### Calculate the total number of cases per order

# %%
total_cases = df.loc[df["unit"] == "Case", "workquantity"].sum() + df["total_cases"].sum()

# %% [markdown]
# ### Calculate the total number of PIs per order

# %%
total_pis = df.loc[df["unit"] == "Pallet", "workquantity"].sum()

# %% [markdown]
# ### Calculate the percentage of eaches in the total sales volume

# %%
percent_eaches = total_eaches / (total_eaches + total_cases + total_pis) * 100

# %% [markdown]
# ### Calculate the percentage of cases in the total sales volume

# %%
percent_cases = total_cases / (total_eaches + total_cases + total_pis) * 100

# %% [markdown]
# ### Calculate the percentage of Pallet in the total sales volume

# %%
percent_pis = total_pis / (total_eaches + total_cases + total_pis) * 100

# %% [markdown]
# ### Print the results

# %%
print("Percentage of Eaches: {:.2f}%".format(percent_eaches))
print("Percentage of Cases: {:.2f}%".format(percent_cases))
print("Percentage of Pallets: {:.2f}%".format(percent_pis))

# %% [markdown]
# **Interpretation**
# 
#     The percentage of Eaches, Cases, and Pallets in the total sales volume. Based on the results, you can
#     determine which unit to focus on increasing to optimize sales.
#     
#     In the given dataset it is ===> Cases.
#     
#     Note: 
#     
#     Pallet ==> does not exist in the UNIT
#     There are 3k+ missing values in the UNIT columns.

# %% [markdown]
# ## Analysis: 2 ==> Increase the quantity based on the total time spent

# %% [markdown]
# ### Aggregate the data by 'ItemNumber' and compute the total time spent
# 

# %%
# Convert the datetime columns to total seconds
df['CLOSEDWORK_seconds'] = (df['closedwork'] - df['closedwork'].min()).dt.total_seconds()
df['WORKINPROCESS_seconds'] = (df['workinprocess'] - df['workinprocess'].min()).dt.total_seconds()

# %%
# Perform the aggregation
agg_df = df.groupby('itemnumber').agg({'CLOSEDWORK_seconds': 'sum', 'WORKINPROCESS_seconds': 'sum', 'workquantity': 'sum'})

# %%
# Divide the sums of seconds by the number of seconds in an hour to convert to hours
agg_df['CLOSEDWORK_hours'] = agg_df['CLOSEDWORK_seconds'] / 3600
agg_df['WORKINPROCESS_hours'] = agg_df['WORKINPROCESS_seconds'] / 3600

# %%
# Drop the columns with seconds
agg_df = agg_df.drop(columns=['CLOSEDWORK_seconds', 'WORKINPROCESS_seconds'])

# Reset the index
agg_df = agg_df.reset_index()

# %% [markdown]
# ### Add the total time spent as a new feature

# %%
agg_df['total_time_spent'] = agg_df['CLOSEDWORK_hours'] + agg_df['WORKINPROCESS_hours']

# %% [markdown]
# ### Data-Splitting

# %%
# split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(agg_df[['total_time_spent']], agg_df['workquantity'], test_size=0.2)

# %%
## shape of training data
train_data.shape, train_target.shape

# %%
## shape of testing data
test_data.shape, test_target.shape

# %% [markdown]
# ## Training Machine learning Model

# %%
# train a linear regression model on the training data
reg = LinearRegression().fit(train_data, train_target)

# %% [markdown]
# ### Make predictions

# %%
predictions = reg.predict(test_data)

# %% [markdown]
# ### Evaluate the model using mean squared error

# %%
mse = mean_squared_error(test_target, predictions)

# %% [markdown]
# ### Compute the % increase in quantity needed based on total time-spent

# %%
quantity_increase = ((predictions - test_target) / test_target) * 100

# %%
print(f"Increase in quantity needed: \n")
work_quantity_to_be_increased = pd.DataFrame(quantity_increase, columns=['workquantity'])

# %%
indexs = work_quantity_to_be_increased.index.values.tolist()

# %% [markdown]
# ## Adding new column ==> % of quantity to be increased based on total time spent

# %%
new_data = df.iloc[indexs]

# %%
new_data['increase_percentage_quantity'] = work_quantity_to_be_increased['workquantity']

# %% [markdown]
# ## Check data  ==> Below only 60 values are printed, see the file below this cell...

# %%
new_data.head(60)

# %% [markdown]
# ## Saving data to observe the quantity to be increase by %

# %%
new_data.to_csv('Percentage_of_Quantity_to_be_increased.csv')
print(f"New data saved!")

# %% [markdown]
# **Interpretation**
# 
#     1): I have, aggregated the sales data by 'ItemNumber' to get the total sales for each item.
#     2): Computed the total time spent on each item in 'workINPROCESS' and 'CLOSEDWORK', and added it as a new
#         column.
#     3): I have spillted the data into two parts:
#         3.1): Training data ==> is used to train machine learning models/algorithms.
#         3.2): Testing data  ==> is used to test machine learning models/algorithms.
#     4): I have made predictions and the model/algorithm has shown me how much of quantity should be increased.
#     5): The quantity increased is added as a new column at the end of the dataset, and that dataset is saved 
#         as "Percentage_of_Quantity_to_be_increased.csv".
#     6): Now open the file and look each "Item" and "increase_percentage_quantity".

# %% [markdown]
# ## Analysis: 3 ==> Increase quantity % based on UNIT, WorkingProcess and ClosedWork

# %% [markdown]
# ## Now lets consider ==> Unit, workingprocess and closed work as important columns

# %%
# Convert the datetime columns to total seconds
df['CLOSEDWORK_seconds'] = (df['closedwork'] - df['closedwork'].min()).dt.total_seconds()
df['WORKINPROCESS_seconds'] = (df['workinprocess'] - df['workinprocess'].min()).dt.total_seconds()

# Perform the aggregation
agg_df = df.groupby(['itemnumber', 'unit']).agg({'CLOSEDWORK_seconds': 'sum', 'WORKINPROCESS_seconds': 'sum', 'workquantity': 'sum'})

# Divide the sums of seconds by the number of seconds in an hour to convert to hours
agg_df['CLOSEDWORK_hours'] = agg_df['CLOSEDWORK_seconds'] / 3600
agg_df['WORKINPROCESS_hours'] = agg_df['WORKINPROCESS_seconds'] / 3600

# Drop the columns with seconds
agg_df = agg_df.drop(columns=['CLOSEDWORK_seconds', 'WORKINPROCESS_seconds'])

# Reset the index
agg_df = agg_df.reset_index()

agg_df['total_time_spent'] = agg_df['CLOSEDWORK_hours'] + agg_df['WORKINPROCESS_hours']

# %% [markdown]
# ### Splitting dataset

# %%
# split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(agg_df[['total_time_spent']], agg_df['workquantity'], test_size=0.2)

# %%
## shape of training data
train_data.shape, train_target.shape

# %%
## shape of testing data
test_data.shape, test_target.shape

# %% [markdown]
# ### Training machine learning model

# %%
# train a linear regression model on the training data
reg = LinearRegression().fit(train_data, train_target)

# %% [markdown]
# ## Making predictions

# %%
predictions = reg.predict(test_data)

# %% [markdown]
# ### Evaluating performance of the model

# %%
mse = mean_squared_error(test_target, predictions)

# %% [markdown]
# ### Compute the % increase in the quantity

# %%
quantity_increase = ((predictions - test_target) / test_target) * 100

# %%
print(f"Increase in quantity needed: \n")
work_quantity_to_be_increased = pd.DataFrame(quantity_increase, columns=['workquantity'])

# %% [markdown]
# ### Adding new column ==> % of quantity to be increased based on  Unit, workingprocess and closed work

# %%
indexs = work_quantity_to_be_increased.index.values.tolist()

# %%
new_data = df.iloc[indexs]

# %%
new_data['increase_percentage_quantity'] = work_quantity_to_be_increased['workquantity']

# %% [markdown]
# ## Check data  ==> Below only 60 values are printed, see the file below this cell...

# %%
new_data.head(60)

# %% [markdown]
# ## Saving data to observe the quantity to be increase by %

# %%
new_data.to_csv('Percentage_of_Quantity_to_be_increased_with_unit.csv')
print(f"New data saved!")

# %% [markdown]
# **Interpretation**
# 
#     1): I have, aggregated the sales data by 'ItemNumber' & 'Unit' to get the total sales for each item.
#     2): Computed the total time spent on each item in 'workINPROCESS' and 'CLOSEDWORK', and added it as a new
#         column.
#     3): I have spillted the data into two parts:
#         3.1): Training data ==> is used to train machine learning models/algorithms.
#         3.2): Testing data  ==> is used to test machine learning models/algorithms.
#     4): I have made predictions and the model/algorithm has shown me how much of quantity should be increased.
#     5): The quantity increased is added as a new column at the end of the dataset, and that dataset is saved 
#         as "Percentage_of_Quantity_to_be_increased.csv".
#     6): Now open the file and look each "Item" and "increase_percentage_quantity".

# %% [markdown]
# ## Selecting the best columns and then training the machine learning model

# %% [markdown]
# <center>========================================</center>

# %% [markdown]
# ## Analysis 4: ==> Increase quantity % based on important columns selected by machine learning model

# %% [markdown]
# ## Why selecting the best column using machine leanring?
#     
#     To find out which columns are important to increase the % of quantity, machine learning models tries to
#     find the best correlation between columns.

# %% [markdown]
# ## Encoding the data

# %%
# initialize a list to store the encoded features
encoded_features = []

# loop through all the features in the dataframe
for feature in df.columns:
    # check if the feature is a categorical variable
    if df[feature].dtype == 'object':
        # encode the categorical variable using one-hot encoding
        encoded = pd.get_dummies(df[feature], prefix=feature)
        encoded_features.append(encoded)
    elif df[feature].dtype == 'datetime64[ns]' or df[feature].dtype == 'datetime64':
        df[feature+'_year'] = df[feature].dt.year
        df[feature+'_month'] = df[feature].dt.month
        df[feature+'_day'] = df[feature].dt.day
        
        df[feature+'_hour'] = df[feature].dt.hour
        df[feature+'_min'] = df[feature].dt.minute
        df[feature+'_sec'] = df[feature].dt.second
        
        
        encoded_features.append(df[feature+'_year'])
        encoded_features.append(df[feature+'_month'])
        encoded_features.append(df[feature+'_day'])
        
        encoded_features.append(df[feature+'_hour'])
        encoded_features.append(df[feature+'_min'])
        encoded_features.append(df[feature+'_sec'])
        

# concatenate the encoded features with the original data
df_encoded = pd.concat([df] + encoded_features, axis=1)

# drop the original categorical features
df_encoded = df_encoded.drop(df.columns[df.dtypes == 'object'], axis=1)

# %%
# now drop all the categorical columns

data = []


for col in df_encoded.columns:
    try:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype == 'datetime64[ns]' or df_encoded[col].dtype == 'datetime64':
            pass
        else:
            data.append(df_encoded[col])
    except:
        pass
print(f"Data is ready!")

# %%
# to dataframe

dataset = pd.DataFrame(data).T
print(f"Dataset if finally ready!")

# %% [markdown]
# ## Separating the columns

# %%
# separate the independent variables and target variable
X = dataset.drop('workquantity', axis=1)
y = df['workquantity']

# %% [markdown]
# ## Training the machine learning model

# %%
# fit the ExtraTreesClassifier to the data
model = ExtraTreesClassifier()
model.fit(X, y)

# %% [markdown]
# ## Ranking the columns

# %%
importance = model.feature_importances_

# %% [markdown]
# ## Create a dictionary to store the feature names and their importance scores

# %%
features = dict(zip(X.columns, importance))

# %% [markdown]
# ## Sort the features by their importance scores

# %%
sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)

# %% [markdown]
# ## Select the top columns that has direct effect on quantity increase

# %%
top_1000 = sorted_features[:1000]

important_columns = []
print("Top 1000 most important features:")
for feature in top_1000:
    print("-", feature[0])
    important_columns.append(feature[0])

# %% [markdown]
# ## Separating important columns

# %%
dataset[important_columns]

# %% [markdown]
# ## Splitting the dataset

# %%
X = dataset[important_columns]
y = dataset['workquantity']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%
# shape of data
X_train.shape, y_train.shape

# %%
# shape of data
X_test.shape, y_test.shape

# %% [markdown]
# ## Training Machine learning model

# %%
model = ExtraTreesRegressor()
model_clf = model.fit(X_train, y_train)
model_pred = model_clf.predict(X_test)

# %%
mse = mean_squared_error(y_test, model_pred)

# %%
print(f"Mean Squared Error of Model: ==> {mse}")

# %% [markdown]
# ## Compute the % increase in quantity based on 1000 columns selected by machine learning model
# 
# 
#     Q: Where did the 1000 columns come from?
#     A: These columns are generated by using machine learning model. And we have selected the best one. Based
#        on the high correlation.

# %%
preds = np.array(model_pred)
tests = np.array(y_test)
train = np.array(y_train)

# %%
quantity_increase = ((preds - tests) / tests) * 100

# %%
print(f"Increase in quantity needed: \n")
work_quantity_to_be_increased = pd.DataFrame(quantity_increase, columns=['workquantity'])

# %% [markdown]
# ### Adding new column ==> % of quantity to be increased

# %%
indexs = work_quantity_to_be_increased.index.values.tolist()

# %%
new_data = df.iloc[indexs]

# %%
new_data['increase_percentage_quantity'] = work_quantity_to_be_increased['workquantity']

# %% [markdown]
# ## checking
#     
#     Note: below is showing only 60 records, to see full records see the .xlsx file 
#     (The file will be generated using below code...)

# %%
new_data.head(60)

# %% [markdown]
# ## Saving data to observe the quantity to be increase by %

# %%
new_data.to_csv('Percentage_of_Quantity_to_be_increased_with_best_columns.csv')
print(f"New data saved!")


