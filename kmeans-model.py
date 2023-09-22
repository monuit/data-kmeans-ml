# %% [markdown]
# ## 0): Libs
# 

# %%
import numpy as np
import pandas as pd

import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler


# %%
from IPython.core.display import display, HTML
display(HTML('<style> .container {width:100% !important} </style>'))

# %% [markdown]
# ## 1): Data Understanding
# 

# %%
fp_1 = "Product Hierarchy.xlsx"
fp_2 = "Sales Invoice 2022.csv"
fp_3 = "Working table sales data dump for year 2022.xlsx"

# %%
product_hierarchy = pd.read_excel(fp_1)
product_hierarchy.columns = product_hierarchy.iloc[0]

sales_invoice = pd.read_csv(fp_2, low_memory=False)
sales_data = pd.read_excel(fp_3)

# %%
product_hierarchy = product_hierarchy.iloc[1:]

# %% [markdown]
# ## 3): Data Wrangling
# 

# %% [markdown]
# <center> <strong>========================================[ All 3-Datasets ]===================================================</strong> </center>
# 

# %% [markdown]
# #### 3.1): Detailed Understanding of data
# 

# %%
## Total records in all 3 datasets

print(f"Total records in Product Hierarchy: ==> ", product_hierarchy.shape[0])
print(f"Total records in Sales Invoice: ==>", sales_invoice.shape[0] )
print(f"Total records in Sales Data: ==> ", sales_data.shape[0])

# %%
### Duplicate values in all 3-datasets

print(f"Total duplicated values in Product Hierarchy: ==> ", product_hierarchy.duplicated().sum())
print(f"Total duplicated values in Sales Invoice: ==> ", sales_invoice.duplicated().sum())
print(f"Total duplicated values in Sales Data: ==> ", sales_data.duplicated().sum())

# %%
## Checking missing values in all 3 datasets

## 1): Product Hierarchy
product_hierarchy.isnull().sum()

# %%
## 2): Sales invoice: 
sales_invoice.isnull().sum()

# %%
## 3): Sales data
sales_data.isnull().sum()

# %% [markdown]
# **Interpretaion**
# 
#     The datasets contains alot of missing values. These values will effect the analysis of the project.
# 

# %% [markdown]
# #### 3.2): Filtering
# 

# %%
#### Product Hierarchy, Sales Invoice, and Sales Data Filtering
def filtering(df):
    return df.rename(columns={col:col.lower().replace(" ", "_") for col in df.columns.values.tolist()})

# %%
product_hierarchy = filtering(product_hierarchy)
sales_invoice = filtering(sales_invoice)
sales_data = filtering(sales_data)

print(f"Filtering of all 3-dataset done!")

# %% [markdown]
# #### 3.3): Typecasting
# 

# %%
### 1): No Typecasting is required for "Product Hierarchy"

### 2): Sales-Data ==> Already Typecasted
sales_data.dtypes

# %%
### 3): Typecasting Sales Invoice

sales_invoice['deliverydate'] = pd.to_datetime(sales_invoice['deliverydate'])
sales_invoice['fulfilmentdate'] = pd.to_datetime(sales_invoice['fulfilmentdate'])
sales_invoice['rsd'] = pd.to_datetime(sales_invoice['rsd'])
sales_invoice['mab'] = pd.to_datetime(sales_invoice['mab'])

print(f"Type casting for Sales Invoice is Done!")

# %%
# check the typecasted features

sales_invoice[['deliverydate', 'fulfilmentdate', 'rsd', 'mab']].dtypes

# %% [markdown]
# #### 3.4): Transformation
# 
#       Due to enormous amount of missing values/poor quality data, we cannot transform the data.
# 

# %% [markdown]
# #### 3.5): Imputing and Handling Missing & Duplicated values
# 

# %%
## 1): Duplicated values
    ## Since the dataset does not have any duplicated values. We will leave it as it is.

# %%
## 2): Handling Missing values

product_hierarchy.fillna(0, inplace=True)
sales_invoice.fillna(0, inplace=True)
sales_data.fillna(0, inplace=True)

print(f"Missing Product Hierarchy done!")
print(f"Missing Sales Invoice done!")
print(f"Missing Sales Data done!")

# %%
## re-checking the missing values in all 3-dataset

if (sum(product_hierarchy.isnull().sum()) > 0) or (sum(sales_invoice.isnull().sum()) > 0) or (sum(sales_data.isnull().sum()) > 0):
    print(f"Datasets contains missing values.")
else:
    print(f"Datasets are ready to be used for analysis!")

# %% [markdown]
# ## 4): Analysis and finding hidden patterns
# 

# %% [markdown]
# <center> <strong>========================================[ Analysing Dataset ]===================================================</strong> </center>
# 

# %%
product_hierarchy['itemnumber'] = product_hierarchy['item_number']

# %% [markdown]
# #### 4.1): Combine both dataset
# 

# %%
comb_1 = product_hierarchy.merge(sales_data, on='itemnumber')

# %%
# get the common columns
common_columns = set(sales_invoice.columns) & set(comb_1.columns)

# convert the common columns set to a list
common_columns = list(common_columns)

# %%
common_columns

# %%
df1 = comb_1.iloc[:20000, :]
df2 = sales_invoice.iloc[:20000, :]

# %%
combined_data = df1.merge(df2, on=['itemnumber', 'warehouse', 'linenumber'])

# %%
# saving the combined dataset
combined_data.to_csv('Combined Data.csv')
print(f"Data saved successfully!")

# %%
# combined data sampled

combined_data.sample(frac=1).head(10)

# %% [markdown]
# #### 4.2): Descriptive stats
# 

# %%
combined_data.describe()

# %% [markdown]
# #### 4.3): Counting unique values in each columns¶
# 

# %%
# count the number of unique values in each categorical column
print(f"Unique values in each Columns:\n")
for col in combined_data.columns:
    if combined_data[col].dtype == 'object':
        print(f"{col}: ====> [{combined_data[col].nunique()}] unique values")

# %% [markdown]
# #### 4.4): Calculating the mean of "Brand"
# 

# %%
grouped_brand_mean = combined_data.groupby(by='brand').mean()
grouped_brand_mean

# %% [markdown]
# #### 4.5): Visualize all the available categories
# 
#     Note: Here '0' means, most of the values are missing.
# 

# %%
plt.figure(figsize=(20, 5))


sb.countplot(x='brand_category', data=combined_data, palette='rocket')
plt.title('Brand Categories')
plt.show()

# %% [markdown]
# #### 4.6): Top 5 most purchased brands
# 

# %%
plt.figure(figsize=(20, 5))

sb.countplot(x='brand', order=combined_data['brand'].value_counts().iloc[:5].index, data=combined_data, palette='rocket')
plt.title('Top 5 brands in demand')
plt.show()

# %% [markdown]
# #### 4.7): Available stock of brands based on it's categories
# 

# %%
brand_cat = combined_data.groupby(['brand'])['brand_category'].count()
brand_cat = pd.DataFrame(brand_cat)
brand_cat = brand_cat.reset_index()

# %%
plt.figure(figsize=(45, 5))
sb.barplot(x='brand', y='brand_category', palette='rocket', data=brand_cat)
plt.title('Available stock of brands based on it\'s categories')
plt.show()

# %% [markdown]
# #### 4.8): Most brands used by countries
# 

# %%
country_brand = combined_data.groupby(['country_of_origin'])['brand'].count()
country_brand = pd.DataFrame(country_brand)
country_brand = country_brand.reset_index()

# %%
plt.figure(figsize=(40, 5))
sb.barplot(x='country_of_origin', y='brand', palette='rocket', data=country_brand)
plt.title('Most brands used by countries')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# #### 4.9): Countries purchasing total number of sub-brands of each brand
# 

# %%
country_subbrand = combined_data.groupby(['country_of_origin', 'brand'])['sub_brand'].count()
country_subbrand = pd.DataFrame(country_subbrand)
country_subbrand = country_subbrand.reset_index()

# display results
country_subbrand

# %%
plt.figure(figsize=(40, 5))
sb.barplot(x='country_of_origin', y='sub_brand', palette='rocket', data=country_subbrand)
plt.title('Most sub-brands used by countries')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# #### 5.0): Finding the correlation among columns
# 

# %%
cd = pd.get_dummies(combined_data, columns=["brand_category", "brand", "sub_brand", "category", "format", "status", "company", "country", "flavor", "product_group", "product_class", "milk_type", "country_of_origin"])

# %%
numerical_features = ["shelf_life_period_in_days", "product_width", "product_length", "product_height", "net_weight", "gross_weight", "tare_weight", "case_width", "case_length", "case_height", "case_weight", "pallet_width", "pallet_length", "pallet_height", "pallet_weight", "density_(lbs_/_cubic_ft)", "shipping_unit_gw"]
cd[numerical_features] = (cd[numerical_features] - cd[numerical_features].mean()) / cd[numerical_features].std()

# %%
corr = cd[numerical_features].fillna(0).corr()

# %%
# visualizing the correlation
plt.figure(figsize=(12, 8))

matrix = np.triu(corr)
sb.heatmap(corr, mask=matrix)

plt.show()

# %% [markdown]
# #### 5.1): Visualize the data using pairplot to identify the relationships between variables:
# 

# %%
sb.pairplot(cd[numerical_features])
plt.show()

# %% [markdown]
# #### 5.2): Product sales over the year
# 

# %%
unique_product_sale = combined_data.groupby(['invoicedate'])['uniqueproduct'].count()
unique_product_sale = pd.DataFrame(unique_product_sale)


# visualize
plt.figure(figsize=(20, 7))

plt.plot(unique_product_sale)
plt.title('Unique product sale over the year')

plt.show()

# %% [markdown]
# **Interpretation**
# 
#     The graph suggests that the product sold have a rough distribution, suggesting that the growth over the year is very less. More specifically the growth is "unpredictable" due to the poor
#     quality data.
# 

# %% [markdown]
# #### 5.3): Most sales over the month
# 

# %%
most_sales_over_month = combined_data.groupby(['invoicemonth'])['invoicenumber'].count()
most_sales_over_month = pd.DataFrame(most_sales_over_month)

# visualize 
plt.figure(figsize=(20, 7))

plt.plot(most_sales_over_month)
plt.title('Most sales over the month')

plt.show()

# %%
plt.figure(figsize=(20, 7))

sb.barplot(x=most_sales_over_month.index.values.tolist(), y=most_sales_over_month['invoicenumber'], data=most_sales_over_month)
plt.title('Most sales over the month')

plt.show()

# %% [markdown]
# **Interpretaions**
# 
#     The graph suggest that most sales occur in "Auguts" & "October".
# 

# %% [markdown]
# #### 5.4): Warehouse that makes the most sales
# 

# %%
most_sales_warhouse = combined_data.groupby(['warehouse'])['invoicenumber'].count()
most_sales_warhouse = pd.DataFrame(most_sales_warhouse)

# visualize
plt.figure(figsize=(30, 5))

sb.barplot(x=most_sales_warhouse.index.values.tolist(), y=most_sales_warhouse['invoicenumber'], data=most_sales_warhouse)
plt.title('Warehouse most sales')

plt.show()

# %% [markdown]
# #### 5.5): Aggregation: Calculate the sum of USDExtdCost by ShipMethod
# 

# %%
agg_df = combined_data.groupby(['shipmethod'])['usdextdcost'].sum().reset_index()

agg_df = agg_df.sort_values(by='usdextdcost', ascending=False).tail(10)

# visuzlize
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(20, 20))
plt.pie(agg_df['usdextdcost'], labels=agg_df['shipmethod'], autopct='%1.1f%%')
plt.axis('equal')
plt.title('Sum of USDExtdCost by ShipMethod')

plt.show()

# %% [markdown]
# #### 5.6): Plot a histogram to show the distribution of actual time
# 

# %%
plt.figure(figsize=(10, 7))
plt.hist(combined_data['actualtimemain'], bins=20)
plt.xlabel('Actual Time')
plt.ylabel('Frequency')
plt.title('Histogram of Actual Time')
plt.show()

# %% [markdown]
# #### 5.7): Plot a bar chart to show the number of work order types¶
# 

# %%
plt.figure(figsize=(10, 7))

plt.bar(combined_data['workordertypemain'].value_counts().index, combined_data['workordertypemain'].value_counts().values)
plt.xlabel('Work Order Type')
plt.ylabel('Count')
plt.title('Bar chart of Work Order Types')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# #### 5.8): Finding hidden patterns
# 

# %%
# Clustering: Group similar products together using k-means clustering
# First, we need to scale the data
X = cd[numerical_features] # independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))


# %%
# Then, we perform k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Add the cluster labels to the original data
combined_data['cluster'] = kmeans.labels_

# %%
## PCA by reducing the dimensionality

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
pca_result = pd.DataFrame(pca_result, columns=['component1', 'component2'])
pca_result['cluster'] = combined_data['cluster']

# %%
# Plot the first two principal components
plt.scatter(pca_result.iloc[:, 0], pca_result.iloc[:, 1], c=pca_result['cluster'])
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# %% [markdown]
# **Interpretation**
# 
#     Kmeans is a type of unsupervised machine learning technique used for clustering. It partitions a set of data points into K number of clusters based on their similarity. In this case, the Kmeans algorithm has found 3 clusters, meaning that it has grouped the data points into 3 distinct groups based on their similarities. These groups are differentiated based on the features of data.
# 
# The visualization show perfect clustering of entire data. The datapoints outside the cluster show outliers or anomalies in the data.
# 


