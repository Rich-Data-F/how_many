#!/usr/bin/env python3

import pandas as pd
from sqlalchemy import create_engine

# Read data from first point of sale
file_path = "../Data/rough_data/7_TCD_Intra_9Jan2025_format.xlsx"
sales_units_prices = pd.read_excel(file_path, sheet_name=0)
print(sales_units_prices.head())
print(sales_units_prices.shape)

# Get unique article count
print(sales_units_prices['Article_name'].nunique())

# Group and process first point of sale data
sales_units_prices_pos1 = sales_units_prices.groupby(['Article_name', 'Units_Price','Point_of_Sale']).sum().reset_index()
print(sales_units_prices_pos1['Article_name'].nunique())
print(sales_units_prices_pos1.shape)

sales_units_pos1 = sales_units_prices[sales_units_prices['Units_Price']=='Units']
print(sales_units_pos1)

sales_prices_pos1 = sales_units_prices[sales_units_prices['Units_Price']=='Price']
print(sales_prices_pos1)

intra_intra_rows = sales_units_prices_pos1[sales_units_prices_pos1['Point_of_Sale'] == 'IntraIntra']
print(intra_intra_rows)

# Read data from second point of sale
file_path = "../Data/rough_data/7_TCD_Fontenelles_9Jan2025_format.xlsx"
sales_units_prices = pd.read_excel(file_path, sheet_name=0)
print(sales_units_prices.head())
print(sales_units_prices.shape)

sales_units_prices_pos2 = sales_units_prices.groupby(['Article_name', 'Units_Price','Point_of_Sale']).sum().reset_index()
print(sales_units_prices_pos2['Article_name'].nunique())
print(sales_units_prices_pos2.shape)
print(sales_units_prices_pos2)

sales_units_pos2 = sales_units_prices_pos2[sales_units_prices_pos2['Units_Price']=='Units']
sales_prices_pos2 = sales_units_prices_pos2[sales_units_prices_pos2['Units_Price']=='Price']
print(sales_prices_pos2)
print(sales_units_pos2)

long_sales_units_pos1 = sales_units_pos1.melt(
    id_vars=["Point_of_Sale", "Article_name", "Units_Price"],
    var_name="Date",
    value_name="Quantity"
)
print(long_sales_units_pos1)

long_sales_units_pos2 = sales_units_pos2.melt(
    id_vars=["Point_of_Sale", "Article_name", "Units_Price"],
    var_name="Date",
    value_name="Quantity"
)
print(long_sales_units_pos2)

long_sales_prices_pos1 = sales_prices_pos1.melt(
    id_vars=["Point_of_Sale", "Article_name", "Units_Price"],
    var_name="Date",
    value_name="Price"
)
print(long_sales_prices_pos1)

long_sales_prices_pos2 = sales_prices_pos2.melt(
    id_vars=["Point_of_Sale", "Article_name", "Units_Price"],
    var_name="Date",
    value_name="Price"
)
print(long_sales_prices_pos2)

# Replace NaN values with 0 in all dataframes
long_sales_units_pos1['Quantity'] = long_sales_units_pos1['Quantity'].fillna(0)
long_sales_units_pos2['Quantity'] = long_sales_units_pos2['Quantity'].fillna(0)
long_sales_prices_pos1['Price'] = long_sales_prices_pos1['Price'].fillna(0)
long_sales_prices_pos2['Price'] = long_sales_prices_pos2['Price'].fillna(0)

def verify_dates(dataframe):
    # Get min and max dates
    min_date = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y').min()
    max_date = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y').max()
    # Calculate number of days between
    days_between = (max_date - min_date).days
    # Get unique dates
    unique_dates = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y').unique()
    print(f"Minimum date: {min_date.strftime('%d/%m/%Y')}")
    print(f"Maximum date: {max_date.strftime('%d/%m/%Y')}")
    print(f"Number of days between: {days_between}")
    print("\nUnique dates:")
    print(len(unique_dates))


verify_dates(long_sales_units_pos1)
verify_dates(long_sales_units_pos2)
verify_dates(long_sales_prices_pos1)
verify_dates(long_sales_prices_pos2)

all_pos_sales = pd.concat([long_sales_prices_pos1, long_sales_prices_pos2, long_sales_units_pos1, long_sales_units_pos2], ignore_index=True)
all_pos_sales.shape
print(all_pos_sales)

all_pos_sales_grouped = all_pos_sales.groupby(['Article_name', 'Point_of_Sale', 'Date']).sum().reset_index()
print(all_pos_sales_grouped)
# Verify if we have either price without quantity or vice versa in any rows
def missing_values_check(dataframe):
    missing_values = dataframe[(dataframe['Quantity'].isnull() & dataframe['Price'].notnull()) | (dataframe['Quantity'].notnull() & dataframe['Price'].isnull())]
    print(missing_values)

missing_values_check(all_pos_sales_grouped)
verify_dates(all_pos_sales_grouped)

all_pos_sales_merged=all_pos_sales_grouped.groupby(['Article_name','Date']).sum().reset_index()

all_pos_sales_merged=all_pos_sales_grouped.drop(columns=['Units_Price','Point_of_Sale'])
all_pos_sales_merged=all_pos_sales_merged.groupby(['Article_name','Date']).sum().reset_index()

print(all_pos_sales_merged)
verify_dates(all_pos_sales_merged)
missing_values_check(all_pos_sales_merged)

df=pd.DataFrame(all_pos_sales_merged['Article_name'].unique())
df.to_excel('../Data/to_explore/unique_article_names.xlsx', index=False, engine='openpyxl')

# Load the mapping file
mapping = pd.read_excel('../Data/to_explore/unique_article_names_for_mapping.xlsx')
#mapping = pd.read_excel('https://docs.google.com/spreadsheets/d/1BXbPeI7iBKj53NUuYGyn_gCDDNw4rzne/edit?usp=sharing&ouid=113287554147677931550&rtpof=true&sd=true',\
#    engine='openpyxl')

# Merge the mapping with the all_pos_sales_units_prices dataframe
historical_sales_units_prices = all_pos_sales_merged.merge(mapping, left_on='Article_name',\
    right_on='Initial_article_name', how='left')

# Replace the Article_name with the renamed_article
historical_sales_units_prices['Article_name'] = historical_sales_units_prices['Renamed_article']

# Drop the renamed_article column
historical_sales_units_prices = historical_sales_units_prices.drop(columns=['Renamed_article', 'Initial_article_name', 'Questions', 'Réponses'])

historical_sales_units_prices['Article_name'].nunique()

historical_sales_units_prices

print(historical_sales_units_prices.shape)
# Convert Date column to datetime format
historical_sales_units_prices["Date"] = pd.to_datetime(historical_sales_units_prices["Date"], format="%d/%m/%Y")

# Fill missing values with zeros
historical_sales_units_prices["Quantity"] = historical_sales_units_prices["Quantity"].fillna(0).astype(int)
historical_sales_units_prices["Price"] = historical_sales_units_prices["Price"].fillna(0).astype(float)
missing_values_check(historical_sales_units_prices)

import matplotlib as plt

# Load the categorised articles data
categorised_articles = pd.read_excel("../Data/transformed_data/categorised_articles.xlsx")
# Merge the subset summary with categorised articles to add the category
historical_sales_unit_price_with_category = historical_sales_units_prices.merge(categorised_articles, on='Article_name', how='left')

# print the updated subset summary with category
print(historical_sales_unit_price_with_category)
missing_values_check(historical_sales_unit_price_with_category)

# Load the categorised articles data
categorised_articles = pd.read_excel("../Data/transformed_data/categorised_articles.xlsx")
# Merge the subset summary with categorised articles to add the category
all_pos_sales_merged_with_categories = all_pos_sales_merged.merge(categorised_articles, on='Article_name', how='left')
all_pos_sales_merged_with_categories=all_pos_sales_merged_with_categories.drop(columns='Unnamed: 4')
# print the updated subset summary with category
print(all_pos_sales_merged_with_categories)

missing_values_check(all_pos_sales_merged_with_categories)

historical_sales_units_prices['Date'] = pd.to_datetime(historical_sales_units_prices['Date'], format='%d/%m/%Y').dt.strftime('%d/%m/%Y')
historical_sales_units_prices.to_excel('../Data/transformed_data/historical_sales_data.xlsx', index=False, engine='openpyxl')
historical_sales_units_prices.to_csv('../Data/transformed_data/historical_sales_data.csv', index=False)


