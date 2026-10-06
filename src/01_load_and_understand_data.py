import pandas as pd

df = pd.read_csv('data/raw/house_price_raw_100.csv')

print(df.head())
print(df.shape)
print(df.columns)

# understand the data
print('=' * 50)
print('\nUnderstanding the data: \n')
print('=' * 50)

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())