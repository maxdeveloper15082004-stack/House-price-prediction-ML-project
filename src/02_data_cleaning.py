import pandas as pd

df = pd.read_csv("data/raw/house_price_raw_100.csv")

print('*' * 50)
print('\nCleaning the data: \n')
print('*' * 50)

df.isnull().sum()
df = df.dropna()
df.duplicated().sum()
df = df.drop_duplicates()
df.shape

print(df.head())

df.to_csv("data/cleaned/house_price_cleaned.csv", index=False)