import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned/house_price_cleaned.csv")

print('*' * 50)
print('\nExploratory Data Analysis: \n')
print('*' * 50)

print(df.head())

df["price"].hist()

plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution")
plt.show()

plt.scatter(df["area_sqft"], df["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
plt.show()