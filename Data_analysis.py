import pandas as pd

data = pd.read_csv("/Users/matheolentz/Desktop/Day_ahead_price_history.csv", sep=";")

print(data.head())
print(data.columns)