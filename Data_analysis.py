import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/matheolentz/Desktop/Day_ahead_price_history.csv", sep=";", encoding="utf-8")
df["datetime"] = pd.to_datetime(df["Start date"], dayfirst=True, errors="coerce")

#print(df.head())
print(df.columns)

data_DELU = df[["datetime", "Germany/Luxembourg [€/MWh] Original resolutions"]]
data_DELU = data_DELU.rename(columns={"Germany/Luxembourg [€/MWh] Original resolutions": "price_EUR_MWh"})
print(data_DELU)


# Plot time series
plt.figure(figsize=(15,5))
plt.plot(data.index, data["price_EUR_MWh"], color="blue", linewidth=1)

plt.title("Germany/Luxembourg Day-Ahead Electricity Prices")
plt.xlabel("Date")
plt.ylabel("Price [€/MWh]")
plt.grid(True)
plt.tight_layout()
plt.show()