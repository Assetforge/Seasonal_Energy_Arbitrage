import pandas as pd
import matplotlib.pyplot as plt

#-------------
# Parameters of the battery
#-------------
capacity = 4400 #in MWh
charge = 1 #in MW
life_cycle = 10
capex = 250  #EUR/kW
Wacc = 0.1
#-------------
# Load Data
#-------------

df = pd.read_csv("/Users/matheolentz/Desktop/Day_ahead_price_history.csv", sep=";", encoding="utf-8")
df["datetime"] = pd.to_datetime(df["Start date"], dayfirst=True, errors="coerce")

print(df.columns)

#Select German prices
data_DELU = df[["datetime", "Germany/Luxembourg [€/MWh] Original resolutions"]]
data_DELU = data_DELU.rename(columns={"Germany/Luxembourg [€/MWh] Original resolutions": "price_EUR_MWh"})
print(data_DELU)


# Plot time series
plt.figure(figsize=(15,5))
plt.plot(data_DELU.index, data_DELU["price_EUR_MWh"], color="blue", linewidth=1)

plt.title("Germany/Luxembourg Day-Ahead Electricity Prices")
plt.xlabel("Date")
plt.ylabel("Price [€/MWh]")
plt.grid(True)
plt.tight_layout()
plt.show()