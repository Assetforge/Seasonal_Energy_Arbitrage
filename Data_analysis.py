import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
CSV_FP = "/Users/matheolentz/Desktop/Day_ahead_price_history.csv"
PRICE_COL = "Germany/Luxembourg [€/MWh] Original resolutions"
DATE_COL = "Start date"
SEP = ";"

capacity_MWh = 4400.0
power_MW = 1.0
eff_batt = 0.95  # round-trip efficiency
start_year = 2025   # you can change if you want to simulate another year

# -------------------------
# Load & prepare daily prices
# -------------------------
df = pd.read_csv(CSV_FP, sep=SEP, encoding="utf-8")
df["datetime"] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime", PRICE_COL]).sort_values("datetime")

prices_daily = df.set_index("datetime")[PRICE_COL].astype(float).resample("D").mean().dropna()

# -------------------------
# Define charge/discharge windows
# -------------------------
charge_start = pd.Timestamp(f"{start_year}-03-01")
charge_end   = charge_start + pd.DateOffset(months=6)
discharge_start = charge_end - pd.DateOffset(years=1)
discharge_end   = charge_start -pd.Timedelta(days=1)

charge_prices = prices_daily.loc[charge_start:charge_end]
discharge_prices = prices_daily.loc[discharge_start:discharge_end]

# -------------------------
# Simulate strategy
# -------------------------
# Energy charged (MWh) = power * hours = capacity by design
energy_charged = capacity_MWh
avg_charge_price = charge_prices.mean()
avg_discharge_price = discharge_prices.mean()

# Revenues and costs
cost = energy_charged * avg_charge_price   # € paid to charge
revenue = energy_charged * eff_batt * avg_discharge_price  # € earned when discharging
profit = revenue - cost

print(f"Charging: {charge_start.date()} → {charge_end.date()} at avg {avg_charge_price:.2f} €/MWh")
print(f"Discharging: {discharge_start.date()} → {discharge_end.date()} at avg {avg_discharge_price:.2f} €/MWh")
print(f"Cycle cost: {cost:,.0f} € | revenue: {revenue:,.0f} € | profit: {profit:,.0f} €")

# -------------------------
# Plot price series + charge/discharge shading
# -------------------------
plt.figure(figsize=(14,6))
plt.plot(prices_daily.index, prices_daily.values, label="Daily Avg Price", color="blue")
plt.axvspan(charge_start, charge_end, color="green", alpha=0.2, label="Charge period")
plt.axvspan(discharge_start, discharge_end, color="red", alpha=0.2, label="Discharge period")
plt.title("Seasonal Arbitrage Strategy (Charge in Summer, Discharge in Winter)")
plt.xlabel("Date")
plt.ylabel("Price [€/MWh]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Plot cumulative PnL over the year
# -------------------------
pnl_series = pd.Series(0.0, index=prices_daily.index)
pnl_series.loc[charge_prices.index] = - (power_MW * 24) * charge_prices.values  # cost per day
pnl_series.loc[discharge_prices.index] = (power_MW * 24) * discharge_prices.values * eff_batt
cumulative_pnl = pnl_series.cumsum()

plt.figure(figsize=(14,5))
plt.plot(cumulative_pnl.index, cumulative_pnl.values, label="Cumulative PnL [€]", color="purple")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Cumulative PnL from Seasonal Strategy")
plt.xlabel("Date")
plt.ylabel("PnL [€]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
