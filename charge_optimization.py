# Save this as optimize_storage.py and run where your CSV is accessible.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right

# -------------------------
# PARAMETERS (edit as needed)
# -------------------------
CSV_FP = "/Users/matheolentz/Desktop/Day_ahead_price_history.csv"
PRICE_COL = "Germany/Luxembourg [€/MWh] Original resolutions"  # adjust if different
DATE_COL = "Start date"  # SMARD EPEX style
SEP = ";"  # CSV separator

capacity_MWh = 4400.0     # MWh
power_MW = 1.0            # MW (charge/discharge power)
eff_batt = 0.95           # round-trip efficiency (battery * electronics combined already)
life_cycle = 10           # max number of cycles (partial counts as full)
min_days_each_side = 30   # for candidate charge/discharge blocks min length (optional)
max_delay_days = 400      # how many days ahead discharge may start (prune search)
currency_multiplier = power_MW  # multiply per-MWh sums by power to get EUR

# -------------------------
# 1) Load & prepare hourly prices
# -------------------------
df = pd.read_csv(CSV_FP, sep=SEP, encoding="utf-8")
df["datetime"] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime", PRICE_COL]).sort_values("datetime")
# ensure we have a continuous hourly index (fill missing hours with NaN)
prices = df.set_index("datetime")[PRICE_COL].astype(float).resample("H").mean()

# -------------------------
# 2) compute H_hours (hours to fully charge/discharge)
# -------------------------
H_hours = int(np.ceil(capacity_MWh / power_MW))
print(f"Hours per full cycle (H): {H_hours}")

# -------------------------
# 3) build rolling block sums for H-hour blocks (charge/discharge blocks)
# -------------------------
# If H_hours > len(prices) then nothing to do
if H_hours >= len(prices):
    raise ValueError("H_hours >= available data length. Need more historical hours or reduce capacity/power.")

# compute rolling sums for contiguous H-hour blocks
rolling_sum = prices.rolling(window=H_hours, min_periods=H_hours).sum().dropna()
# rolling_sum index corresponds to the timestamp of the *end* of the H-hour block
# We'll convert to start times for clarity:
block_end_times = rolling_sum.index
block_start_times = block_end_times - pd.to_timedelta(H_hours - 1, unit="h")
block_sums = rolling_sum.values  # sums of prices over each H-hour block

# To speed lookups, turn block data into DataFrame
blocks_df = pd.DataFrame({
    "start": block_start_times,
    "end": block_end_times,
    "sum_price": block_sums
}).reset_index(drop=True)

# -------------------------
# 4) generate candidate cycles
# Each candidate is: charge block i (start_i..end_i) and discharge block j (start_j..end_j)
# with j.start >= i.end + min_gap (we will require no overlap)
# profit_EUR = currency_multiplier * (eff_batt * sum_price_discharge - sum_price_charge)
# Each cycle occupies time interval [start_i, end_j] for overlap constraints.
# -------------------------
candidates = []
n_blocks = len(blocks_df)

# precompute a mapping from timestamp to block index to ease searching
start_to_idx = {blocks_df.loc[idx, "start"]: idx for idx in blocks_df.index}

max_delay_hours = int(max_delay_days * 24)

for i in range(n_blocks):
    charge_start = blocks_df.loc[i, "start"]
    charge_end = blocks_df.loc[i, "end"]
    # earliest discharge block index j must have start > charge_end  (no overlap)
    # find first block with start strictly after charge_end
    j_min_time = charge_end + pd.Timedelta(hours=1)
    # find j_min index using binary search on blocks_df['start'] (monotonic)
    # create array of starts:
    # We'll rather loop j from i+1 upward but prune by max_delay
    for j in range(i + 1, n_blocks):
        discharge_start = blocks_df.loc[j, "start"]
        # enforce delay limit
        if (discharge_start - charge_end).total_seconds() / 3600.0 > max_delay_hours:
            break
        discharge_sum = blocks_df.loc[j, "sum_price"]
        charge_sum = blocks_df.loc[i, "sum_price"]
        profit_per_MWh = eff_batt * discharge_sum - charge_sum  # EUR per MWh-of-charged-energy
        profit_EUR = profit_per_MWh * currency_multiplier
        if profit_EUR <= 0:
            # optionally skip non-positive candidates to reduce number
            continue
        # cycle occupies interval from charge_start to discharge_end (blocks_df.loc[j,"end"])
        interval_start = charge_start
        interval_end = blocks_df.loc[j, "end"]
        candidates.append({
            "i_block": i,
            "j_block": j,
            "charge_start": charge_start,
            "charge_end": charge_end,
            "discharge_start": discharge_start,
            "discharge_end": interval_end,
            "charge_sum": charge_sum,
            "discharge_sum": discharge_sum,
            "profit_EUR": float(profit_EUR)
        })

# If no positive candidates found, we may still want to consider best negative? but usually none.
if len(candidates) == 0:
    print("No positive-profit candidate cycles found with current parameters.")
else:
    print(f"Generated {len(candidates)} candidate cycles (positive-profit filtered).")

# -------------------------
# 5) Weighted interval scheduling with limit K = life_cycle
# We convert candidates into intervals with start/end timestamps. We sort by end time.
# Then classic DP extension for selecting up to K intervals:
# DP[t][k] = max profit using first t intervals (sorted by end) with up to k intervals chosen.
# For efficiency, we use 1-D arrays per k loop.
# -------------------------
if len(candidates) > 0:
    cand_df = pd.DataFrame(candidates)
    # sort by interval end
    cand_df = cand_df.sort_values("discharge_end").reset_index(drop=True)
    m = len(cand_df)
    # build array of end times for binary search
    end_times = list(cand_df["discharge_end"])
    start_times = list(cand_df["charge_start"])
    profits = list(cand_df["profit_EUR"])

    # p[t] = index of the last interval that ends before the t-th interval starts (or -1)
    p = []
    for t in range(m):
        # find rightmost interval with end < start_times[t]
        s_t = start_times[t]
        # use bisect_right on end_times
        idx = bisect_right(end_times, s_t - pd.Timedelta(microseconds=1)) - 1
        p.append(idx)

    # DP table: (K+1) x (m+1) -> we will do iterative per k to save memory
    K = life_cycle
    # initialize
    # dp[k][t] = max profit using intervals up to t-1 with at most k intervals
    dp = np.zeros((K + 1, m + 1), dtype=float)
    take = np.zeros((K + 1, m + 1), dtype=bool)  # for reconstruction

    for k in range(1, K + 1):
        for t in range(1, m + 1):
            # option1: skip interval t-1
            opt1 = dp[k, t - 1]
            # option2: take interval t-1
            idx_prev = p[t - 1] + 1  # convert to dp index (p is 0-based, dp uses +1)
            opt2 = profits[t - 1] + dp[k - 1, idx_prev]
            if opt2 > opt1:
                dp[k, t] = opt2
                take[k, t] = True
            else:
                dp[k, t] = opt1

    best_total_profit = dp[K, m]
    print(f"Best total profit selecting up to {K} cycles: EUR {best_total_profit:,.2f}")

    # reconstruct chosen intervals (from dp)
    chosen = []
    k = K
    t = m
    while k > 0 and t > 0:
        if take[k, t]:
            # take interval t-1
            chosen_idx = t - 1
            chosen.append(cand_df.loc[chosen_idx].to_dict())
            # move to p[t-1]
            t = p[t - 1] + 1
            k -= 1
        else:
            t -= 1

    chosen = list(reversed(chosen))
    chosen_df = pd.DataFrame(chosen)
    print("Chosen cycles:")
    print(chosen_df[["charge_start", "charge_end", "discharge_start", "discharge_end", "profit_EUR"]])

    # -------------------------
    # 6) Build hourly cashflow & PnL time series
    # During charge blocks: cashflow = - price * power (you pay)
    # During discharge blocks: cashflow = + price * power * eff_batt (you receive less due to efficiency)
    # -------------------------
    hourly_cf = pd.Series(0.0, index=prices.index)
    for row in chosen:
        # charge block index i_block -> start..end inclusive (H hours)
        cs = row["charge_start"]
        ce = row["charge_end"]
        ds = row["discharge_start"]
        de = row["discharge_end"]
        # Charge hours: pay price * power
        hourly_cf.loc[cs:ce] -= prices.loc[cs:ce] * power_MW
        # Discharge hours: receive price * power * eff
        hourly_cf.loc[ds:de] += prices.loc[ds:de] * power_MW * eff_batt

    # cumulative PnL
    cumulative_pnl = hourly_cf.cumsum()

    # -------------------------
    # 7) PLOTS
    # -------------------------
    plt.figure(figsize=(14,6))
    plt.plot(prices.index, prices.values, label="Hourly Price [€/MWh]", alpha=0.6)
    # mark charge/discharge periods
    for _, r in chosen_df.iterrows():
        plt.axvspan(r["charge_start"], r["charge_end"], color="blue", alpha=0.15)
        plt.axvspan(r["discharge_start"], r["discharge_end"], color="orange", alpha=0.15)
    plt.title("Price series with chosen charge (blue) and discharge (orange) blocks")
    plt.xlabel("Date")
    plt.ylabel("Price [€/MWh]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # PnL plot
    plt.figure(figsize=(12,5))
    plt.plot(cumulative_pnl.index, cumulative_pnl.values, label="Cumulative PnL [EUR]")
    plt.title("Cumulative PnL from Selected Cycles")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL [EUR]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar chart of profits per selected cycle
    plt.figure(figsize=(10,4))
    plt.bar(range(len(chosen_df)), chosen_df["profit_EUR"].values)
    plt.xticks(range(len(chosen_df)),
               [f"{str(pd.to_datetime(x).date())}->{str(pd.to_datetime(y).date())}" for x, y in zip(chosen_df["charge_start"], chosen_df["discharge_end"])],
               rotation=45, ha="right")
    plt.ylabel("Profit per cycle [EUR]")
    plt.title("Profit of chosen cycles")
    plt.tight_layout()
    plt.show()
