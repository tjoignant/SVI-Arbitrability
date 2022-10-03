import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import black_scholes


# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.expand_frame_repr', False)

# Load Options Data
df_calls = pd.read_excel("datas/SX5E_Calls_Dec22.xlsx", header=0).dropna()
df_puts = pd.read_excel("datas/SX5E_Puts_Dec22.xlsx", header=0).dropna()
df = pd.concat([df_calls, df_puts])

# Set Spot Value & Date
df["Spot"] = 3318.20
df["Spot Date"] = dt.datetime(day=30, month=9, year=2022)

# Remove Low Volume Options
df = df[df["Volm"] > 500].copy()
df = df.reset_index(drop=True)

# Parse The Option's Type, Strike Percentage, Underlying & Maturity
df["Type"] = df["Ticker"].apply(lambda x: "Call" if "C" in x.split(" ")[2] else "Put")
df["Underlying"] = df["Ticker"].apply(lambda x: x.split(" ")[0])
df["Maturity"] = df["Ticker"].apply(lambda x: dt.datetime.strptime(x.split(" ")[1], "%m/%d/%y"))
df["Strike Perc"] = df["Strike"]/df["Spot"]

# Compute Mid Price
df["Mid"] = (df["Bid"] + df["Ask"]) / 2

# Dropping Useless Columns
df = df[["Type", "Underlying", "Spot", "Spot Date", "Maturity", "Strike", "Strike Perc", "Mid"]]

# Set Check Dataframes
df_calls_check = df[df["Type"] == "Call"].copy()
df_puts_check = df[df["Type"] == "Put"].copy()

# Coherence Check (1) : Call/Put Spreads > 0
df_calls_check["Spread"] = df_calls_check["Mid"] - df_calls_check["Mid"].shift(-1)
df_puts_check["Spread"] = df_puts_check["Mid"] - df_puts_check["Mid"].shift(1)
nb_call_spread_neg = len(list(filter(lambda x: (x <= 0), df_calls_check["Spread"])))
nb_put_spread_neg = len(list(filter(lambda x: (x <= 0), df_puts_check["Spread"])))
if nb_call_spread_neg > 0:
    print("Call Spreads Check Failed")
else:
    print("Call Spreads Check Passed")
if nb_put_spread_neg > 0:
    print("Put Spreads Check Failed")
else:
    print("Put Spreads Check Passed")

# Coherence Check (2) : Call/Put Butterflies > 0
df_calls_check["Butterfly"] = df_calls_check["Mid"] - df_calls_check["Mid"].shift(-1) * ((df_calls_check["Strike"].shift(-1) - df_calls_check["Strike"]) / (df_calls_check["Strike"].shift(-2) - df_calls_check["Strike"].shift(-1))+1) + df_calls_check["Mid"].shift(-2) * ((df_calls_check["Strike"].shift(-1) - df_calls_check["Strike"]) / (df_calls_check["Strike"].shift(-2) - df_calls_check["Strike"].shift(-1))+0)
df_puts_check["Butterfly"] = df_puts_check["Mid"] - df_puts_check["Mid"].shift(1) * (1+(df_puts_check["Strike"].shift(1) - df_puts_check["Strike"])/(df_puts_check["Strike"].shift(2) - df_puts_check["Strike"].shift(1))) + df_puts_check["Mid"].shift(2) * (0+(df_puts_check["Strike"].shift(1) - df_puts_check["Strike"])/(df_puts_check["Strike"].shift(2) - df_puts_check["Strike"].shift(1)))
nb_call_butterfly_neg = len(list(filter(lambda x: (x <= 0), df_calls_check["Butterfly"])))
nb_put_butterfly_neg = len(list(filter(lambda x: (x <= 0), df_puts_check["Butterfly"])))
if nb_call_butterfly_neg > 0:
    print("Call Butterflies Check Failed")
else:
    print("Call Butterflies Check Passed")
if nb_put_butterfly_neg > 0:
    print("Put Butterflies Check Failed")
else:
    print("Put Butterflies Check Passed")

# Convert Mid To ATF Mid
df["ATF Mid"] = df["Mid"]

# Compute Implied Vol
df["Implied Vol"] = df.apply(lambda x: black_scholes.BS_ImpliedVol(f=1, k=x["Strike Perc"], t=((x["Maturity"]-x["Spot Date"]).days)/365, MktPrice=x["ATF Mid"]/x["Spot"], df=1, OptType=x["Type"][0]), axis=1)
df = df[df["Implied Vol"] != -1].copy()
print(df)

# Calibration

# Plot Vol Smile
df_calls = df[df["Type"] == "Call"].copy()
df_puts = df[df["Type"] == "Put"].copy()
plt.plot(df_calls["Strike Perc"], df_calls["Implied Vol"], 'o', color='orange', label="Calls")
plt.plot(df_puts["Strike Perc"], df_puts["Implied Vol"], 'o', color='blue', label="Puts")
plt.title("SX5E Dec22 - Vol Smile")
plt.legend()
plt.grid()
plt.show()
