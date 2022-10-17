import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import black_scholes

# Inputs
spot = 3375.46
spot_date = dt.datetime(day=7, month=10, year=2022)
min_volume = 1
nb_options = []
nb_options_text = []

# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.expand_frame_repr', False)

# Load Options Data
df_list = []
folder = 'datas'
for file in sorted((f for f in os.listdir(folder) if not f.startswith(".")), key=str.lower):
    # Do not open hidden files
    if file[0] != "~":
        df_list.append(pd.read_excel(f"{folder}/{file}", header=0, engine='openpyxl').dropna())
df = pd.concat(df_list)

# Set Spot Value & Date
df["Spot"] = spot
df["Spot Date"] = spot_date

# Remove Low Volume Options
df = df[df["Volm"] >= min_volume].copy()
df = df.reset_index(drop=True)

# Parse The Option's Type, Strike Percentage, Underlying & Maturity
df["Type"] = df["Ticker"].apply(lambda x: "Call" if "C" in x.split(" ")[2] else "Put")
df["Underlying"] = df["Ticker"].apply(lambda x: x.split(" ")[0])
df["Underlying"] = df["Underlying"].apply(lambda x: x[1:-1] if x[0] == "W" else x)
df["Maturity"] = df["Ticker"].apply(lambda x: dt.datetime.strptime(x.split(" ")[1], "%m/%d/%y"))
df["Strike Perc"] = df["Strike"] / df["Spot"]

# Compute Mid Price & Maturity (in years)
df["Mid"] = (df["Bid"] + df["Ask"]) / 2
df["Maturity (in Y)"] = df.apply(lambda x: (x["Maturity"] - x["Spot Date"]).days / 365, axis=1)

# Dropping Useless Columns
df = df[["Type", "Underlying", "Spot", "Spot Date", "Maturity", "Maturity (in Y)", "Strike", "Strike Perc", "Mid", "Volm", "IVM"]]

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Initial")

# Data Coherence Verification
print(f"\n1) Coherence Check :")
nb_arbitrage = 1
while nb_arbitrage > 0:
    nb_arbitrage = 0
    index_list = []
    # No Butterfly Arbitrage
    for maturity in df["Maturity"].unique():
        for type in df["Type"].unique():
            df_check = df[(df["Maturity"] == maturity) & (df["Type"] == type)].copy()
            df_check = df_check.sort_values(by="Strike", ascending=True)
            if len(df_check.index) >= 3:
                if type == "Call":
                    df_check["Butterfly"] = df_check["Mid"] - df_check["Mid"].shift(-1) * \
                                            ((df_check["Strike"].shift(-1) - df_check["Strike"]) /
                                             (df_check["Strike"].shift(-2) - df_check["Strike"].shift(-1)) + 1) + \
                                            df_check["Mid"].shift(-2) * (
                                                    (df_check["Strike"].shift(-1) - df_check["Strike"]) /
                                                    (df_check["Strike"].shift(-2) - df_check["Strike"].shift(-1)))
                    id_with_arbitrage = list(df_check[df_check["Butterfly"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[
                            [row_id, min(df_check.index[row_id_pos + 1], df_check.index[-1]), min(df_check.index[row_id_pos + 2], df_check.index[-1])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                else:
                    df_check["Butterfly"] = df_check["Mid"] - df_check["Mid"].shift(1) * \
                                            (1 + (df_check["Strike"].shift(1) - df_check["Strike"]) / (
                                                    df_check["Strike"].shift(2) - df_check["Strike"].shift(1))) + \
                                            df_check["Mid"].shift(2) * ((
                                                        df_check["Strike"].shift(1) - df_check["Strike"]) / (
                                                                                df_check["Strike"].shift(2) -
                                                                                df_check["Strike"].shift(1)))
                    id_with_arbitrage = list(df_check[df_check["Butterfly"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[[row_id, max(df_check.index[row_id_pos - 1], df_check.index[0]), max(df_check.index[row_id_pos - 2], df_check.index[0])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                index_list = list(set(index_list + id_to_remove))
    # No Calendar Spread Arbitrage
    for type in df["Type"].unique():
        for strike in df["Strike"].unique():
            df_check = df[(df["Strike"] == strike) & (df["Type"] == type)].copy()
            df_check = df_check.sort_values(by="Maturity", ascending=True)
            if len(df["Maturity"].unique()) >= 2:
                df_check["Calendar Spread"] = df_check["Mid"].diff()
                id_with_arbitrage = list(df_check[df_check["Calendar Spread"] <= 0].index)
                id_to_remove = []
                for row_id in id_with_arbitrage:
                    row_id_pos = list(df_check.index).index(row_id)
                    df_select = df_check.loc[[row_id, max(df_check.index[row_id_pos - 1], df_check.index[0])]]
                    id_to_remove.append(df_select[['Volm']].idxmin()[0])
                index_list = list(set(index_list + id_to_remove))
    # Remove Less Liquid Options Causing An Arbitrage
    nb_arbitrage = len(index_list)
    if nb_arbitrage > 0:
        df = df.drop(index_list).reset_index(drop=True)
print(f"  - Butterfly Arbitrage : OK")
print(f"  - Calendar Spread Arbitrage : OK")
print(f"  - Nb of options removed : {nb_options[-1] - len(df.index)}")

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Arbitrages")

# Retrieve Forward & ZC (per maturity)
df = df.sort_values(by="Maturity", ascending=True)
print("\n2) Computing Forward & ZC (per maturity) :")
for maturity in df["Maturity"].unique():
    df_reg = df[df["Maturity"] == maturity].copy()
    # Remove Strikes With less than 1 Calls & Puts
    for strike in df_reg["Strike"].unique():
        if len(df_reg[(df_reg["Strike"] == strike)].index) == 1:
            df_reg = df_reg[df_reg["Strike"] != strike].copy()
    # Remove Strikes With less than 2 Calls & Puts (no regression possible)
    if len(df_reg.index) < 4:
        df = df[df["Maturity"] != maturity].copy()
    # Else --> Compute ZC & Forward
    else:
        Y_list = []
        K_list = df_reg["Strike"].unique()
        for strike in K_list:
            Y_list.append(float(df_reg[(df_reg["Strike"] == strike) & (df_reg["Type"] == "Call")]["Mid"]) - float(df_reg[(df_reg["Strike"] == strike) & (df_reg["Type"] == "Put")]["Mid"]))
        x = np.array(np.array(K_list)/spot).reshape((-1, 1))
        y = np.array(np.array(Y_list)/spot)
        model = LinearRegression().fit(x, y)
        beta = model.coef_[0]
        alpha = model.intercept_
        zc = -beta
        forward = alpha/zc
        df.loc[df["Maturity"] == maturity, ['Forward', 'ZC']] = forward, zc
print(f" - Nb of options removed : {nb_options[-1] - len(df.index)}")

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Regression")

# Remove ITM Options
print("\n3) Removing ITM Options :")
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()
print(f" - Nb of options removed : {nb_options[-1] - len(df.index)}")

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("ITM Options")

# Compute Implied Volatilities
print(f"\n4) Computing BS Implied Volatilities :")
df["Implied Vol"] = df.apply(
    lambda x: black_scholes.BS_ImpliedVol(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                          MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0]), axis=1)
df = df[df["Implied Vol"] != -1].copy()
print(f" - Nb of options removed : {nb_options[-1] - len(df.index)}")

print(f"\nNb of points used in vol surface : {len(df.index)}")

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Newton-Raphson")

# Compute Implied Total Variance
df["Implied Total Var"] = df["Implied Vol"] * df["Implied Vol"] * df["Maturity (in Y)"]

# Plot Calibration Infos
forward_list = []
zc_list = []
# Plot Implied Forward & ZC
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    forward_list.append(df_bis["Forward"].unique())
    zc_list.append(df_bis["ZC"].unique())
plt.plot(df["Maturity"].unique(), forward_list, label="Forward")
plt.plot(df["Maturity"].unique(), zc_list, label="ZC")
plt.title(f"Implied Forward & ZC")
plt.legend()
plt.grid()
plt.show()
# Plot Implied Total Variance
percentage = 70
for strike in df["Strike"].unique():
    df_strike = df[(df["Strike"] == strike)].copy()
    if len(df_strike["Maturity"].unique()) >= percentage / 100 * len(df["Maturity"].unique()):
        total_implied_var = []
        for maturity in df_strike["Maturity"].unique():
            df_bis = df[(df["Strike"] == strike) & (df["Maturity"] == maturity)].copy()
            total_implied_var.append(df_bis["Implied Total Var"].unique())
        plt.plot(df_strike["Maturity"].unique(), total_implied_var, label=strike)
plt.grid()
plt.legend()
plt.title("Implied Total Variance (by most liquid strikes)")
plt.show()
# Plot Number of Options Used
plt.plot(nb_options_text, nb_options, "-")
plt.title("Number of options used")
plt.grid()
plt.show()

# Calibration (Nelder-Mead)
