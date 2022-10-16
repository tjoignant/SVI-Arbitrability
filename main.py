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

# Compute Mid Price
df["Mid"] = (df["Bid"] + df["Ask"]) / 2

# Dropping Useless Columns
df = df[["Type", "Underlying", "Spot", "Spot Date", "Maturity", "Strike", "Strike Perc", "Mid", "Volm", "IVM"]]

# Data Coherence Verification
for udl in df["Underlying"].unique():
    print(f"\nCoherence Check for {udl} :")
    options_removed = 0
    nb_arbitrage = 1
    while nb_arbitrage > 0:
        nb_arbitrage = 0
        index_list = []
        # No Butterfly Arbitrage
        for maturity in df["Maturity"].unique():
            for type in df["Type"].unique():
                df_check = df[(df["Underlying"] == udl) & (df["Maturity"] == maturity) & (df["Type"] == type)].copy()
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
                df_check = df[(df["Underlying"] == udl) & (df["Strike"] == strike) & (df["Type"] == type)].copy()
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
        # Remove Options
        nb_arbitrage = len(index_list)
        if nb_arbitrage > 0:
            df = df.drop(index_list).reset_index(drop=True)
            options_removed = options_removed + nb_arbitrage
    print(f"  - Butterfly Arbitrage : OK")
    print(f"  - Calendar Spread Arbitrage : OK")
    print(f"  - Nb of options removed : {options_removed}")

nb_options = len(df.index)

# Retrieve Forward & ZC (per maturity)
df = df.sort_values(by="Maturity", ascending=True)
print("\nComputing Forward & ZC per maturity :")
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
print(f" - Nb of options removed : {nb_options - len(df.index)}")
nb_options = len(df.index)

# Remove ITM Options
print("\nRemoving ITM Options :")
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()
print(f" - Nb of options removed : {nb_options - len(df.index)}")
nb_options = len(df.index)

# Compute Implied Volatilities
print(f"\nComputing BS Implied Volatilities ({nb_options} options used)")
df["Implied Vol"] = df.apply(
    lambda x: black_scholes.BS_ImpliedVol(f=x["Forward"], k=x["Strike Perc"],
                                          t=(x["Maturity"] - x["Spot Date"]).days / 365,
                                          MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0]), axis=1)

# Drop Error Points
print("\nRemoving Error Points")
df = df[df["Implied Vol"] != -1].copy()
print(f" - Nb of options removed : {nb_options - len(df.index)}")
print(f" - Nb of points used in vol surface : {len(df.index)}")

# Calibration

# Plot Vol Smiles
for udl in df["Underlying"].unique():
    for maturity in df["Maturity"].unique():
        df_calls = df[(df["Maturity"] == maturity) & (df["Type"] == "Call")].copy()
        df_puts = df[(df["Maturity"] == maturity) & (df["Type"] == "Put")].copy()
        plt.plot(df_calls["Strike Perc"], df_calls["Implied Vol"], 'o', color='orange', label="Calls")
        plt.plot(df_puts["Strike Perc"], df_puts["Implied Vol"], 'o', color='blue', label="Puts")
        plt.plot(df_calls["Strike Perc"], df_calls["IVM"]/100, '.', color='black', label="BBG Calls")
        plt.plot(df_puts["Strike Perc"], df_puts["IVM"]/100, '.', color='grey', label="BBG Puts")
        plt.title(f'{udl} - {pd.to_datetime(str(maturity)).strftime("%d.%m.%Y")} - ATM Vol')
        plt.legend()
        plt.grid()
        plt.show()
