import os

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import black_scholes

# Inputs
spot = 3318.20
spot_date = dt.datetime(day=30, month=9, year=2022)
min_volume = 1

# Set Pandas Display Settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.expand_frame_repr', False)

# Load Options Data
df_list = []
folder = 'datas'
for file in os.listdir(folder):
    df_list.append(pd.read_excel(f"{folder}/{file}", header=0).dropna())
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
df["Maturity"] = df["Ticker"].apply(lambda x: dt.datetime.strptime(x.split(" ")[1], "%m/%d/%y"))
df["Strike Perc"] = df["Strike"] / df["Spot"]

# Compute Mid Price
df["Mid"] = (df["Bid"] + df["Ask"]) / 2

# Dropping Useless Columns
df = df[["Type", "Underlying", "Spot", "Spot Date", "Maturity", "Strike", "Strike Perc", "Mid", "Volm"]]

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
                        df_select = df_check.iloc[
                            [row_id, max(row_id + 1, df_check.index[-1]), max(row_id + 2, df_check.index[-1])]]
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
                        df_select = df_check.iloc[[row_id, max(row_id - 1, df_check.index[0]), max(row_id - 2, df_check.index[0])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                index_list = list(set(index_list + id_to_remove))
        # No Calendar Spread Arbitrage
        for type in df["Type"].unique():
            for strike in df["Strike"].unique():
                df_check = df[(df["Underlying"] == udl) & (df["Strike"] == strike) & (df["Type"] == type)].copy()
                df_check = df_check.sort_values(by="Maturity", ascending=True)
                if len(df["Maturity"].unique()) > 1:
                    df_check["Calendar Spread"] = df["Mid"].diff()
                    id_with_arbitrage = list(df_check[df_check["Calendar Spread"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_arbitrage:
                        df_select = df_check.iloc[[row_id, max(row_id - 1, df_check.index[0])]]
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

# Retrieve Forward & ZC (per maturity)
for maturity in df["Maturity"].unique():
    df_check = df[df["Maturity"] == maturity].copy()
    # COMPUTE HERE REGRESSION
    # ...
    forward = 1
    zc = 1
    df.loc[df["Maturity"] == maturity, ['Forward', 'ZC']] = forward, zc

# Remove ITM Options
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()

# Compute Implied Volatilities
print("\nComputing BS Implied Volatilities")
df["Implied Vol"] = df.apply(
    lambda x: black_scholes.BS_ImpliedVol(f=x["Forward"], k=x["Strike Perc"],
                                          t=(x["Maturity"] - x["Spot Date"]).days / 365,
                                          MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0]), axis=1)

# Drop Error Points
df = df[df["Implied Vol"] != -1].copy()

# Calibration

# Plot Vol Smiles
for udl in df["Underlying"].unique():
    for maturity in df["Maturity"].unique():
        df_calls = df[df["Type"] == "Call"].copy()
        df_puts = df[df["Type"] == "Put"].copy()
        plt.plot(df_calls["Strike Perc"], df_calls["Implied Vol"], 'o', color='orange', label="Calls")
        plt.plot(df_puts["Strike Perc"], df_puts["Implied Vol"], 'o', color='blue', label="Puts")
        plt.title(f'{udl} - {pd.to_datetime(str(maturity)).strftime("%d.%m.%Y")} - Vol Smile')
        plt.legend()
        plt.grid()
        plt.show()
