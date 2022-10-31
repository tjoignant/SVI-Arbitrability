import os
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression

import calibration
import black_scholes

# Initialisation
start = time.perf_counter()
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
df["Pretty Maturity"] = df["Maturity"].apply(lambda x: x.strftime("%b-%y"))

# Dropping Useless Columns
df = df[["Type", "Underlying", "Spot", "Spot Date", "Maturity", "Pretty Maturity", "Maturity (in Y)", "Strike",
         "Strike Perc", "Mid", "Volm", "IVM"]]

# Timer
end = time.perf_counter()
print(f"\n1/ Datas Loaded & Parsed ({round(end - start, 1)}s)")
start = end

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Initial")

# Data Coherence Verification
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
                            [row_id, min(df_check.index[row_id_pos + 1], df_check.index[-1]),
                             min(df_check.index[row_id_pos + 2], df_check.index[-1])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                else:
                    df_check["Butterfly"] = df_check["Mid"] - df_check["Mid"].shift(1) * \
                                            (1 + (df_check["Strike"].shift(1) - df_check["Strike"]) / (
                                                    df_check["Strike"].shift(2) - df_check["Strike"].shift(1))) + \
                                            df_check["Mid"].shift(2) * ((
                                                                                df_check["Strike"].shift(1) - df_check[
                                                                            "Strike"]) / (
                                                                                df_check["Strike"].shift(2) -
                                                                                df_check["Strike"].shift(1)))
                    id_with_arbitrage = list(df_check[df_check["Butterfly"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[[row_id, max(df_check.index[row_id_pos - 1], df_check.index[0]),
                                                  max(df_check.index[row_id_pos - 2], df_check.index[0])]]
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

# Timer
end = time.perf_counter()
print(f"2/ Arbitrage Coherence Checked ({round(end - start, 1)}s)")
start = end

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Arbitrages")

# Retrieve Forward & ZC (per maturity)
df = df.sort_values(by="Maturity", ascending=True)
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
            Y_list.append(float(df_reg[(df_reg["Strike"] == strike) & (df_reg["Type"] == "Call")]["Mid"]) - float(
                df_reg[(df_reg["Strike"] == strike) & (df_reg["Type"] == "Put")]["Mid"]))
        x = np.array(np.array(K_list) / spot).reshape((-1, 1))
        y = np.array(np.array(Y_list) / spot)
        model = LinearRegression().fit(x, y)
        beta = model.coef_[0]
        alpha = model.intercept_
        zc = -beta
        forward = alpha / zc
        df.loc[df["Maturity"] == maturity, ['Forward', 'ZC']] = forward, zc

# Remove ITM Options
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()

# Timer
end = time.perf_counter()
print(f"3/ Forward & ZC Computed + ITM Options Removed ({round(end - start, 1)}s)")
start = end

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Regression")

# Compute Implied Volatilities
df["Implied Vol"] = df.apply(
    lambda x: black_scholes.BS_ImpliedVol(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                          MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0]), axis=1)

# # Compute Implied Volatilities with Brent method
# df["Implied Vol Brent"] = df.apply(
#     lambda x: black_scholes.BS_ImpliedVol_Brent(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
#                                           MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0]), axis=1)
# df["Ecart"] = abs(df["Implied Vol Brent"] - df["Implied Vol"])
# comparison = []
# for i in range(len(df["Implied Vol"])):
#     comparison.append([])
#     comparison[i].append(df["Implied Vol"].iloc[i])
#     comparison[i].append(df["Implied Vol Brent"].iloc[i])
#     comparison[i].append(df["Ecart"].iloc[i])
#     print(comparison[i])
# df.drop(["Implied Vol Brent", "Ecart"], axis=1)
#
# # Drop Error Points
# df = df[df["Implied Vol"] != -1].copy()

# Create Implied Vol Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["Implied Vol"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_surface = pd.concat(df_list, axis=1)
df_surface.sort_index(inplace=True)

# Timer
end = time.perf_counter()
print(f"4/ Market Implied Volatilities Computed ({round(end - start, 1)}s)")
start = end

# Add Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Newton-Raphson")

# Compute Log Forward Moneyness & Implied Total Variance
df["Log Forward Moneyness"] = df.apply(lambda x: np.log(x["Strike Perc"] / x["Forward"]), axis=1)
df["Implied TV"] = df["Implied Vol"] * df["Implied Vol"] * df["Maturity (in Y)"]

# Calibrate SVI Curves & Compute ATM Implied Total Variance
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = calibration.SVI_calibration(
        k_list=list(df_mat["Log Forward Moneyness"]),
        mktTotVar_list=list(df_mat["Implied TV"]),
        weights_list=list(df_mat["Implied Vol"]),
    )
    df.loc[df["Maturity"] == maturity, ['SVI Params']] = [SVI_params] * len(df_mat.index)
    df.loc[df["Maturity"] == maturity, ['ATMF Implied TV']] = \
        calibration.SVI(k=0, a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"],
                        m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])

# Timer
end = time.perf_counter()
print(f"5/ SVI Curves Calibrated ({round(end - start, 1)}s)")
start = end

# Calibrate SSVI Surface
SSVI_params = calibration.SSVI_calibration(
    k_list=list(df["Log Forward Moneyness"]),
    atmfTotVar_list=list(df["ATMF Implied TV"]),
    mktTotVar_list=list(df["Implied TV"]),
    weights_list=list(df["Implied Vol"]),
)

# Timer
end = time.perf_counter()
print(f"6/ SSVI Surface Calibrated ({round(end - start, 1)}s)")
start = end

# Display Calibration Results
print("\nSSVI calibrated parameters:")
print(f"  - rho = {SSVI_params['rho_']}")
print(f"  - eta = {SSVI_params['eta_']}")
print(f"  - lambda = {SSVI_params['lambda_']}")

# Set Graphs Infos
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig.suptitle(f"Calibration Results (as of {spot_date.strftime('%d-%b-%Y')})", fontweight='bold', fontsize=14)

# Plot Number of Options Used
axs[0, 0].plot(nb_options_text, nb_options, "-")
axs[0, 0].set_title("Options Per Calibration Steps")
axs[0, 0].grid()

# Plot Implied Forward & ZC
maturity_list = []
forward_list = []
zc_list = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    forward_list.append(list(df_bis["Forward"].unique())[0])
    zc_list.append(list(df_bis["ZC"].unique())[0])
    maturity_list.append(maturity)
axs[1, 0].plot(maturity_list, forward_list, label="Forward")
axs[1, 0].plot(maturity_list, zc_list, label="ZC")
axs[1, 0].set_title(f"Implied Forward & ZC")
axs[1, 0].legend(loc="upper right")
axs[1, 0].grid()
axs[1, 0].xaxis.set_major_formatter(DateFormatter("%b-%y"))
axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Plot Implied Total Variance
percentage = 70
for strike in df["Strike"].unique():
    df_strike = df[(df["Strike"] == strike)].copy()
    if len(df_strike["Maturity"].unique()) >= percentage / 100 * len(df["Maturity"].unique()):
        total_implied_var = []
        for maturity in df_strike["Maturity"].unique():
            df_bis = df[(df["Strike"] == strike) & (df["Maturity"] == maturity)].copy()
            total_implied_var.append(df_bis["Implied TV"].unique())
        axs[0, 1].plot(df_strike["Maturity"].unique(), total_implied_var, label=strike)
axs[0, 1].grid()
axs[0, 1].legend(loc="upper right")
axs[0, 1].set_title("Market Implied Total Variance")
axs[0, 1].xaxis.set_major_formatter(DateFormatter("%b-%y"))
axs[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Plot Market Implied Volatilities
for maturity in df["Pretty Maturity"].unique():
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    axs[1, 1].plot(df_bis["Strike"], df_bis["Implied Vol"], label=maturity)
axs[1, 1].scatter(df["Strike"], df["IVM"] / 100, marker=".", color="black", label="BBG IV")
axs[1, 1].grid()
axs[1, 1].set_title("Market Implied Volatility")
axs[1, 1].legend(loc="upper right")

# Plot SVI Calibration
k_list = np.linspace(-1, 1, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_bis["SVI Params"])[0]
    svi_list = [calibration.SVI(k=k, a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"],
                                m_=SVI_params["m_"], sigma_=SVI_params["sigma_"]) for k in k_list]
    axs[0, 2].plot(k_list, svi_list, label=list(df_bis["Pretty Maturity"])[0])
    axs[0, 2].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs[0, 2].grid()
axs[0, 2].legend(loc="upper right")
axs[0, 2].set_title("SVI Calibration")

# Plot SSVI Calibration
k_list = np.linspace(-1, 1, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["ATMF Implied TV"])[0]
    svi_list = [calibration.SSVI(k=k, theta=theta, rho_=SSVI_params["rho_"], eta_=SSVI_params["eta_"],
                                 lambda_=SSVI_params["lambda_"]) for k in k_list]
    axs[1, 2].plot(k_list, svi_list, label=list(df_bis["Pretty Maturity"])[0])
    axs[1, 2].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs[1, 2].grid()
axs[1, 2].legend(loc="upper right")
axs[1, 2].set_title("SSVI Calibration")

# Show Graphs
plt.tight_layout()
plt.show()
