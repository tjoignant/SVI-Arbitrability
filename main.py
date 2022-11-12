import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
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
min_volume = 10
strike_list = [2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]
nb_options = []
nb_options_text = []
tick_font_size = 8.5
title_font_size = 11
timer_id = 1
legend_loc = "upper right"
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

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

# Compute Mid Price & Maturity
df["Mid"] = (df["Bid"] + df["Ask"]) / 2
df["Mid Perc"] = df["Mid"] / df["Spot"]
df["Maturity (in Y)"] = df.apply(lambda x: (x["Maturity"] - x["Spot Date"]).days / 365, axis=1)
df["Pretty Maturity"] = df["Maturity"].apply(lambda x: x.strftime("%b-%y"))

# Timer
end = time.perf_counter()
print(f"\n{timer_id}/ Datas Loaded & Parsed ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Set Nb Options Lists
nb_options.append(len(df.index))
nb_options_text.append("Initial")

# Data Coherence Verification
nb_arbitrage = 1
while nb_arbitrage > 0:
    nb_arbitrage = 0
    index_list = []
    # Identify Call/Put Spread & Butterfly Arbitrages
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
                    df_check["Spread"] = df_check["Mid"] - df_check["Mid"].shift(-1)
                    id_with_butterfly_arbitrage = list(df_check[df_check["Butterfly"] <= 0].index)
                    id_with_spread_arbitrage = list(df_check[df_check["Spread"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_butterfly_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[
                            [row_id, min(df_check.index[row_id_pos + 1], df_check.index[-1]),
                             min(df_check.index[row_id_pos + 2], df_check.index[-1])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                    for row_id in id_with_spread_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[
                            [row_id, min(df_check.index[row_id_pos + 1], df_check.index[-1])]]
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
                    df_check["Spread"] = df_check["Mid"] - df_check["Mid"].shift(1)
                    id_with_butterfly_arbitrage = list(df_check[df_check["Butterfly"] <= 0].index)
                    id_with_spread_arbitrage = list(df_check[df_check["Spread"] <= 0].index)
                    id_to_remove = []
                    for row_id in id_with_butterfly_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[[row_id, max(df_check.index[row_id_pos - 1], df_check.index[0]),
                                                  max(df_check.index[row_id_pos - 2], df_check.index[0])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                    for row_id in id_with_spread_arbitrage:
                        row_id_pos = list(df_check.index).index(row_id)
                        df_select = df_check.loc[
                            [row_id, max(df_check.index[row_id_pos - 1], df_check.index[0])]]
                        id_to_remove.append(df_select[['Volm']].idxmin()[0])
                index_list = list(set(index_list + id_to_remove))
    # Identify Calendar Spread Arbitrages
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
    # Remove Less Liquid Options Causing An Arbitrage (either Calendar Spread, Call/Put Spread, Butterfly)
    nb_arbitrage = len(index_list)
    if nb_arbitrage > 0:
        df = df.drop(index_list).reset_index(drop=True)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Arbitrage Coherence Checked ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Add Remaining Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Arbitrages")

# Retrieve Forward & ZC (per maturity)
for maturity in df["Maturity"].unique():
    df_reg = df[df["Maturity"] == maturity].copy()
    # Remove Strikes with less than 1 Calls & Puts
    for strike in df_reg["Strike"].unique():
        if len(df_reg[(df_reg["Strike"] == strike)].index) == 1:
            df_reg = df_reg[df_reg["Strike"] != strike].copy()
    # Remove Maturities with less than 2 Pair of Call(K) & Put(K) (no regression possible)
    if len(df_reg.index) < 4:
        df = df[df["Maturity"] != maturity].copy()
    # Else Compute ZC & Forward
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

# Add Remaining Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Forward+ZC")

# Remove ITM Options
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Market Implied Forward & ZC Computed ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Add Remaining Nb Options
nb_options.append(len(df.index))
nb_options_text.append("ITM Opt.")

# Compute Implied Volatilities
df["Implied Vol (Newton-Raphson)"] = df.apply(
    lambda x: black_scholes.BS_IV_Newton_Raphson(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                                 MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0])[0],
    axis=1)
df["Implied Vol (Brent)"] = df.apply(
    lambda x: black_scholes.BS_IV_Brent(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                        MktPrice=x["Mid"] / x["Spot"], df=x["ZC"], OptType=x["Type"][0])[0], axis=1)

# Set Brent Implied Volatilities
df["Implied Vol"] = df["Implied Vol (Brent)"]

# Add Remaining Nb Options
nb_options_brent = nb_options + [len(df[df["Implied Vol (Brent)"] != -1].index)]
nb_options_nr = nb_options + [len(df[df["Implied Vol (Newton-Raphson)"] != -1].index)]
nb_options_text.append("Implied Vol.")

# Drop Implied Vol Error Points
df = df[df["Implied Vol"] != -1].copy()

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Market Implied Volatilities Computed with Brent & Newton-Raphson ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute Log Forward Moneyness & Implied Total Variance (Implied TV)
df["Log Forward Moneyness"] = df.apply(lambda x: np.log(x["Strike Perc"] / x["Forward"]), axis=1)
df["Implied TV"] = df["Implied Vol"] * df["Implied Vol"] * df["Maturity (in Y)"]

# Calibrate SVI Curves
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

# Calibrate SSVI Surface
SSVI_params = calibration.SSVI_calibration(
    k_list=list(df["Log Forward Moneyness"]),
    atmfTotVar_list=list(df["ATMF Implied TV"]),
    mktTotVar_list=list(df["Implied TV"]),
    weights_list=list(df["Implied Vol"]),
)

# Calibrate eSSVI Surface
eSSVI_params = calibration.eSSVI_calibration(
    k_list=list(df["Log Forward Moneyness"]),
    atmfTotVar_list=list(df["ATMF Implied TV"]),
    mktTotVar_list=list(df["Implied TV"]),
    weights_list=list(df["Implied Vol"]),
)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ SVI, SSVI & eSSVI Surfaces Calibrated ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute SVI, SSVI & eSSVI Total Variance (TV)
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_mat["SVI Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SVI TV']] = \
        df[df["Maturity"] == maturity].apply(lambda x:
                                             calibration.SVI(k=x["Log Forward Moneyness"], a_=SVI_params["a_"],
                                                             b_=SVI_params["b_"],
                                                             rho_=SVI_params["rho_"], m_=SVI_params["m_"],
                                                             sigma_=SVI_params["sigma_"]), axis=1)
df["SSVI TV"] = df.apply(lambda x:
                         calibration.SSVI(k=x["Log Forward Moneyness"], theta=x["ATMF Implied TV"],
                                          rho_=SSVI_params["rho_"],
                                          eta_=SSVI_params["eta_"], lambda_=SSVI_params["lambda_"]), axis=1)
df["eSSVI TV"] = df.apply(lambda x:
                          calibration.eSSVI(k=x["Log Forward Moneyness"], theta=x["ATMF Implied TV"],
                                            a_=eSSVI_params["a_"],
                                            b_=eSSVI_params["b_"], c_=eSSVI_params["c_"], eta_=eSSVI_params["eta_"],
                                            lambda_=eSSVI_params["lambda_"]), axis=1)

# Compute SVI, SSVI & eSSVI Volatilities
df["SVI Vol"] = df.apply(lambda x: np.sqrt(x["SVI TV"] / x["Maturity (in Y)"]), axis=1)
df["SSVI Vol"] = df.apply(lambda x: np.sqrt(x["SSVI TV"] / x["Maturity (in Y)"]), axis=1)
df["eSSVI Vol"] = df.apply(lambda x: np.sqrt(x["eSSVI TV"] / x["Maturity (in Y)"]), axis=1)

# Compute SVI, SSVI & eSSVI Absolute Volatility Errors (in %)
df["SVI Vol Error Perc"] = df.apply(lambda x: abs(x["SVI Vol"] - x["Implied Vol"]) * 100, axis=1)
df["SSVI Vol Error Perc"] = df.apply(lambda x: abs(x["SSVI Vol"] - x["Implied Vol"]) * 100, axis=1)
df["eSSVI Vol Error Perc"] = df.apply(lambda x: abs(x["eSSVI Vol"] - x["Implied Vol"]) * 100, axis=1)

# Compute SVI, SSVI & eSSVI BS Absolute Price Errors (in %)
df["SVI Price Error Perc"] = \
    df.apply(lambda x: abs(black_scholes.BS_Price(
        f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"], v=x["SVI Vol"], df=x["ZC"],
        OptType=x["Type"][0]) - x["Mid Perc"]) * 100, axis=1)
df["SSVI Price Error Perc"] = \
    df.apply(lambda x: abs(black_scholes.BS_Price(
        f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"], v=x["SSVI Vol"], df=x["ZC"],
        OptType=x["Type"][0]) - x["Mid Perc"]) * 100, axis=1)
df["eSSVI Price Error Perc"] = \
    df.apply(lambda x: abs(black_scholes.BS_Price(
        f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"], v=x["eSSVI Vol"], df=x["ZC"],
        OptType=x["Type"][0]) - x["Mid Perc"]) * 100, axis=1)

# Compute SVI Absolute Volatility Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SVI Vol Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_svi_vol_error_surface = pd.concat(df_list, axis=1)
df_svi_vol_error_surface.sort_index(inplace=True)

# Compute SSVI Absolute Volatility Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SSVI Vol Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_ssvi_vol_error_surface = pd.concat(df_list, axis=1)
df_ssvi_vol_error_surface.sort_index(inplace=True)

# Compute eSSVI Absolute Volatility Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["eSSVI Vol Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_essvi_vol_error_surface = pd.concat(df_list, axis=1)
df_essvi_vol_error_surface.sort_index(inplace=True)

# Compute SVI Absolute Price Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SVI Price Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_svi_price_error_surface = pd.concat(df_list, axis=1)
df_svi_price_error_surface.sort_index(inplace=True)

# Compute SSVI Absolute Price Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SSVI Price Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_ssvi_price_error_surface = pd.concat(df_list, axis=1)
df_ssvi_price_error_surface.sort_index(inplace=True)

# Compute eSSVI Absolute Price Error Surface
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["eSSVI Price Error Perc"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_essvi_price_error_surface = pd.concat(df_list, axis=1)
df_essvi_price_error_surface.sort_index(inplace=True)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ SVI, SSVI & eSSVI Absolute Volatility & Price Errors Computed ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute SVI, SSVI & eSSVI Skew
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_mat["SVI Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SVI Skew']] = df[df["Maturity"] == maturity].apply(lambda x:
                                                                                            calibration.SVI_skew(
                                                                                                strike=x["Strike Perc"],
                                                                                                forward=x["Forward"],
                                                                                                maturity=x[
                                                                                                    "Maturity (in Y)"],
                                                                                                a_=SVI_params["a_"],
                                                                                                b_=SVI_params["b_"],
                                                                                                rho_=SVI_params["rho_"],
                                                                                                m_=SVI_params["m_"],
                                                                                                sigma_=SVI_params[
                                                                                                    "sigma_"]), axis=1)
df["SSVI Skew"] = df.apply(lambda x:
                           calibration.SSVI_skew(strike=x["Strike Perc"], theta=x["ATMF Implied TV"],
                                                 maturity=x["Maturity (in Y)"],
                                                 forward=x["Forward"], rho_=SSVI_params["rho_"],
                                                 eta_=SSVI_params["eta_"],
                                                 lambda_=SSVI_params["lambda_"]), axis=1)
df["eSSVI Skew"] = df.apply(lambda x:
                            calibration.eSSVI_skew(strike=x["Strike Perc"], theta=x["ATMF Implied TV"],
                                                   maturity=x["Maturity (in Y)"],
                                                   forward=x["Forward"], eta_=eSSVI_params["eta_"],
                                                   lambda_=eSSVI_params["lambda_"],
                                                   a_=eSSVI_params["a_"], b_=eSSVI_params["b_"],
                                                   c_=eSSVI_params["c_"], ), axis=1)

# Compute SVI, SSVI & eSSVI Convexity
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_mat["SVI Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SVI Convexity']] = df[df["Maturity"] == maturity].apply(lambda x:
                                                                                                 calibration.SVI_convexity(
                                                                                                     strike=x[
                                                                                                         "Strike Perc"],
                                                                                                     forward=x[
                                                                                                         "Forward"],
                                                                                                     maturity=x[
                                                                                                         "Maturity (in Y)"],
                                                                                                     a_=SVI_params[
                                                                                                         "a_"],
                                                                                                     b_=SVI_params[
                                                                                                         "b_"],
                                                                                                     rho_=SVI_params[
                                                                                                         "rho_"],
                                                                                                     m_=SVI_params[
                                                                                                         "m_"],
                                                                                                     sigma_=SVI_params[
                                                                                                         "sigma_"]),
                                                                                                 axis=1)

# Compute BS Delta Strike, Vega, d1 & d2
df["Delta Strike"] = df.apply(
    lambda x: black_scholes.BS_Delta_Strike(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                            v=x["Implied Vol"], df=x["ZC"], OptType=x["Type"][0]), axis=1)
df["Vega"] = df.apply(
    lambda x: black_scholes.BS_Vega(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                    v=x["Implied Vol"], df=x["ZC"], OptType=x["Type"][0]), axis=1)
df["d1"] = df.apply(
    lambda x: black_scholes.BS_d1(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                  v=x["Implied Vol"]), axis=1)
df["d2"] = df.apply(
    lambda x: black_scholes.BS_d2(f=x["Forward"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                  v=x["Implied Vol"]), axis=1)

# Reorder Dataframe
df = df.sort_values(by=["Maturity", "Strike"], ascending=[True, True])

# Compute Gourion-Lucic Skew Bounds
for maturity in df["Maturity"].unique():
    # Types Condition
    call_cond = (df["Maturity"] == maturity) & (df["Type"] == "Call")
    put_cond = (df["Maturity"] == maturity) & (df["Type"] == "Put")
    # S_min (call)
    df_bis = df[call_cond].copy()
    df.loc[call_cond, ['s_min']] = ((df_bis["Mid"] - df_bis["Mid"].shift(1)) / (
            df_bis["Strike"] - df_bis["Strike"].shift(1)) - df_bis["Delta Strike"]) / df_bis["Vega"]
    df.loc[df[call_cond].index[0], "s_min"] = ((df_bis["Mid"].values[0] - df_bis["Spot"].values[0]) / (
        df_bis["Strike"].values[0]) - df_bis["Delta Strike"].values[0]) / df_bis["Vega"].values[0]
    # S_max (call)
    df.loc[call_cond, ['s_max']] = ((df_bis["Mid"].shift(-1) - df_bis["Mid"]) / (
            df_bis["Strike"].shift(-1) - df_bis["Strike"]) - df_bis["Delta Strike"]) / df_bis["Vega"]
    df.loc[df[call_cond].index[-1], "s_max"] = - df_bis["Delta Strike"].values[-1] / df_bis["Vega"].values[-1]
    # S_min (put)
    df_bis = df[put_cond].copy()
    df.loc[put_cond, ['s_min']] = ((df_bis["Mid"] - df_bis["Mid"].shift(1)) / (
            df_bis["Strike"] - df_bis["Strike"].shift(1)) - df_bis["Delta Strike"]) / df_bis["Vega"]
    df.loc[df[put_cond].index[0], "s_min"] = - df_bis["Delta Strike"].values[0] / df_bis["Vega"].values[0]
    # S_max (put)
    df.loc[put_cond, ['s_max']] = ((df_bis["Mid"].shift(-1) - df_bis["Mid"]) / (
            df_bis["Strike"].shift(-1) - df_bis["Strike"]) - df_bis["Delta Strike"]) / df_bis["Vega"]
    df.loc[df[put_cond].index[-1], "s_max"] = ((df_bis["Spot"].values[-1] - df_bis["Mid"].values[-1]) / (
        df_bis["Strike"].values[-1]) - df_bis["Delta Strike"].values[-1]) / df_bis["Vega"].values[-1]

# Compute SVI, SSVI & eSSVI Gourion-Lucic Convexity Bounds
df["s_conv_min SVI"] = df.apply(
    lambda x: -(1 / x["SVI Vol"]) * (1 / (x["Strike"] * np.sqrt(x["Maturity (in Y)"])) + x["d2"] * x["SVI Skew"]) *
              (1 / (x["Strike"] * np.sqrt(x["Maturity (in Y)"])) + x["d1"] * x["SVI Skew"]) - (1 / x["Strike"]) *
              x["SVI Skew"], axis=1)

# Compute SVI, SSVI & eSSVI Gourion-Lucic Skew Bounds Test
df["SVI Skew Test"] = df.apply(lambda x: 0 if x["s_min"] < x["SVI Skew"] < x["s_max"] else 1, axis=1)
df["SSVI Skew Test"] = df.apply(lambda x: 0 if x["s_min"] < x["SSVI Skew"] < x["s_max"] else 1, axis=1)
df["eSSVI Skew Test"] = df.apply(lambda x: 0 if x["s_min"] < x["eSSVI Skew"] < x["s_max"] else 1, axis=1)

# Create Gourion-Lucic Skew Bounds Arbitrability Surface (SVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SVI Skew Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_svi_skew_arb_surface = pd.concat(df_list, axis=1)
df_svi_skew_arb_surface.sort_index(inplace=True)

# Create Gourion-Lucic Skew Bounds Arbitrability Surface (SSVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SSVI Skew Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_ssvi_skew_arb_surface = pd.concat(df_list, axis=1)
df_ssvi_skew_arb_surface.sort_index(inplace=True)

# Create Gourion-Lucic Skew Bounds Arbitrability Surface (eSSVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["eSSVI Skew Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_essvi_skew_arb_surface = pd.concat(df_list, axis=1)
df_essvi_skew_arb_surface.sort_index(inplace=True)

# Compute SVI, SSVI & eSSVI Gourion-Lucic Skew & Convexity Bounds Test
df["SVI Skew+Convexity Test"] = df.apply(
    lambda x: 0 if x["s_min"] < x["SVI Skew"] < x["s_max"] and x["s_conv_min SVI"] < x["SVI Convexity"] else 1, axis=1)

# Create Gourion-Lucic Skew & Convexity Bounds Arbitrability Surface (SVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SVI Skew+Convexity Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_svi_skew_conv_arb_surface = pd.concat(df_list, axis=1)
df_svi_skew_conv_arb_surface.sort_index(inplace=True)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Gourion-Lucic Arbitrability Bounds Test Concluded ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute SVI, SSVI & eSSVI Binary European Up & In (BEUI) Price
df["SVI BEUI"] = df.apply(lambda x: -(x["Delta Strike"] + x["Vega"] * x["SVI Skew"]), axis=1)
df["SSVI BEUI"] = df.apply(lambda x: -(x["Delta Strike"] + x["Vega"] * x["SSVI Skew"]), axis=1)
df["eSSVI BEUI"] = df.apply(lambda x: -(x["Delta Strike"] + x["Vega"] * x["eSSVI Skew"]), axis=1)

# Compute SVI, SSVI & eSSVI Binary European Down & In (BEDI) Price
df["SVI BEDI"] = df.apply(lambda x: 1 - x["SVI BEUI"], axis=1)
df["SSVI BEDI"] = df.apply(lambda x: 1 - x["SSVI BEUI"], axis=1)
df["eSSVI BEDI"] = df.apply(lambda x: 1 - x["eSSVI BEUI"], axis=1)

# Compute SVI, SSVI & eSSVI Shark-Jaw Test
for maturity in df["Maturity"].unique():
    # Type Condition
    call_cond = (df["Maturity"] == maturity) & (df["Type"] == "Call")
    put_cond = (df["Maturity"] == maturity) & (df["Type"] == "Put")
    # Call Triangles (CT)
    df_bis = df[call_cond].copy()
    df.loc[call_cond, ['SVI SJ CT']] = df_bis["Mid"].shift(1) - df_bis["Mid"] - (
                df_bis["Strike"].shift(1) - df_bis["Strike"]) * df_bis["SVI BEDI"]
    df.loc[call_cond, ['SSVI SJ CT']] = df_bis["Mid"].shift(1) - df_bis["Mid"] - (
                df_bis["Strike"].shift(1) - df_bis["Strike"]) * df_bis["SSVI BEDI"]
    df.loc[call_cond, ['eSSVI SJ CT']] = df_bis["Mid"].shift(1) - df_bis["Mid"] - (
                df_bis["Strike"].shift(1) - df_bis["Strike"]) * df_bis["eSSVI BEDI"]
    # Put Triangles (PT)
    df_bis = df[put_cond].copy()
    df.loc[put_cond, ['SVI SJ PT']] = df_bis["Mid"].shift(-1) - df_bis["Mid"] - (
                df_bis["Strike"] - df_bis["Strike"].shift(-1)) * df_bis["SVI BEDI"]
    df.loc[put_cond, ['SSVI SJ PT']] = df_bis["Mid"].shift(-1) - df_bis["Mid"] - (
                df_bis["Strike"] - df_bis["Strike"].shift(-1)) * df_bis["SSVI BEDI"]
    df.loc[put_cond, ['eSSVI SJ PT']] = df_bis["Mid"].shift(-1) - df_bis["Mid"] - (
                df_bis["Strike"] - df_bis["Strike"].shift(-1)) * df_bis["eSSVI BEDI"]

# Compute SVI, SSVI & eSSVI Shark-Jaw Test
df["SVI SJ Test"] = df.apply(lambda x: 0
    if (x["SVI SJ CT"] > 0 and x["SVI SJ PT"] > 0) else 0
        if (pd.isna(x["SVI SJ CT"]) and x["SVI SJ PT"] > 0) else 0
            if (x["SVI SJ CT"] > 0 and pd.isna(x["SVI SJ PT"])) else 0
                if (pd.isna(x["SVI SJ CT"]) and pd.isna(x["SVI SJ PT"])) else 1, axis=1)
df["SSVI SJ Test"] = df.apply(lambda x: 0
    if (x["SSVI SJ CT"] > 0 and x["SSVI SJ PT"] > 0) else 0
        if (pd.isna(x["SSVI SJ CT"]) and x["SSVI SJ PT"] > 0) else 0
            if (x["SSVI SJ CT"] > 0 and pd.isna(x["SSVI SJ PT"])) else 0
                if (pd.isna(x["SSVI SJ CT"]) and pd.isna(x["SSVI SJ PT"])) else 1, axis=1)
df["eSSVI SJ Test"] = df.apply(lambda x: 0
    if (x["eSSVI SJ CT"] > 0 and x["eSSVI SJ PT"] > 0) else 0
        if (pd.isna(x["eSSVI SJ CT"]) and x["eSSVI SJ PT"] > 0) else 0
            if (x["eSSVI SJ CT"] > 0 and pd.isna(x["eSSVI SJ PT"])) else 0
                if (pd.isna(x["eSSVI SJ CT"]) and pd.isna(x["eSSVI SJ PT"])) else 1, axis=1)

# Create Shark-Jaw Arbitrability Surface (SVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SVI SJ Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_svi_sj_arb_surface = pd.concat(df_list, axis=1)
df_svi_sj_arb_surface.sort_index(inplace=True)

# Create Shark-Jaw Arbitrability Surface (SSVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["SSVI SJ Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_ssvi_sj_arb_surface = pd.concat(df_list, axis=1)
df_ssvi_sj_arb_surface.sort_index(inplace=True)

# Create Shark-Jaw Arbitrability Surface (eSSVI)
df_list = []
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
    df_mat.index = df_mat["Strike"]
    df_mat = df_mat[["eSSVI SJ Test"]]
    df_mat.columns = [maturity]
    df_list.append(df_mat)
df_essvi_sj_arb_surface = pd.concat(df_list, axis=1)
df_essvi_sj_arb_surface.sort_index(inplace=True)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Gourion Shark-Jaw Test Concluded ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Set Graphs Infos
plt.tight_layout()
fig1, axs1 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig2, axs2 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig3, axs3 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig4, axs4 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5), sharey=True)
fig1.suptitle(f"Market Data Coherence Verification ({spot_date.strftime('%d-%b-%Y')})", fontweight='bold',
              fontsize=12.5)
fig2.suptitle(f"Parametric Volatilities Calibration ({spot_date.strftime('%d-%b-%Y')})", fontweight='bold',
              fontsize=12.5)
fig3.suptitle(f"Volatility Calibration Errors ({spot_date.strftime('%d-%b-%Y')})", fontweight='bold',
              fontsize=12.5)
fig4.suptitle(f"Arbitograms ({spot_date.strftime('%d-%b-%Y')})", fontweight='bold',
              fontsize=12.5)

# Plot Number of Options Per Steps
nb_options_text.append("Calibration")
nb_options_nr.append(nb_options_nr[-1])
nb_options_brent.append(nb_options_brent[-1])
axs1[0, 0].step(nb_options_text, nb_options_nr, "--", where='post', label="Newton Raphson")
axs1[0, 0].step(nb_options_text, nb_options_brent, where='post', label="Brent")
axs1[0, 0].set_title("Option Number Per Steps", fontsize=title_font_size)
axs1[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[0, 0].legend(loc=legend_loc)
axs1[0, 0].grid()

# Plot Implied Forward & ZC
maturity_list = []
forward_list = []
zc_list = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    forward_list.append(list(df_bis["Forward"].unique())[0])
    zc_list.append(list(df_bis["ZC"].unique())[0])
    maturity_list.append(maturity)
axs1[1, 0].plot(maturity_list, forward_list, label="Forward")
axs1[1, 0].plot(maturity_list, zc_list, label="ZC")
axs1[1, 0].set_title(f"Market Implied Forward & ZC", fontsize=title_font_size)
axs1[1, 0].legend(loc="upper right")
axs1[1, 0].grid()
axs1[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 0].xaxis.set_major_formatter(DateFormatter("%b-%y"))
axs1[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Plot Implied Total Variance
percentage = 70
for strike in df["Strike"].unique():
    df_strike = df[(df["Strike"] == strike)].copy()
    if len(df_strike["Maturity"].unique()) >= percentage / 100 * len(df["Maturity"].unique()):
        total_implied_var = []
        for maturity in df_strike["Maturity"].unique():
            df_bis = df[(df["Strike"] == strike) & (df["Maturity"] == maturity)].copy()
            total_implied_var.append(df_bis["Implied TV"].unique())
        axs1[0, 1].plot(df_strike["Maturity"].unique(), total_implied_var, label=strike)
axs1[0, 1].grid()
axs1[0, 1].legend(loc=legend_loc)
axs1[0, 1].set_title("Market Implied Total Variances", fontsize=title_font_size)
axs1[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[0, 1].xaxis.set_major_formatter(DateFormatter("%b-%y"))
axs1[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Plot Market Implied Volatilities
for maturity in df["Pretty Maturity"].unique():
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    axs1[1, 1].plot(df_bis["Strike"], df_bis["Implied Vol"], label=maturity)
axs1[1, 1].scatter(df["Strike"], df["IVM"] / 100, marker=".", color="black", label="BBG IV")
axs1[1, 1].grid()
axs1[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 1].set_title("Market Implied Volatilities", fontsize=title_font_size)
axs1[1, 1].legend(loc=legend_loc)

# Plot Market Implied Delta Strike
for maturity, i in zip(df["Pretty Maturity"].unique(), range(0, len(df["Pretty Maturity"].unique()))):
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    df_put_bis = df_bis[(df_bis["Type"] == "Put")].copy()
    df_call_bis = df_bis[(df_bis["Type"] == "Call")].copy()
    axs1[0, 2].plot(df_put_bis["Strike"], df_put_bis["Delta Strike"], label=maturity, color=color_list[i])
    axs1[0, 2].plot(df_call_bis["Strike"], df_call_bis["Delta Strike"], color=color_list[i])
axs1[0, 2].grid()
axs1[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[0, 2].set_title("Market Implied Deltas Strike", fontsize=title_font_size)
axs1[0, 2].legend(loc=legend_loc)

# Plot Market Implied Vegas
for maturity in df["Pretty Maturity"].unique():
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    axs1[1, 2].plot(df_bis["Strike"], df_bis["Vega"], label=maturity)
axs1[1, 2].grid()
axs1[1, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 2].set_title("Market Implied Vegas", fontsize=title_font_size)
axs1[1, 2].legend(loc=legend_loc)

# Plot SVI Calibration
k_list = np.linspace(-1, 1, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_bis["SVI Params"])[0]
    svi_list = [calibration.SVI(k=k, a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"],
                                m_=SVI_params["m_"], sigma_=SVI_params["sigma_"]) for k in k_list]
    axs2[0, 0].plot(k_list, svi_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 0].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 0].grid()
axs2[0, 0].legend(loc=legend_loc)
axs2[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 0].set_title("SVI Calibration", fontsize=title_font_size)

# Plot SSVI Calibration
k_list = np.linspace(-1, 1, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["ATMF Implied TV"])[0]
    ssvi_list = [calibration.SSVI(k=k, theta=theta, rho_=SSVI_params["rho_"], eta_=SSVI_params["eta_"],
                                  lambda_=SSVI_params["lambda_"]) for k in k_list]
    axs2[0, 1].plot(k_list, ssvi_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 1].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 1].grid()
axs2[0, 1].legend(loc=legend_loc)
axs2[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 1].set_title("SSVI Calibration", fontsize=title_font_size)

# Plot eSSVI Calibration
k_list = np.linspace(-1, 1, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["ATMF Implied TV"])[0]
    essvi_list = [
        calibration.eSSVI(k=k, theta=theta, a_=eSSVI_params["a_"], b_=eSSVI_params["b_"], c_=eSSVI_params["c_"],
                          eta_=eSSVI_params["eta_"], lambda_=eSSVI_params["lambda_"]) for k in k_list]
    axs2[0, 2].plot(k_list, essvi_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 2].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 2].grid()
axs2[0, 2].legend(loc=legend_loc)
axs2[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 2].set_title("eSSVI Calibration", fontsize=title_font_size)

# Plot SVI Parameters Evolution
svi_a = []
svi_b = []
svi_rho = []
svi_m = []
svi_sigma = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_bis["SVI Params"])[0]
    svi_a.append(SVI_params["a_"])
    svi_b.append(SVI_params["b_"])
    svi_rho.append(SVI_params["rho_"])
    svi_m.append(SVI_params["m_"])
    svi_sigma.append(SVI_params["sigma_"])
axs2[1, 0].plot(df["Pretty Maturity"].unique(), svi_a, label="a")
axs2[1, 0].plot(df["Pretty Maturity"].unique(), svi_b, label="b")
axs2[1, 0].plot(df["Pretty Maturity"].unique(), svi_rho, label="rho")
axs2[1, 0].plot(df["Pretty Maturity"].unique(), svi_m, label="m")
axs2[1, 0].plot(df["Pretty Maturity"].unique(), svi_sigma, label="sigma")
axs2[1, 0].grid()
axs2[1, 0].legend(loc=legend_loc)
axs2[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 0].set_title("SVI Parameters Evolution", fontsize=title_font_size)

# Plot SSVI/eSSVI Phi Parameter Evolution
ssvi_phi = []
essvi_phi = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["ATMF Implied TV"])[0]
    ssvi_phi.append(calibration.SSVI_phi(theta=theta, eta_=SSVI_params["eta_"], lambda_=SSVI_params["lambda_"]))
    essvi_phi.append(calibration.eSSVI_phi(theta=theta, eta_=eSSVI_params["eta_"], lambda_=eSSVI_params["lambda_"]))
axs2[1, 1].plot(df["Pretty Maturity"].unique(), ssvi_phi, label="SSVI")
axs2[1, 1].plot(df["Pretty Maturity"].unique(), essvi_phi, label="eSSVI")
axs2[1, 1].grid()
axs2[1, 1].legend(loc=legend_loc)
axs2[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 1].set_title("Phi Parameter Evolution", fontsize=title_font_size)

# Plot SSVI/eSSVI Rho Parameter Evolution
ssvi_rho = []
essvi_rho = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["ATMF Implied TV"])[0]
    ssvi_rho.append(SSVI_params["rho_"])
    essvi_rho.append(
        calibration.eSSVI_rho(theta=theta, a_=eSSVI_params["a_"], b_=eSSVI_params["b_"], c_=eSSVI_params["c_"]))
axs2[1, 2].plot(df["Pretty Maturity"].unique(), ssvi_rho, label="SSVI")
axs2[1, 2].plot(df["Pretty Maturity"].unique(), essvi_rho, label="eSSVI")
axs2[1, 2].grid()
axs2[1, 2].legend(loc=legend_loc)
axs2[1, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 2].set_title("Rho Parameter Evolution", fontsize=title_font_size)

# Plot Volatility Calibration Error Surfaces
g1 = sns.heatmap(df_svi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 0], cbar=False, annot=True,
                 vmin=0, vmax=20)
g2 = sns.heatmap(df_ssvi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 1], cbar=False, annot=True,
                 vmin=0, vmax=20)
g3 = sns.heatmap(df_essvi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 2], annot=True, vmin=0,
                 vmax=20)
for g, ax, name in zip([g1, g2, g3], [axs3[0, 0], axs3[0, 1], axs3[0, 2]], ["SVI", "SSVI", "eSSVI"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Vol. Error (in %)", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Price Calibration Error Surfaces
g1 = sns.heatmap(df_svi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 0], cbar=False, annot=True,
                 vmin=0, vmax=1)
g2 = sns.heatmap(df_ssvi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 1], cbar=False, annot=True,
                 vmin=0, vmax=1)
g3 = sns.heatmap(df_essvi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 2], annot=True, vmin=0,
                 vmax=1)
for g, ax, name in zip([g1, g2, g3], [axs3[1, 0], axs3[1, 1], axs3[1, 2]], ["SVI", "SSVI", "eSSVI"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Price Error (in %)", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Skew Arbitrability Surfaces (of most liquid strikes)
g1 = sns.heatmap(df_svi_skew_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 0], cbar=False, vmin=-1, vmax=1,
                 annot=True)
g2 = sns.heatmap(df_ssvi_skew_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 1], cbar=False, vmin=-1, vmax=1,
                 annot=True)
g3 = sns.heatmap(df_essvi_skew_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 2], vmin=-1, vmax=1,
                 annot=True)
for g, ax, name in zip([g1, g2, g3], [axs4[0, 0], axs4[0, 1], axs4[0, 2]], ["SVI", "SSVI", "eSSVI"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Gourion-Lucic Bounds", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Shark-Jaw Arbitrability Surfaces (of most liquid strikes)
g1 = sns.heatmap(df_svi_sj_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 0], cbar=False, vmin=-1, vmax=1,
                 annot=True)
g2 = sns.heatmap(df_ssvi_sj_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 1], cbar=False, vmin=-1, vmax=1,
                 annot=True)
g3 = sns.heatmap(df_essvi_sj_arb_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 2], vmin=-1, vmax=1,
                 annot=True)
for g, ax, name in zip([g1, g2, g3], [axs4[1, 0], axs4[1, 1], axs4[1, 2]], ["SVI", "SSVI", "eSSVI"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Shark-Jaw", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Create Result Folder (if do not exist)
if not os.path.exists('results'):
    os.makedirs('results')

# Export Dataframes
with pd.ExcelWriter("results/Results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Dataframe")
    fig1.savefig('results/Market Data Coherence Verification.png')
    fig2.savefig('results/Parametric Volatilities Calibration.png')
    fig3.savefig('results/Volatility Calibration Errors.png')
    fig4.savefig('results/Arbitograms.png')

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Results Exported + Graphs Built ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Display Absolute Calibration Arbitrability (ACA) Scores
print("\nACA Scores:")
print(f" - SVI : {round((1 - df['SVI Skew Test'].mean()) * 10, 2)}")
print(f" - SSVI : {round((1 - df['SSVI Skew Test'].mean()) * 10, 2)}")
print(f" - eSSVI : {round((1 - df['eSSVI Skew Test'].mean()) * 10, 2)}")

# Display Graph
plt.show()
