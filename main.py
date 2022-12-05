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

import utils
import calibration
import black_scholes

# Variables Initialisation
start = time.perf_counter()
spot = 3375.46
spot_date = dt.datetime(day=7, month=10, year=2022)
min_volume = 10
strike_list = [2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100]
nb_options = []
nb_options_text = []
tick_font_size = 8.5
title_font_size = 11
timer_id = 1
legend_loc = "upper right"
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
log_forward_moneyness_min, log_forward_moneyness_max = -0.75, 0.75
use_durrleman_cond = False

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
df["Mid Perc"] = df["Mid"] / spot
df["Maturity (in Y)"] = df.apply(lambda x: (x["Maturity"] - x["Spot Date"]).days / 365, axis=1)
df["Pretty Maturity"] = df["Maturity"].apply(lambda x: x.strftime("%b-%y"))

# Timer
end = time.perf_counter()
print(f"\n{timer_id}/ Market Datas Loaded & Parsed ({round(end - start, 1)}s)")
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
            if len(df_check.index) >= 2:
                # Calls
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
                # Puts
                else:
                    df_check["Butterfly"] = df_check["Mid"] - df_check["Mid"].shift(1) * \
                                            (1 + (df_check["Strike"].shift(1) - df_check["Strike"]) / (
                                            df_check["Strike"].shift(2) - df_check["Strike"].shift(1))) + \
                                            df_check["Mid"].shift(2) * ((df_check["Strike"].shift(1) - df_check[
                                            "Strike"]) / (df_check["Strike"].shift(2) - df_check["Strike"].shift(1)))
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
print(f"{timer_id}/ Arbitrages Removed : Calendar Spread + Butterfly ({round(end - start, 1)}s)")
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
        df.loc[df["Maturity"] == maturity, ['Forward Perc', 'ZC Perc']] = forward, zc

# Compute Real Forward
df["Forward"] = df["Forward Perc"] * spot

# Add Remaining Nb Options
nb_options.append(len(df.index))
nb_options_text.append("Forward+ZC")

# Remove ITM Options
df = df[((df["Type"] == "Call") & (df["Strike"] >= spot)) | ((df["Type"] == "Put") & (df["Strike"] <= spot))].copy()

# Add Remaining Nb Options
nb_options.append(len(df.index))
nb_options_text.append("ITM Opt.")

# Compute Implied Volatilities
df["Implied Vol (Newton-Raphson)"] = df.apply(
    lambda x: black_scholes.BS_IV_Newton_Raphson(f=x["Forward Perc"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                                 MktPrice=x["Mid Perc"], df=x["ZC Perc"], OptType=x["Type"][0])[0],
    axis=1)
df["Implied Vol (Brent)"] = df.apply(
    lambda x: black_scholes.BS_IV_Brent(f=x["Forward Perc"], k=x["Strike Perc"], t=x["Maturity (in Y)"],
                                        MktPrice=x["Mid Perc"], df=x["ZC Perc"], OptType=x["Type"][0])[0], axis=1)

# Set Brent Implied Volatilities
df["Implied Vol"] = df["Implied Vol (Brent)"]

# Add Remaining Nb Options
nb_options_brent = nb_options + [len(df[df["Implied Vol (Brent)"] != -1].index)]
nb_options_nr = nb_options + [len(df[df["Implied Vol (Newton-Raphson)"] != -1].index)]
nb_options_text.append("Implied Vol.")

# Drop Implied Vol Error Points
df = df[df["Implied Vol"] != -1].copy()

# Reorder Dataframe
df = df.sort_values(by=["Maturity", "Strike"], ascending=[True, True])

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Implied Forward, ZC & Volatilities Computed ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute Implied Greeks
df = utils.compute_greeks(df=df, fwd_col="Forward Perc", vol_col="Implied Vol", strike_col="Strike Perc", maturity_col="Maturity (in Y)", df_col="ZC Perc", type_col="Type", name="Implied")

# Compute Log Forward Moneyness & Implied Total Variance (Implied TV)
df["Log Forward Moneyness"] = df.apply(lambda x: np.log(x["Strike Perc"] / (x["Forward Perc"])), axis=1)
df["Implied TV"] = df["Implied Vol"] * df["Implied Vol"] * df["Maturity (in Y)"]

# Set Minimisation Weight Column
df["Weight"] = df["Implied Vol"]

# Calibrate SVI Curves + Compute SVI ATMF Implied TV
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = calibration.SVI_calibration(
        k_list=list(df_mat["Log Forward Moneyness"]),
        mktTotVar_list=list(df_mat["Implied TV"]),
        weights_list=list(df_mat["Weight"]),
        use_durrleman_cond=use_durrleman_cond,
    )
    df.loc[df["Maturity"] == maturity, ['SVI Params']] = [SVI_params] * len(df_mat.index)
    df.loc[df["Maturity"] == maturity, ['SVI ATMF Implied TV']] = \
        calibration.SVI(k=0, a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"],
                        m_=SVI_params["m_"], sigma_=SVI_params["sigma_"])

# Calibrate SSVI Surface
SSVI_params = calibration.SSVI_calibration(
    k_list=list(df["Log Forward Moneyness"]),
    atmfTotVar_list=list(df["SVI ATMF Implied TV"]),
    mktTotVar_list=list(df["Implied TV"]),
    weights_list=list(df["Weight"]),
    use_durrleman_cond=use_durrleman_cond,
)
df['SSVI Params'] = [SSVI_params] * len(df.index)

# Calibrate eSSVI Surface
eSSVI_params = calibration.eSSVI_calibration(
    k_list=list(df["Log Forward Moneyness"]),
    atmfTotVar_list=list(df["SVI ATMF Implied TV"]),
    mktTotVar_list=list(df["Implied TV"]),
    weights_list=list(df["Weight"]),
    use_durrleman_cond=use_durrleman_cond,
)
df['eSSVI Params'] = [eSSVI_params] * len(df.index)

# Calibrate SABR Curves
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SABR_params = calibration.SABR_calibration(
        f_list=list(df_mat["Forward Perc"]),
        K_list=list(df_mat["Strike Perc"]),
        T_list=list(df_mat["Maturity (in Y)"]),
        mktImpVol_list=list(df_mat["Implied Vol"]),
        weights_list=list(df_mat["Weight"]),
        use_durrleman_cond=use_durrleman_cond,
    )
    df.loc[df["Maturity"] == maturity, ['SABR Params']] = [SABR_params] * len(df_mat.index)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ SVI, SSVI, eSSVI & SABR Calibrated ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute Total Variance (TV)
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_mat["SVI Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SVI TV']] = \
        df[df["Maturity"] == maturity].apply(lambda x:
                                             calibration.SVI(k=x["Log Forward Moneyness"], a_=SVI_params["a_"],
                                                             b_=SVI_params["b_"],
                                                             rho_=SVI_params["rho_"], m_=SVI_params["m_"],
                                                             sigma_=SVI_params["sigma_"]), axis=1)
    SABR_params = list(df_mat["SABR Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SABR TV']] = \
        df[df["Maturity"] == maturity].apply(lambda x:
                                             pow(calibration.SABR(
                                                 f=x["Forward Perc"], K=x["Strike Perc"], T=x["Maturity (in Y)"],
                                                 alpha_=SABR_params["alpha_"], rho_=SABR_params["rho_"],
                                                 nu_=SABR_params["nu_"]), 2) *
                                             x["Maturity (in Y)"], axis=1)
df["SSVI TV"] = df.apply(lambda x:
                         calibration.SSVI(k=x["Log Forward Moneyness"], theta=x["SVI ATMF Implied TV"],
                                          rho_=SSVI_params["rho_"],
                                          eta_=SSVI_params["eta_"], lambda_=SSVI_params["lambda_"]), axis=1)
df["eSSVI TV"] = df.apply(lambda x:
                          calibration.eSSVI(k=x["Log Forward Moneyness"], theta=x["SVI ATMF Implied TV"],
                                            a_=eSSVI_params["a_"],
                                            b_=eSSVI_params["b_"], c_=eSSVI_params["c_"], eta_=eSSVI_params["eta_"],
                                            lambda_=eSSVI_params["lambda_"]), axis=1)

# Compute BS Absolute Volatility & BS Price Errors (in %)
for surface in ["SVI", "SSVI", "eSSVI", "SABR"]:
    df[f"{surface} Vol"] = df.apply(lambda x: np.sqrt(x[f"{surface} TV"] / x["Maturity (in Y)"]), axis=1)
    df[f"{surface} Vol Error Perc"] = df.apply(lambda x: abs(x[f"{surface} Vol"] - x["Implied Vol"]) * 100, axis=1)
    df[f"{surface} Price Error Perc"] = \
        df.apply(lambda x: abs(black_scholes.BS_Price(
            f=x["Forward Perc"], k=x["Strike Perc"], t=x["Maturity (in Y)"], v=x[f"{surface} Vol"], df=x["ZC Perc"],
            OptType=x["Type"][0]) - x["Mid Perc"]) * 100, axis=1)

# Compute Absolute Volatility Error Surfaces
df_svi_vol_error_surface, svi_min, svi_max = utils.create_surface(df=df, column_name="SVI Vol Error Perc",
                                                                  strike_list=strike_list)
df_ssvi_vol_error_surface, ssvi_min, ssvi_max = utils.create_surface(df=df, column_name="SSVI Vol Error Perc",
                                                                     strike_list=strike_list)
df_essvi_vol_error_surface, essvi_min, essvi_max = utils.create_surface(df=df, column_name="eSSVI Vol Error Perc",
                                                                        strike_list=strike_list)
df_sabr_vol_error_surface, sabr_min, sabr_max = utils.create_surface(df=df, column_name="SABR Vol Error Perc",
                                                                        strike_list=strike_list)
vol_error_surface_min = min(svi_min, ssvi_min, essvi_min, sabr_min)
vol_error_surface_max = max(svi_max, ssvi_max, essvi_max, sabr_max)

# Compute Absolute BS Price Error Surfaces
df_svi_price_error_surface, svi_min, svi_max = utils.create_surface(df=df, column_name="SVI Price Error Perc",
                                                                    strike_list=strike_list)
df_ssvi_price_error_surface, ssvi_min, ssvi_max = utils.create_surface(df=df, column_name="SSVI Price Error Perc",
                                                                       strike_list=strike_list)
df_essvi_price_error_surface, essvi_min, essvi_max = utils.create_surface(df=df, column_name="eSSVI Price Error Perc",
                                                                          strike_list=strike_list)
df_sabr_price_error_surface, sabr_min, sabr_max = utils.create_surface(df=df, column_name="SABR Price Error Perc",
                                                                          strike_list=strike_list)
price_error_surface_min = min(svi_min, ssvi_min, essvi_min, sabr_min)
price_error_surface_max = max(svi_max, ssvi_max, essvi_max, sabr_max)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Absolute Volatility & BS Price Errors Computed ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Compute Skew
for maturity in df["Maturity"].unique():
    df_mat = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_mat["SVI Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SVI Skew']] = df[df["Maturity"] == maturity].apply(
        lambda x: calibration.SVI_skew(strike=x["Strike Perc"], forward=x["Forward Perc"],
                                       maturity=x["Maturity (in Y)"], a_=SVI_params["a_"], b_=SVI_params["b_"],
                                       rho_=SVI_params["rho_"], m_=SVI_params["m_"],
                                       sigma_=SVI_params["sigma_"]), axis=1)
    SABR_params = list(df_mat["SABR Params"])[0]
    df.loc[df["Maturity"] == maturity, ['SABR Skew']] = df[df["Maturity"] == maturity].apply(
        lambda x: calibration.SABR_skew(f=x["Forward Perc"], K=x["Strike Perc"], T=x["Maturity (in Y)"],
                                        alpha_=SABR_params["alpha_"], rho_=SABR_params["rho_"],
                                        nu_=SABR_params["nu_"]), axis=1)

df["SSVI Skew"] = df.apply(lambda x:
                           calibration.SSVI_skew(strike=x["Strike Perc"], theta=x["SVI ATMF Implied TV"],
                                                 maturity=x["Maturity (in Y)"],
                                                 forward=x["Forward Perc"], rho_=SSVI_params["rho_"],
                                                 eta_=SSVI_params["eta_"],
                                                 lambda_=SSVI_params["lambda_"]), axis=1)
df["eSSVI Skew"] = df.apply(lambda x:
                            calibration.eSSVI_skew(strike=x["Strike Perc"], theta=x["SVI ATMF Implied TV"],
                                                   maturity=x["Maturity (in Y)"],
                                                   forward=x["Forward Perc"], eta_=eSSVI_params["eta_"],
                                                   lambda_=eSSVI_params["lambda_"],
                                                   a_=eSSVI_params["a_"], b_=eSSVI_params["b_"],
                                                   c_=eSSVI_params["c_"], ), axis=1)

# Compute Skew Surfaces
df_svi_skew_surface, svi_min, svi_max = utils.create_surface(df=df, column_name="SVI Skew",
                                                             strike_list=strike_list)
df_ssvi_skew_surface, ssvi_min, ssvi_max = utils.create_surface(df=df, column_name="SSVI Skew",
                                                                strike_list=strike_list)
df_essvi_skew_surface, essvi_min, essvi_max = utils.create_surface(df=df, column_name="eSSVI Skew",
                                                                   strike_list=strike_list)
df_sabr_skew_surface, sabr_min, sabr_max = utils.create_surface(df=df, column_name="SABR Skew",
                                                                   strike_list=strike_list)
skew_surface_min = min(svi_min, ssvi_min, essvi_min, sabr_min)
skew_surface_max = max(svi_max, ssvi_max, essvi_max, sabr_min)

# Compute Call/Put Bis Mid Price (Call-Put Parity)
df["Call Bis Type"] = "Call"
df["Put Bis Type"] = "Put"
df["Call Bis Mid Perc"] = df.apply(lambda x: x["Mid Perc"] if x["Type"] == "Call" else x["Mid Perc"] + x["ZC Perc"] * (x["Forward Perc"] - x["Strike Perc"]), axis=1)
df["Put Bis Mid Perc"] = df.apply(lambda x: x["Mid Perc"] if x["Type"] == "Put" else x["Mid Perc"] - x["ZC Perc"] * (x["Forward Perc"] - x["Strike Perc"]), axis=1)

# Compute Undiscounted Call/Put Prices
df["Call Bis Mid Fwd Perc"] = df["Call Bis Mid Perc"] / df["ZC Perc"]
df["Put Bis Mid Fwd Perc"] = df["Put Bis Mid Perc"] / df["ZC Perc"]

# Compute Call/Put Bis Greeks
df = utils.compute_greeks(df=df, fwd_col="Forward Perc", vol_col="Implied Vol", strike_col="Strike Perc",
                          maturity_col="Maturity (in Y)", df_col="ZC Perc", type_col="Call Bis Type", name="Call Bis")
df = utils.compute_greeks(df=df, fwd_col="Forward Perc", vol_col="Implied Vol", strike_col="Strike Perc",
                          maturity_col="Maturity (in Y)", df_col="ZC Perc", type_col="Put Bis Type", name="Put Bis")

# Compute Skew Bounds
for maturity in df["Maturity"].unique():
    cond = (df["Maturity"] == maturity)
    df_bis = df[cond].copy()
    # S_min
    df.loc[cond, [f"s_min"]] = ((df_bis["Call Bis Mid Fwd Perc"] - df_bis["Call Bis Mid Fwd Perc"].shift(1)) / (
            df_bis["Strike Perc"] - df_bis["Strike Perc"].shift(1)) - df_bis[f"Call Bis Delta Strike"]) / \
                                                     df_bis[f"Call Bis Vega"]
    df.loc[df[cond].index[0], f"s_min"] = ((df_bis["Call Bis Mid Fwd Perc"].values[0] - 1) / (
            df_bis["Strike Perc"].values[0]) - df_bis[f"Call Bis Delta Strike"].values[0]) / \
                                                     df_bis[f"Call Bis Vega"].values[0]
    # S_max
    df.loc[cond, [f"s_max"]] = ((df_bis["Call Bis Mid Fwd Perc"].shift(-1) - df_bis["Call Bis Mid Fwd Perc"]) / (
            df_bis["Strike Perc"].shift(-1) - df_bis["Strike Perc"]) - df_bis[f"Call Bis Delta Strike"]) / \
                                                     df_bis[f"Call Bis Vega"]
    df.loc[df[cond].index[-1], f"s_max"] = - df_bis[f"Call Bis Delta Strike"].values[
        -1] / df_bis[f"Call Bis Vega"].values[-1]

# Compute Skew Bounds Spread
df["s_spread"] = df["s_max"] - df["s_min"]

# Compute Skew Bounds Surfaces
df_smin_surface, smin_min, smin_max = utils.create_surface(df=df, column_name="s_min", strike_list=strike_list)
df_smax_surface, smax_min, smax_max = utils.create_surface(df=df, column_name="s_max", strike_list=strike_list)
smin_surface_min = min(smin_min, smax_min)
smin_surface_max = max(smin_max, smax_max)

# Compute Shark-Jaw Skew Bounds Test
for surface in ["SVI", "SSVI", "eSSVI", "SABR"]:
    df[f"{surface} Bounds Test"] = df.apply(lambda x: 0 if x["s_min"] < x[f"{surface} Skew"] < x["s_max"] else 1, axis=1)

# Compute Absolute Calibration Arbitrability Scores (ACA Skew Bounds)
df_aca_skew_bounds = pd.DataFrame(index=df["Pretty Maturity"].unique(), columns=["SVI", "SSVI", "eSSVI", "SABR"])
for col in df_aca_skew_bounds.columns:
    for maturity in df["Pretty Maturity"].unique():
        df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
        df_aca_skew_bounds.loc[maturity, f"{col}"] = round((1 - df_bis[f'{col} Bounds Test'].mean()) * 10, 2)
    df_aca_skew_bounds.loc["Overall", f"{col}"] = round((1 - df[f'{col} Bounds Test'].mean()) * 10, 2)

# Compute Arbitrability Surfaces (Skew Bounds)
df_svi_bounds_arb_surface, svi_min, svi_max = utils.create_surface(df=df, column_name="SVI Bounds Test",
                                                                   strike_list=strike_list)
df_ssvi_bounds_arb_surface, ssvi_min, ssvi_max = utils.create_surface(df=df, column_name="SSVI Bounds Test",
                                                                      strike_list=strike_list)
df_essvi_bounds_arb_surface, essvi_min, essvi_max = utils.create_surface(df=df, column_name="eSSVI Bounds Test",
                                                                         strike_list=strike_list)
df_sabr_bounds_arb_surface, sabr_min, sabr_max = utils.create_surface(df=df, column_name="SABR Bounds Test",
                                                                         strike_list=strike_list)

# Compute European Binary Prices (BEUI/BEDI)
for surface in ["SVI", "SSVI", "eSSVI", "SABR"]:
    df[f"{surface} BEUI"] = df.apply(lambda x: - (x["Call Bis Delta Strike"] + x["Call Bis Vega"] * x[f"{surface} Skew"]), axis=1)
    df[f"{surface} BEDI"] = 1 - df[f"{surface} BEUI"]

# Compute Call/Put Triangles
for surface in ["SVI", "SSVI", "eSSVI", "SABR"]:
    for maturity in df["Maturity"].unique():
        cond = (df["Maturity"] == maturity)
        df_bis = df[cond].copy()
        # Call Triangles (CT)
        df.loc[cond, [f"{surface} CT"]] = df_bis["Call Bis Mid Fwd Perc"].shift(1) - df_bis["Call Bis Mid Fwd Perc"] - (
                df_bis["Strike Perc"] - df_bis["Strike Perc"].shift(1)) * df_bis[f"{surface} BEUI"]
        # Put Triangles (PT)
        df.loc[cond, [f"{surface} PT"]] = df_bis["Put Bis Mid Fwd Perc"].shift(-1) - df_bis["Put Bis Mid Fwd Perc"] - (
                df_bis["Strike Perc"].shift(-1) - df_bis["Strike Perc"]) * df_bis[f"{surface} BEDI"]

# Compute Shark-Jaw Call/Put Triangles Test
for surface in ["SVI", "SSVI", "eSSVI", "SABR"]:
    df[f"{surface} Triangles Test"] = df.apply(lambda x: 0
        if (x[f"{surface} CT"] > 0 and x[f"{surface} PT"] > 0) else 0
            if (pd.isna(x[f"{surface} CT"]) and x[f"{surface} PT"] > 0) else 0
                if (x[f"{surface} CT"] > 0 and pd.isna(x[f"{surface} PT"])) else 1, axis=1)

# Compute Absolute Calibration Arbitrability Scores (ACA Call/Put Triangles)
df_aca_call_triangles = pd.DataFrame(index=df["Pretty Maturity"].unique(), columns=["SVI", "SSVI", "eSSVI", "SABR"])
for col in df_aca_call_triangles.columns:
    for maturity in df["Pretty Maturity"].unique():
        df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
        df_aca_call_triangles.loc[maturity, f"{col}"] = round((1 - df_bis[f'{col} Triangles Test'].mean()) * 10, 2)
    df_aca_call_triangles.loc["Overall", f"{col}"] = round((1 - df[f'{col} Triangles Test'].mean()) * 10, 2)

# Compute Arbitrability Surfaces (Call/Put Triangles)
df_svi_triangles_arb_surface, svi_min, svi_max = utils.create_surface(df=df, column_name="SVI Triangles Test",
                                                                  strike_list=strike_list)
df_ssvi_triangles_arb_surface, ssvi_min, ssvi_max = utils.create_surface(df=df, column_name="SSVI Triangles Test",
                                                                     strike_list=strike_list)
df_essvi_triangles_arb_surface, essvi_min, essvi_max = utils.create_surface(df=df, column_name="eSSVI Triangles Test",
                                                                        strike_list=strike_list)
df_sabr_triangles_arb_surface, sabr_min, sabr_max = utils.create_surface(df=df, column_name="SABR Triangles Test",
                                                                        strike_list=strike_list)

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Shark-Jaw Test : Skew Bounds + Call/Put Triangles ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Create Graphs Figure
fig1, axs1 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5))
fig2, axs2 = plt.subplots(nrows=2, ncols=4, figsize=(15, 7.5))
fig3, axs3 = plt.subplots(nrows=2, ncols=4, figsize=(15, 7.5), sharey=True)
fig4, axs4 = plt.subplots(nrows=2, ncols=3, figsize=(15, 7.5), sharey=True)
fig5, axs5 = plt.subplots(nrows=2, ncols=4, figsize=(15, 7.5), sharey=True)

# Set Graphs Figure Title
fig1.suptitle(f"Market Data ({spot_date.strftime('%d-%b-%Y')})",
              fontweight='bold', fontsize=12.5)
fig2.suptitle(f"Calibration Results ({spot_date.strftime('%d-%b-%Y')})",
              fontweight='bold', fontsize=12.5)
fig3.suptitle(f"Calibrated Vol. Errors ({spot_date.strftime('%d-%b-%Y')})",
              fontweight='bold', fontsize=12.5)
fig4.suptitle(f"Calibrated Vol. Skew ({spot_date.strftime('%d-%b-%Y')})",
              fontweight='bold', fontsize=12.5)
fig5.suptitle(f"Shark-Jaw Arbitograms ({spot_date.strftime('%d-%b-%Y')})",
              fontweight='bold', fontsize=12.5)

# Plot Fig1: Number of Options Per Steps
nb_options_text.append("Calibration")
nb_options_nr.append(nb_options_nr[-1])
nb_options_brent.append(nb_options_brent[-1])
axs1[0, 0].step(nb_options_text, nb_options_nr, "--", where='post', label="Newton Raphson")
axs1[0, 0].step(nb_options_text, nb_options_brent, where='post', label="Brent")
axs1[0, 0].set_title("Option Number Per Steps", fontsize=title_font_size)
axs1[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[0, 0].legend(loc=legend_loc)
axs1[0, 0].grid()

# Plot Fig1: Implied Forward & ZC
maturity_list = []
forward_list = []
zc_list = []
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    forward_list.append(list(df_bis["Forward Perc"].unique())[0])
    zc_list.append(list(df_bis["ZC Perc"].unique())[0])
    maturity_list.append(maturity)
axs1[1, 0].plot(maturity_list, forward_list, label="Forward (in %)")
axs1[1, 0].plot(maturity_list, zc_list, label="ZC")
axs1[1, 0].set_title(f"Market Implied Forward & ZC", fontsize=title_font_size)
axs1[1, 0].legend(loc="upper right")
axs1[1, 0].grid()
axs1[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 0].xaxis.set_major_formatter(DateFormatter("%b-%y"))
axs1[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Plot Fig1: Implied Total Variance
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

# Plot Fig1: Market Implied Volatilities
for maturity in df["Pretty Maturity"].unique():
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    axs1[1, 1].plot(df_bis["Strike"], df_bis["Implied Vol"], label=maturity)
axs1[1, 1].scatter(df["Strike"], df["IVM"] / 100, marker=".", color="black", label="BBG IV")
axs1[1, 1].grid()
axs1[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 1].set_title("Market Implied Volatilities", fontsize=title_font_size)
axs1[1, 1].legend(loc=legend_loc)

# Plot Fig1: Market Implied Delta Strike
for maturity, i in zip(df["Pretty Maturity"].unique(), range(0, len(df["Pretty Maturity"].unique()))):
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    df_put_bis = df_bis[(df_bis["Type"] == "Put")].copy()
    df_call_bis = df_bis[(df_bis["Type"] == "Call")].copy()
    axs1[0, 2].plot(df_put_bis["Strike"], df_put_bis["Implied Delta Strike"], label=maturity, color=color_list[i])
    axs1[0, 2].plot(df_call_bis["Strike"], df_call_bis["Implied Delta Strike"], color=color_list[i])
axs1[0, 2].grid()
axs1[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[0, 2].set_title("Market Implied Deltas Strike", fontsize=title_font_size)
axs1[0, 2].legend(loc=legend_loc)

# Plot Fig1: Market Implied Vegas
for maturity in df["Pretty Maturity"].unique():
    df_bis = df[(df["Pretty Maturity"] == maturity)].copy()
    df_bis = df_bis.sort_values(by="Strike", ascending=False)
    axs1[1, 2].plot(df_bis["Strike"], df_bis["Implied Vega"], label=maturity)
axs1[1, 2].grid()
axs1[1, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs1[1, 2].set_title("Market Implied Vegas", fontsize=title_font_size)
axs1[1, 2].legend(loc=legend_loc)

# Plot Fig2: SVI Calibration (Implied TV)
k_list = np.linspace(log_forward_moneyness_min, log_forward_moneyness_max, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_bis["SVI Params"])[0]
    tv_list = [calibration.SVI(k=k, a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"],
                                m_=SVI_params["m_"], sigma_=SVI_params["sigma_"]) for k in k_list]
    axs2[0, 0].plot(k_list, tv_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 0].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 0].grid()
axs2[0, 0].legend(loc=legend_loc)
axs2[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 0].set_title("SVI Implied TV", fontsize=title_font_size)

# Plot Fig2: SSVI Calibration (Implied TV)
k_list = np.linspace(log_forward_moneyness_min, log_forward_moneyness_max, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["SVI ATMF Implied TV"])[0]
    tv_list = [calibration.SSVI(k=k, theta=theta, rho_=SSVI_params["rho_"], eta_=SSVI_params["eta_"],
                                  lambda_=SSVI_params["lambda_"]) for k in k_list]
    axs2[0, 1].plot(k_list, tv_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 1].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 1].grid()
axs2[0, 1].legend(loc=legend_loc)
axs2[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 1].set_title("SSVI Implied TV", fontsize=title_font_size)

# Plot Fig2: eSSVI Calibration (Implied TV)
k_list = np.linspace(log_forward_moneyness_min, log_forward_moneyness_max, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["SVI ATMF Implied TV"])[0]
    tv_list = [
        calibration.eSSVI(k=k, theta=theta, a_=eSSVI_params["a_"], b_=eSSVI_params["b_"], c_=eSSVI_params["c_"],
                          eta_=eSSVI_params["eta_"], lambda_=eSSVI_params["lambda_"]) for k in k_list]
    axs2[0, 2].plot(k_list, tv_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 2].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 2].grid()
axs2[0, 2].legend(loc=legend_loc)
axs2[0, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 2].set_title("eSSVI Implied TV", fontsize=title_font_size)

# Plot Fig2: SABR Calibration (Implied TV)
k_list = np.linspace(log_forward_moneyness_min, log_forward_moneyness_max, 400)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SABR_params = list(df_bis["SABR Params"])[0]
    maturity = list(df_bis["Maturity (in Y)"])[0]
    forward = list(df_bis["Forward Perc"])[0]
    tv_list = [pow(calibration.SABR(f=forward, K=forward * np.exp(k), T=maturity, alpha_=SABR_params["alpha_"],
                                rho_=SABR_params["rho_"], nu_=SABR_params["nu_"]), 2) * maturity for k in k_list]
    axs2[0, 3].plot(k_list, tv_list, label=list(df_bis["Pretty Maturity"])[0])
    axs2[0, 3].scatter(list(df_bis["Log Forward Moneyness"]), list(df_bis["Implied TV"]), marker="+")
axs2[0, 3].grid()
axs2[0, 3].legend(loc=legend_loc)
axs2[0, 3].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[0, 3].set_title("SABR Implied TV", fontsize=title_font_size)

# Plot Fig2: SVI Calibration (Durrleman Condition)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SVI_params = list(df_bis["SVI Params"])[0]
    k_list, g_list = calibration.SVI_Durrleman_Condition(
        a_=SVI_params["a_"], b_=SVI_params["b_"], rho_=SVI_params["rho_"], m_=SVI_params["m_"],
        sigma_=SVI_params["sigma_"])
    axs2[1, 0].plot(k_list, g_list, label=list(df_bis["Pretty Maturity"])[0])
axs2[1, 0].grid()
axs2[1, 0].legend(loc=legend_loc)
axs2[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 0].set_title("SVI Durrleman Condition", fontsize=title_font_size)

# Plot Fig2: SSVI Calibration (Durrleman Condition)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["SVI ATMF Implied TV"])[0]
    k_list, g_list = calibration.SSVI_Durrleman_Condition(
        theta=theta, rho_=SSVI_params["rho_"], eta_=SSVI_params["eta_"], lambda_=SSVI_params["lambda_"])
    axs2[1, 1].plot(k_list, g_list, label=list(df_bis["Pretty Maturity"])[0])
axs2[1, 1].grid()
axs2[1, 1].legend(loc=legend_loc)
axs2[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 1].set_title("SSVI Durrleman Condition", fontsize=title_font_size)

# Plot Fig2: eSSVI Calibration (Durrleman Condition)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    theta = list(df_bis["SVI ATMF Implied TV"])[0]
    k_list, g_list = calibration.eSSVI_Durrleman_Condition(
        theta=theta, a_=eSSVI_params["a_"], b_=eSSVI_params["b_"], c_=eSSVI_params["c_"],
        eta_=eSSVI_params["eta_"], lambda_=eSSVI_params["lambda_"])
    axs2[1, 2].plot(k_list, g_list, label=list(df_bis["Pretty Maturity"])[0])
axs2[1, 2].grid()
axs2[1, 2].legend(loc=legend_loc)
axs2[1, 2].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 2].set_title("eSSVI Durrleman Condition", fontsize=title_font_size)

# Plot Fig2: SABR Calibration (Durrleman Condition)
for maturity in df["Maturity"].unique():
    df_bis = df[(df["Maturity"] == maturity)].copy()
    SABR_params = list(df_bis["SABR Params"])[0]
    maturity = list(df_bis["Maturity (in Y)"])[0]
    forward = list(df_bis["Forward Perc"])[0]
    k_list, g_list = calibration.SABR_Durrleman_Condition(f=forward, T=maturity, alpha_=SABR_params["alpha_"],
                                                          rho_=SABR_params["rho_"], nu_=SABR_params["nu_"])
    axs2[1, 3].plot(k_list, g_list, label=list(df_bis["Pretty Maturity"])[0])
axs2[1, 3].grid()
axs2[1, 3].legend(loc=legend_loc)
axs2[1, 3].tick_params(axis='both', which='major', labelsize=tick_font_size)
axs2[1, 3].set_title("SABR Durrleman Condition", fontsize=title_font_size)

# Plot Fig3: Calibrated Volatility Error Surfaces
g1 = sns.heatmap(df_svi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 0], cbar=False, annot=True,
                 vmin=vol_error_surface_min, vmax=vol_error_surface_max)
g2 = sns.heatmap(df_ssvi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 1], cbar=False, annot=True,
                 vmin=vol_error_surface_min, vmax=vol_error_surface_max)
g3 = sns.heatmap(df_essvi_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 2], cbar=False, annot=True,
                 vmin=vol_error_surface_min, vmax=vol_error_surface_max)
g4 = sns.heatmap(df_sabr_vol_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[0, 3], cbar=True, annot=True,
                 vmin=vol_error_surface_min, vmax=vol_error_surface_max)
for g, ax, name in zip([g1, g2, g3, g4], [axs3[0, 0], axs3[0, 1], axs3[0, 2], axs3[0, 3]], ["SVI", "SSVI", "eSSVI", "SABR"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Vol. Error (in %)", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Fig3: Calibrated Price Error Surfaces
g1 = sns.heatmap(df_svi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 0], cbar=False, annot=True,
                 vmin=price_error_surface_min, vmax=price_error_surface_max)
g2 = sns.heatmap(df_ssvi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 1], cbar=False, annot=True,
                 vmin=price_error_surface_min, vmax=price_error_surface_max)
g3 = sns.heatmap(df_essvi_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 2], cbar=False, annot=True,
                 vmin=price_error_surface_min, vmax=price_error_surface_max)
g4 = sns.heatmap(df_sabr_price_error_surface.values, linewidths=1, cmap='Blues', ax=axs3[1, 3], cbar=True, annot=True,
                 vmin=price_error_surface_min, vmax=price_error_surface_max)
for g, ax, name in zip([g1, g2, g3, g4], [axs3[1, 0], axs3[1, 1], axs3[1, 2], axs3[1, 3]], ["SVI", "SSVI", "eSSVI", "SABR"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} BS Price Error (in %)", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Fig4: Calibrated Volatility Skew Surfaces (SVI, SSVI, Min)
skew_min = min(skew_surface_min, smin_surface_min)
skew_max = max(skew_surface_max, smin_surface_max)
g1 = sns.heatmap(df_svi_skew_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 0], cbar=False, annot=True,
                 vmin=skew_min, vmax=skew_max)
g2 = sns.heatmap(df_ssvi_skew_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 1], cbar=False, annot=True,
                 vmin=skew_min, vmax=skew_max)
g3 = sns.heatmap(df_smin_surface.values, linewidths=1, cmap='Blues', ax=axs4[0, 2], cbar=True, annot=True,
                 vmin=skew_min, vmax=skew_max)
for g, ax, name in zip([g1, g2, g3], [axs4[0, 0], axs4[0, 1], axs4[0, 2]], ["SVI", "SSVI", "Min Bound"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name}", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Fig4: Calibrated Volatility Skew Surfaces (eSSVI, SABR, Max)
g1 = sns.heatmap(df_essvi_skew_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 0], cbar=False, annot=True,
                 vmin=skew_min, vmax=skew_max)
g2 = sns.heatmap(df_sabr_skew_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 1], cbar=False, annot=True,
                 vmin=skew_min, vmax=skew_max)
g3 = sns.heatmap(df_smax_surface.values, linewidths=1, cmap='Blues', ax=axs4[1, 2], cbar=True, annot=True,
                 vmin=skew_min, vmax=skew_max)
for g, ax, name in zip([g1, g2, g3], [axs4[1, 0], axs4[1, 1], axs4[1, 2]], ["eSSVI", "SABR", "Max Bound"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name}", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Fig5: Shark-Jaw Arbitogram Surfaces (Skew Bounds)
g1 = sns.heatmap(df_svi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[0, 0], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g2 = sns.heatmap(df_ssvi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[0, 1], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g3 = sns.heatmap(df_essvi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[0, 2], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g4 = sns.heatmap(df_sabr_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[0, 3], cbar=False, annot=True,
                 vmin=-1, vmax=1)
for g, ax, name in zip([g1, g2, g3, g4], [axs5[0, 0], axs5[0, 1], axs5[0, 2], axs5[0, 3]], ["SVI", "SSVI", "eSSVI", "SABR"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Skew Bounds", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Plot Fig5: Shark-Jaw Arbitogram Surfaces (Call/Put Triangles)
g1 = sns.heatmap(df_svi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[1, 0], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g2 = sns.heatmap(df_ssvi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[1, 1], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g3 = sns.heatmap(df_essvi_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[1, 2], cbar=False, annot=True,
                 vmin=-1, vmax=1)
g4 = sns.heatmap(df_sabr_bounds_arb_surface.values, linewidths=1, cmap='Blues', ax=axs5[1, 3], cbar=False, annot=True,
                 vmin=-1, vmax=1)
for g, ax, name in zip([g1, g2, g3, g4], [axs5[1, 0], axs5[1, 1], axs5[1, 2], axs5[1, 3]], ["SVI", "SSVI", "eSSVI", "SABR"]):
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.set_title(f"{name} Call/Put Triangles", fontsize=title_font_size)
    g.set_ylabel('')
    g.set_xlabel('')
    g.set_xticklabels(df["Pretty Maturity"].unique(), rotation=0)
    g.set_yticklabels([f"{int(strike / spot * 100)}%" for strike in strike_list], rotation=0)

# Export Dataframes in Results Folder
if not os.path.exists('results'):
    os.makedirs('results')
with pd.ExcelWriter("results/Results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Dataframe")
    df_aca_skew_bounds.to_excel(writer, sheet_name="ACA (Bounds)")
    df_aca_call_triangles.to_excel(writer, sheet_name="ACA (Triangles)")
    fig1.savefig('results/1. Market Data.png')
    fig2.savefig('results/2. Calibration Results.png')
    fig3.savefig('results/3. Calibrated Vol. Errors.png')
    fig4.savefig('results/4. Calibrated Vol. Skew.png')
    fig5.savefig('results/5. Shark-Jaw Arbitograms.png')

# Timer
end = time.perf_counter()
print(f"{timer_id}/ Results Exported + Graphs Built ({round(end - start, 1)}s)")
start = end
timer_id = timer_id + 1

# Display ACA Scores
print("\nACA Scores (Skew Bounds) :")
print(df_aca_skew_bounds)
print("\nACA Scores (Call/Put Triangles) :")
print(df_aca_call_triangles)

# Display All Figures
plt.show()
