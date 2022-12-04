import pandas as pd

import black_scholes

def create_surface(df, column_name, strike_list=None):
    df_list = []
    for maturity in df["Maturity"].unique():
        if strike_list:
            df_mat = df[(df["Maturity"] == maturity) & (df["Strike"].isin(strike_list))].copy()
        else:
            df_mat = df[(df["Maturity"] == maturity)].copy()
        df_mat.index = df_mat["Strike"]
        df_mat = df_mat[[column_name]]
        df_mat.columns = [maturity]
        df_list.append(df_mat)
    new_df = pd.concat(df_list, axis=1)
    new_df.sort_index(inplace=True)
    return new_df, min(new_df.min()), max(new_df.max())

def compute_greeks(df, fwd_col, vol_col, strike_col, maturity_col, df_col, type_col, name):
    df[f"{name} Delta Strike"] = df.apply(
        lambda x: black_scholes.BS_Delta_Strike(f=x[fwd_col], k=x[strike_col], t=x[maturity_col],
                                                v=x[vol_col], df=x[df_col], OptType=x[type_col][0]), axis=1)
    df[f"{name} Vega"] = df.apply(
        lambda x: black_scholes.BS_Vega(f=x[fwd_col], k=x[strike_col], t=x[maturity_col],
                                        v=x[vol_col], df=x[df_col], OptType=x[type_col][0]), axis=1)
    df[f"{name} d1"] = df.apply(
        lambda x: black_scholes.BS_d1(f=x[fwd_col], k=x[strike_col], t=x[maturity_col],
                                      v=x[vol_col]), axis=1)
    df[f"{name} d2"] = df.apply(
        lambda x: black_scholes.BS_d2(f=x[fwd_col], k=x[strike_col], t=x[maturity_col],
                                      v=x[vol_col]), axis=1)
    return df