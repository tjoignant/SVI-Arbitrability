import numpy as np
from scipy.stats import norm

MAX_ITERS = 1000
MAX_ERROR = pow(10, -6)


def BS_d1(f: float, k: float, t: float, v: float):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :return: d1
    """
    return (np.log(f / k) + v * v * t / 2) / v / np.sqrt(t)


def BS_d2(f: float, k: float, t: float, v: float):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :return: d2
    """
    return BS_d1(f, k, t, v) - v * np.sqrt(t)


def NormalDistrib(z: float, mean=0, stdev=1):
    """
    :param z: Datapoint
    :param mean: Normal Distribution Expectation
    :param stdev: Normal Distribution Standard Deviation
    :return: Datapoint Normal Value
    """
    return norm.pdf(z, loc=mean, scale=stdev)


def SNorm(z: float):
    """
    :param z: Datapoint
    :return: Datapoint Cumulative Normal Value
    """
    return norm.cdf(z)


def BS_Price(f: float, k: float, t: float, v: float, df: float, OptType: str):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Price
    """
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "C": df * (f * SNorm(d1) - k * SNorm(d2)),
        "P": df * (-f * SNorm(-d1) + k * SNorm(-d2)),
        "C+P": df * (f * SNorm(d1) - SNorm(-d1)) - k * (SNorm(d2) - SNorm(-d2)),
        "C-P": df * (f * SNorm(d1) + SNorm(-d1)) - k * (SNorm(d2) + SNorm(-d2)),
    }
    return switcher.get(OptType.upper(), 0)


def BS_Delta(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Delta
    """
    d1 = BS_d1(f, k, t, v)
    switcher = {
        "C": df * SNorm(d1),
        "P": -df * SNorm(-d1),
        "C+P": df * (SNorm(d1) - SNorm(-d1)),
        "C-P": df * (SNorm(d1) + SNorm(-d1)),
    }
    return switcher.get(OptType.upper(), 0)


def BS_Delta_Strike(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Delta
    """
    d2 = BS_d1(f, k, t, v)
    switcher = {
        "C": -df * SNorm(d2),
        "P": df * SNorm(-d2),
        "C+P": df * (-SNorm(d2) + SNorm(-d2)),
        "C-P": df * (-SNorm(d2) - SNorm(-d2)),
    }
    return switcher.get(OptType.upper(), 0)


def BS_Theta(f, k, t, v, df, r, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param r: Annual Risk Free Rate (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Theta
    """
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "C": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) - (r * f * SNorm(d1)) + (r * k * SNorm(d2))),
        "P": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) + (r * f * SNorm(-d1)) - (r * k * SNorm(-d2))),
        "C+P": BS_Theta(f, k, t, v, df, r, "C") + BS_Theta(f, k, t, v, df, r, "P"),
        "C-P": BS_Theta(f, k, t, v, df, r, "C") - BS_Theta(f, k, t, v, df, r, "P"),
    }
    return switcher.get(OptType.upper(), 0)


def BS_Gamma(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Gamma
    """
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": df * fd1 / (f * v * np.sqrt(t)),
        "P": df * fd1 / (f * v * np.sqrt(t)),
        "C+P": 2 * df * fd1 / (f * v * np.sqrt(t)),
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)


def BS_Vega(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Vega
    """
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": df * f * fd1 * np.sqrt(t),
        "P": df * f * fd1 * np.sqrt(t),
        "C+P": 2 * df * f * fd1 * np.sqrt(t),
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)


def BS_Vanna(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Vanna
    """
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": -df * fd1 * d2 / v,
        "P": -df * fd1 * d2 / v,
        "C+P": -2 * df * fd1 * d2 / v,
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)


def BS_Volga(f, k, t, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Volga
    """
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    switcher = {
        "C": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "P": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C+P": 2 * df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C-P": 0,
    }
    return switcher.get(OptType.upper(), 0)


def BS_IV_Dichotomy(f, k, t, MktPrice, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Volatility
    """
    nb_iter = 0
    v_list = [0.01, 1]
    v = (v_list[0] + v_list[1]) / 2
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        v = (v_list[0] + v_list[1]) / 2
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        if func > 0:
            v_list[0] = v
        else:
            v_list[1] = v
        nb_iter = nb_iter + 1
    return v, nb_iter


def BS_IV_Newton_Raphson(f, k, t, MktPrice, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Volatility
    """
    nb_iter = 0
    v = 0.30
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        veg = BS_Vega(f, k, t, v, df, OptType)
        if veg == 0:
            return -1, nb_iter
        else:
            v = v + func / veg
            func = MktPrice - BS_Price(f, k, t, v, df, OptType)
            nb_iter = nb_iter + 1
    return v, nb_iter


def BS_IV_Brent(f, k, t, MktPrice, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Volatility using Brent method
    """
    # Initialisation
    nb_iter = 0
    v_a = 0.01
    v_b = 1
    func_a = MktPrice - BS_Price(f, k, t, v_a, df, OptType)
    func_b = MktPrice - BS_Price(f, k, t, v_b, df, OptType)
    if func_a * func_b >= 0:
        return -1, nb_iter
    else:
        if abs(func_a) < abs(func_b):
            v_a, v_b = v_b, v_a
        v_c = v_a
        v_s = v_c
        v_d = v_c
        func_c = func_a
        func_s = func_c
        mflag = True

        # Iterations
        while abs(func_b) > MAX_ERROR and nb_iter < MAX_ITERS:

            if func_a != func_c and func_b != func_c:
                # Inverse Quadratic Interpolation
                a = (v_a * func_b * func_c) / ((func_a - func_b) * (func_a - func_c))
                b = (v_b * func_a * func_c) / ((func_b - func_a) * (func_b - func_c))
                c = (v_c * func_a * func_b) / ((func_c - func_a) * (func_c - func_b))
                v_s = a + b + c
            else:
                # Secant Method
                v_s = v_b - func_b * (v_b - v_a) / (func_b - func_a)

            # Bisection Method
            if not ((3 * v_a + v_b) / 4 <= v_s <= v_b) or (
                    mflag == True and abs(v_s - v_b) >= (abs(v_b - v_c) / 2)) or (
                    mflag == False and abs(v_s - v_b) >= abs(v_c - v_d) / 2) or (
                    mflag == True and abs(v_b - v_c) < abs(MAX_ERROR)) or (
                    mflag == False and abs(v_c - v_d) < abs(MAX_ERROR)):
                v_s = (v_a + v_b) / 2
                mflag = True
            else:
                mflag = False

            v_d, v_c = v_c, v_b
            func_s = MktPrice - BS_Price(f, k, t, v_s, df, OptType)
            func_a = MktPrice - BS_Price(f, k, t, v_a, df, OptType)

            if func_a * func_s < 0:
                v_b = v_s
            else:
                v_a = v_s

            func_a = MktPrice - BS_Price(f, k, t, v_a, df, OptType)
            func_b = MktPrice - BS_Price(f, k, t, v_b, df, OptType)
            func_c = MktPrice - BS_Price(f, k, t, v_c, df, OptType)

            if abs(func_a) < abs(func_b):
                v_a, v_b = v_b, v_a

            nb_iter = nb_iter + 1

        v_result = v_b if abs(func_b) < abs(func_s) else v_s
        return v_result, nb_iter


def BS_IV_Nelder_Mead_1D(f, k, t, MktPrice, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Volatility
    """
    # Initialisation
    nb_iter = 0
    x_list = [0.01, 1]
    fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
    # Sorting
    if fx_list[1] < fx_list[0]:
        temp = x_list[0]
        x_list[0] = x_list[1]
        x_list[1] = temp
    fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
    while fx_list[0] > MAX_ERROR and nb_iter < MAX_ITERS:
        # Reflexion
        xr = x_list[0] + (x_list[0] - x_list[1])
        fxr = abs(BS_Price(f, k, t, xr, df, OptType) - MktPrice)
        # Expansion
        if fxr < fx_list[0]:
            xe = x_list[0] + 2 * (x_list[0] - x_list[1])
            fxe = abs(BS_Price(f, k, t, xe, df, OptType) - MktPrice)
            if fxe <= fxr:
                x_list = [xe, x_list[0]]
            else:
                x_list = [xr, x_list[0]]
        # Contraction
        else:
            x_list = [x_list[0], 0.5 * (x_list[0] + x_list[1])]
        # Recompute Each Error
        fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
        # Sorting X List
        if fx_list[1] < fx_list[0]:
            temp = x_list[0]
            x_list[0] = x_list[1]
            x_list[1] = temp
        fx_list = [abs(BS_Price(f, k, t, x, df, OptType) - MktPrice) for x in x_list]
        # Add Nb Iter
        nb_iter = nb_iter + 1
    return x_list[0], nb_iter
