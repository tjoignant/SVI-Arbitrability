import numpy as np
from scipy.stats import norm

MAX_ITERS = 10000
MAX_ERROR = pow(10, -6)
EPS = 0.01


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


def BS_ImpliedVol(f, k, t, MktPrice, df, OptType):
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
            return -1
        else:
            v = v + func / veg
            func = MktPrice - BS_Price(f, k, t, v, df, OptType)
            nb_iter = nb_iter + 1
    return v


def BS_ImpliedStrike(f, MktPrice, t, df, v, OptType):
    """
    :param f: Forward (in %)
    :param t: Maturity (in Years)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param v: Constant Annual Volatility (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Strike
    """
    nb_iter = 0
    k = f
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        Deriv = (BS_Price(f, k + EPS, t, v, df, OptType) - BS_Price(f, k, t, v, df, OptType)) / EPS
        k = k + func / Deriv
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        nb_iter = nb_iter + 1
    return k


def BS_ImpliedMaturity(f, k, MktPrice, v, df, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param MktPrice: Option's Market Price (in %)
    :param df: Discount Factor (in %)
    :param v: Constant Annual Volatility (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Maturity
    """
    nb_iter = 0
    t = 5
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        Deriv = (BS_Price(f, k + EPS, t, v, df, OptType) - BS_Price(f, k, t, v, df, OptType)) / EPS
        t = t + func / Deriv
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        nb_iter = nb_iter + 1
    return t


def BS_ImpliedDiscFactor(f, k, t, v, MktPrice, OptType):
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param MktPrice: Option's Market Price (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Discount Factor
    """
    return MktPrice / BS_Price(f, k, t, v, 1, OptType)


def BS_ImpliedForward(MktPrice, k, t, v, df, OptType):
    """
    :param MktPrice: Option's Market Price (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param df: Discount Factor (in %)
    :param OptType: Either "C", "P", "C+P" or "C-P"
    :return: Implied Forward
    """
    nb_iter = 0
    f = k
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        Deriv = (BS_Price(f, k + EPS, t, v, df, OptType) - BS_Price(f, k, t, v, df, OptType)) / EPS
        f = f + func / Deriv
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        nb_iter = nb_iter + 1
    return f
