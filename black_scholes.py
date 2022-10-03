import numpy as np


MAX_ITERS = 10000
MAX_ERROR = pow(10, -30)
EPS = 0.01
MAX_GAUSSIAN_VALUE = 13.190905958273
MATH_SQR2PI = 2.50662827


def BS_d1(f, k, t, v):
    return (np.log(f / k) + v * v * t / 2) / v / np.sqrt(t)

def BS_d2(f, k, t, v):
    return BS_d1(f, k, t, v) - v * np.sqrt(t)

def NormalDistrib(z, mean=0, stdev=1):
    if abs(z) > MAX_GAUSSIAN_VALUE:
        return 0
    else:
        return np.exp(-0.5 * ((z - mean) / stdev) ** 2) / MATH_SQR2PI / stdev

def SNorm(z):
    c1 = 2.506628
    c2 = 0.31938153
    c3 = -0.3565638
    c4 = 1.781477937
    c5 = -1.821255978
    c6 = 1.330274429
    if abs(z) > MAX_GAUSSIAN_VALUE:
        return 1 if z > 0 else 0
    else:
        w = -1 if z < 0 else 1
        y = 1 / (1 + 0.2316419 * w * z)
        x = y * (c3 + y * (c4 + y * c5 + y * c6))
        return 0.5 + w * (0.5 - (np.exp(-z * z / 2) / c1) * (y * (c2 + x)))

def BS_Price(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    swticher = {
        "C": df * (f * SNorm(d1) - k * SNorm(d2)),
        "P": df * (-f * SNorm(-d1) + k * SNorm(-d2)),
        "C+P": df * (f * SNorm(d1) - SNorm(-d1)) - k * (SNorm(d2) - SNorm(-d2)),
        "C-P": df * (f * SNorm(d1) + SNorm(-d1)) - k * (SNorm(d2) + SNorm(-d2)),
    }
    return swticher.get(OptType.upper(), 0)

def BS_Delta(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    swticher = {
        "C": df * SNorm(d1),
        "P": -df * SNorm(-d1),
        "C+P": df * (SNorm(d1) - SNorm(-d1)),
        "C-P": df * (SNorm(d1) + SNorm(-d1)),
    }
    return swticher.get(OptType.upper(), 0)

def BS_Theta(f, k, t, v, df, r, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    swticher = {
        "C": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) - (r * f * SNorm(d1)) + (r * k * SNorm(d2))),
        "P": -df * ((f * SNorm(d1) * v) / (2 * np.sqrt(t)) + (r * f * SNorm(-d1)) - (r * k * SNorm(-d2))),
        "C+P": BS_Theta(f, k, t, v, df, r, "C") + BS_Theta(f, k, t, v, df, r, "P"),
        "C-P": BS_Theta(f, k, t, v, df, r, "C") - BS_Theta(f, k, t, v, df, r, "P"),
    }
    return swticher.get(OptType.upper(), 0)

def BS_Gamma(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    swticher = {
        "C": df * fd1 / (f * v * np.sqrt(t)),
        "P": df * fd1 / (f * v * np.sqrt(t)),
        "C+P": 2 * df * fd1 / (f * v * np.sqrt(t)),
        "C-P": 0,
    }
    return swticher.get(OptType.upper(), 0)

def BS_Vega(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    fd1 = NormalDistrib(d1)
    swticher = {
        "C": df * f * fd1 * np.sqrt(t),
        "P": df * f * fd1 * np.sqrt(t),
        "C+P": 2 * df * f * fd1 * np.sqrt(t),
        "C-P": 0,
    }
    return swticher.get(OptType.upper(), 0)

def BS_Vanna(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    swticher = {
        "C": -df * fd1 * d2 / v,
        "P": -df * fd1 * d2 / v,
        "C+P": -2 * df * fd1 * d2 / v,
        "C-P": 0,
    }
    return swticher.get(OptType.upper(), 0)

def BS_Volga(f, k, t, v, df, OptType):
    d1 = BS_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = NormalDistrib(d1)
    swticher = {
        "C": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "P": df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C+P": 2 * df * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "C-P": 0,
    }
    return swticher.get(OptType.upper(), 0)

def BS_ImpliedVol(f, k, t, MktPrice, df, OptType):
    nb_iter = 0
    v = 0.3
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
    return MktPrice / BS_Price(f, k, t, v, 1, OptType)

def BS_ImpliedForward(MktPrice, k, t, v, df, OptType):
    nb_iter = 0
    f = k
    func = MktPrice - BS_Price(f, k, t, v, df, OptType)
    while abs(func) > MAX_ERROR and nb_iter < MAX_ITERS:
        Deriv = (BS_Price(f, k + EPS, t, v, df, OptType) - BS_Price(f, k, t, v, df, OptType)) / EPS
        f = f + func / Deriv
        func = MktPrice - BS_Price(f, k, t, v, df, OptType)
        nb_iter = nb_iter + 1
    return f
