import math
import numpy as np
import scipy.optimize as optimize


def SVI(k: float, a_: float, b_: float, rho_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness
    :param a_: adjusts the vertical deplacement of the smile
    :param b_: adjust the angle between left and right asymptotes
    :param rho_: adjust the orientation of the graph
    :param m_: adjusts the horizontal deplacement of the smile
    :param sigma_: adjusts the smoothness of the vertex
    :return: total variance
    """
    return a_ + b_ * (rho_ * (k - m_) + math.sqrt(pow(k - m_, 2) + pow(sigma_, 2)))


def SVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param params_list: parameters list : [a_, b_, rho_, m_, sigma_]
    :param inputs_list: inputs list : [(k_1), (k_2), (k_3), ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: mean squared volatility error (MSVE)
    """
    MSVE = 0
    for i in range(0, len(inputs_list)):
        MSVE = MSVE + weights_list[i] * pow(
            SVI(k=inputs_list[i][0], a_=params_list[0], b_=params_list[1], rho_=params_list[2], m_=params_list[3],
                sigma_=params_list[4]) - mktTotVar_list[i], 2)
    return MSVE / len(inputs_list)


def SVI_calibration(k_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param k_list: inputs list : [k_1, k_2, k_3, ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: calibrated parameters list
    """
    init_params_list = [0.01, 0.30, 0, 0, 0.01]
    inputs_list = [(k,) for k in k_list]
    result = optimize.minimize(
        SVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list)
    )
    final_params = list(result.x)
    return {
        "a_": final_params[0],
        "b_": final_params[1],
        "rho_": final_params[2],
        "m_": final_params[3],
        "sigma_": final_params[4],
    }


def SVI_skew(strike: float, forward: float, maturity: float, a_: float, b_: float, rho_: float, m_: float, sigma_: float):
    num = b_ * ((np.log(strike/forward) - m_) / (np.sqrt(pow(np.log(strike/forward) - m_, 2) + pow(sigma_, 2))) + rho_)
    den = 2 * maturity * strike * np.sqrt((a_ + b_ * (np.sqrt(pow(np.log(strike/forward) - m_, 2) + pow(sigma_, 2)) + rho_ * np.log(strike/forward) - m_ * rho_)) / maturity)
    return num / den


def SVI_convexity(strike: float, forward: float, maturity: float, a_: float, b_: float, rho_: float, m_: float,
                  sigma_: float):
    num1 = pow(np.log(strike / forward) - m_, 2)
    num1 = -num1
    den1 = pow(strike, 2) * pow(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2), 3 / 2)

    num2 = np.log(strike / forward) - m_
    num2 = -num2
    den2 = pow(strike, 2) * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2))

    num3 = 1
    den3 = pow(strike, 2) * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2))

    num4 = -rho_
    den4 = pow(strike, 2)

    dentot = 2 * maturity * np.sqrt((a_ + b_*(np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)) + rho_*(
        np.log(strike / forward) - m_))) / maturity)

    firstterm = (b_ * ((num1 / den1) + (num2 / den2) + (num3 / den3) + (num4 / den4))) / dentot

    num5 = pow(b_, 2) * pow(((np.log(strike / forward) - m_) / (
                strike * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)))) + rho_ / strike, 2)
    den5 = 4 * pow(maturity, 2)
    pow((a_ + b_*(np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)) + rho_ * (
                np.log(strike / forward) - m_))) / maturity, 3 / 2)

    secondterm = num5 / den5

    return firstterm - secondterm


def SSVI_phi(theta: float, eta_: float, lambda_: float):
    """
    :param theta: ATM total variance
    :param eta_: curvature function parameter
    :param lambda_: curvature function parameter
    :return: curvature function result
    """
    return eta_ * pow(theta, -lambda_)


def SSVI(k: float, theta: float, rho_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness
    :param theta: ATM total variance
    :param rho_: spot vol constant correlation
    :param eta_: curvature function parameter
    :param lambda_: curvature function parameter
    :return: total variance
    """
    return 0.5 * theta * (
            1 + rho_ * SSVI_phi(theta, eta_, lambda_) * k +
            np.sqrt(pow(SSVI_phi(theta, eta_, lambda_) * k + rho_, 2) + 1 - pow(rho_, 2)))


def SSVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param params_list: parameters list : [rho_, eta_, lambda_]
    :param inputs_list: inputs list : [(k_1, theta_1), (k_2, theta_2), (k_3, theta_3), ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: mean squared volatility error (MSVE)
    """
    MSVE = 0
    for i in range(0, len(inputs_list)):
        MSVE = MSVE + weights_list[i] * pow(
            SSVI(k=inputs_list[i][0], theta=inputs_list[i][1],
                 rho_=params_list[0], eta_=params_list[1], lambda_=params_list[2]) - mktTotVar_list[i], 2)
    return MSVE / len(inputs_list)


def SSVI_calibration(k_list: list, atmfTotVar_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param k_list: inputs list : [k_1, k_2, k_3, ...]
    :param atmfTotVar_list: ATMF implied total variance list : [atmfTotVar_1, atmfTotVar_2, atmfTotVar_3, ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: calibrated parameters list
    """
    init_params_list = [-0.75, 1, 0.5]
    inputs_list = [(k, atmfTotVar) for k, atmfTotVar in zip(k_list, atmfTotVar_list)]
    result = optimize.minimize(
        SSVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list)
    )
    final_params = list(result.x)
    return {
        "rho_": final_params[0],
        "eta_": final_params[1],
        "lambda_": final_params[2],
    }


def SSVI_skew(strike: float, theta: float, forward: float, maturity: float, rho_: float, eta_: float, lambda_: float):
    phi = SSVI_phi(theta, eta_, lambda_)
    num = 0.5 * theta * ((phi * (rho_ + phi * np.log(strike / forward))) / (strike * (np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(strike / forward), 2) + 1))) + (rho_ * phi) / strike)
    den = 2 * maturity * np.sqrt((0.5 * theta * (np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(strike / forward), 2) + 1) + rho_ * phi * np.log(strike/forward)) + 1) / maturity)
    return num / den


def eSSVI_phi(theta: float, eta_: float, lambda_: float):
    """
    :param theta: ATM total variance
    :param eta_: curvature function parameter
    :param lambda_: curvature function parameter
    :return: curvature function result
    """
    return eta_ * pow(theta, -lambda_)


def eSSVI_rho(theta: float, a_: float, b_: float, c_: float):
    """
    :param theta: ATM total variance
    :param a_: spot/vol correlation function parameter
    :param b_: spot/vol correlation function parameter
    :param c_: spot/vol correlation function parameter
    :return: curvature function result
    """
    return a_ * np.exp(-b_ * theta) + c_


def eSSVI(k: float, theta: float, a_: float, b_: float, c_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness
    :param theta: ATM total variance
    :param a_: spot/vol correlation function parameter
    :param b_: spot/vol correlation function parameter
    :param c_: spot/vol correlation function parameter
    :param eta_: curvature function parameter
    :param lambda_: curvature function parameter
    :return: total variance
    """
    return 0.5 * theta * (
            1 + eSSVI_rho(theta, a_, b_, c_) * SSVI_phi(theta, eta_, lambda_) * k +
            np.sqrt(pow(SSVI_phi(theta, eta_, lambda_) * k + eSSVI_rho(theta, a_, b_, c_), 2) + 1 -
                    pow(eSSVI_rho(theta, a_, b_, c_), 2)))


def eSSVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param params_list: parameters list : [a_, b_, c_, eta_, lambda_]
    :param inputs_list: inputs list : [(k_1, theta_1), (k_2, theta_2), (k_3, theta_3), ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: mean squared volatility error (MSVE)
    """
    MSVE = 0
    for i in range(0, len(inputs_list)):
        MSVE = MSVE + weights_list[i] * pow(
            eSSVI(k=inputs_list[i][0], theta=inputs_list[i][1],
                  a_=params_list[0], b_=params_list[1], c_=params_list[2], eta_=params_list[3], lambda_=params_list[4])
            - mktTotVar_list[i], 2)
    return MSVE / len(inputs_list)


def eSSVI_calibration(k_list: list, atmfTotVar_list: list, mktTotVar_list: list, weights_list: list):
    """
    :param k_list: inputs list : [k_1, k_2, k_3, ...]
    :param atmfTotVar_list: ATMF implied total variance list : [atmfTotVar_1, atmfTotVar_2, atmfTotVar_3, ...]
    :param mktTotVar_list: market implied total variance list : [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: weights list : [w_1, w_2, w_3, ...]
    :return: calibrated parameters list
    """
    init_params_list = [-0.75, 0.5, 0, 1, 0.5]
    inputs_list = [(k, atmfTotVar) for k, atmfTotVar in zip(k_list, atmfTotVar_list)]
    result = optimize.minimize(
        eSSVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list)
    )
    final_params = list(result.x)
    return {
        "a_": final_params[0],
        "b_": final_params[1],
        "c_": final_params[2],
        "eta_": final_params[3],
        "lambda_": final_params[4],
    }


def eSSVI_skew(strike: float, theta: float, forward: float, maturity: float, eta_: float, lambda_: float, a_: float,
               b_: float, c_: float):
    rho = eSSVI_rho(theta, a_, b_, c_)
    phi = eSSVI_phi(theta, eta_, lambda_)
    num = 0.5 * theta * ((phi * (rho + phi * np.log(strike / forward))) / (strike * (np.sqrt(-pow(rho, 2) + pow(rho + phi * np.log(strike / forward), 2) + 1))) + (rho * phi) / strike)
    den = 2 * maturity * np.sqrt((0.5 * theta * (np.sqrt(-pow(rho, 2) + pow(rho + phi * np.log(strike / forward), 2) + 1) + rho * phi * np.log(strike/forward)) + 1) / maturity)
    return num / den
