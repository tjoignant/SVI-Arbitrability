import math
import numpy as np
import scipy.optimize as optimize


def Durrleman_Condition(k_list, tot_var_list, log_forward_skew_list, log_forward_convexity_list):
    return np.power(
        1 - (np.array(k_list) * np.array(log_forward_skew_list)) / (2 * np.array(tot_var_list)), 2) - \
           (np.power(np.array(log_forward_skew_list), 2) / 4) * (1 / np.array(tot_var_list) + 1 / 4) + \
           np.array(log_forward_convexity_list) / 2


def SVI(k: float, a_: float, b_: float, rho_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness (input)
    :param a_: adjusts the vertical deplacement of the smile (param)
    :param b_: adjusts the angle between left and right asymptotes (param)
    :param rho_: adjusts the orientation of the graph (param)
    :param m_: adjusts the horizontal deplacement of the smile (param)
    :param sigma_: adjusts the smoothness of the vertex (param)
    :return: total variance
    """
    return a_ + b_ * (rho_ * (k - m_) + math.sqrt(pow(k - m_, 2) + pow(sigma_, 2)))


def SVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list,
                              use_durrleman_cond: bool):
    """
    :param params_list: [a_, b_, rho_, m_, sigma_]
    :param inputs_list: [(k_1), (k_2), (k_3), ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            SVI(k=inputs_list[i][0], a_=params_list[0], b_=params_list[1], rho_=params_list[2], m_=params_list[3],
                sigma_=params_list[4]) - mktTotVar_list[i], 2)
    MSVE = SVE / len(inputs_list)
    penality = 0
    # Penality
    if use_durrleman_cond:
        k_list, g_list = SVI_Durrleman_Condition(a_=params_list[0], b_=params_list[1], rho_=params_list[2],
                                                 m_=params_list[3], sigma_=params_list[4])
        penality = 0
        if min(g_list) < 0:
            penality = 10e5
    return MSVE + penality


def SVI_calibration(k_list: list, mktTotVar_list: list, weights_list: list, use_durrleman_cond: bool):
    """
    :param k_list: [k_1, k_2, k_3, ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {a_, b_, rho_, m_, sigma_}
    """
    init_params_list = [-0.01, 0.05, -0.01, 0.03, 0.3]
    inputs_list = [(k,) for k in k_list]
    result = optimize.minimize(
        SVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "a_": final_params[0],
        "b_": final_params[1],
        "rho_": final_params[2],
        "m_": final_params[3],
        "sigma_": final_params[4],
    }


def SVI_skew(strike: float, forward: float, maturity: float, a_: float, b_: float, rho_: float, m_: float,
             sigma_: float):
    """
    :param strike: strike
    :param forward: forward
    :param maturity: maturity
    :param a_: SVI parameter
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI skew
    """
    num1 = np.log(strike / forward) - m_
    den1 = np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2))

    num2 = a_ + b_ * (np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)) + rho_ * np.log(
        strike / forward) - m_ * rho_)
    den2 = maturity

    numtot = b_ * (rho_ + num1 / den1)
    dentot = 2 * maturity * strike * np.sqrt(num2 / den2)

    return numtot / dentot


def SVI_log_forward_skew(k: float, b_: float, rho_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI log forward skew
    """
    return b_ * ((k - m_) / (np.sqrt(pow(k - m_, 2) + pow(sigma_, 2))) + rho_)


def SVI_convexity(strike: float, forward: float, maturity: float, a_: float, b_: float, rho_: float, m_: float,
                  sigma_: float):
    """
    :param strike: strike
    :param forward: forward
    :param maturity: maturity
    :param a_: SVI parameter
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI convexity
    """
    num1 = - pow(np.log(strike / forward) - m_, 2)
    den1 = pow(strike, 2) * pow(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2), 3 / 2)

    num2 = - np.log(strike / forward) - m_
    den2 = pow(strike, 2) * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2))

    num3 = 1
    den3 = pow(strike, 2) * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2))

    num4 = -rho_
    den4 = pow(strike, 2)

    dentot = 2 * maturity * np.sqrt(
        (a_ + b_ * (np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)) + rho_ * (
                np.log(strike / forward) - m_))) / maturity)

    firstterm = (b_ * ((num1 / den1) + (num2 / den2) + (num3 / den3) + (num4 / den4))) / dentot

    num5 = pow(b_, 2) * pow(((np.log(strike / forward) - m_) / (
            strike * np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)))) + rho_ / strike, 2)
    den5 = 4 * pow(maturity, 2) * pow(
        (a_ + b_ * (np.sqrt(pow(np.log(strike / forward) - m_, 2) + pow(sigma_, 2)) + rho_ * (
                np.log(strike / forward) - m_))) / maturity, 3 / 2)

    secondterm = num5 / den5

    return firstterm - secondterm


def SVI_log_forward_convexity(k: float, b_: float, m_: float, sigma_: float):
    """
    :param k: log forward moneyness
    :param b_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :return: SVI log forward convexity
    """
    return (b_ * pow(sigma_, 2)) / (pow(pow(m_ - k, 2) + pow(sigma_, 2), 3 / 2))


def SVI_Durrleman_Condition(a_: float, b_: float, rho_: float, m_: float, sigma_: float, min_k=-1, max_k=1, nb_k=200):
    """
    :param a_: SVI parameter
    :param b_: SVI parameter
    :param rho_: SVI parameter
    :param m_: SVI parameter
    :param sigma_: SVI parameter
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    tot_var_list = [SVI(k=k, a_=a_, b_=b_, rho_=rho_, m_=m_, sigma_=sigma_) for k in k_list]
    log_forward_skew_list = [SVI_log_forward_skew(k=k, b_=b_, rho_=rho_, m_=m_, sigma_=sigma_) for k in k_list]
    log_forward_convexity_list = [SVI_log_forward_convexity(k=k, b_=b_, m_=m_, sigma_=sigma_) for k in k_list]
    return k_list, Durrleman_Condition(k_list=k_list, tot_var_list=tot_var_list,
                                       log_forward_skew_list=log_forward_skew_list,
                                       log_forward_convexity_list=log_forward_convexity_list)


def SSVI_phi(theta: float, eta_: float, lambda_: float):
    """
    :param theta: ATM total variance
    :param eta_: curvature parameter
    :param lambda_: curvature parameter
    :return: curvature function result
    """
    return eta_ * pow(theta, -lambda_)


def SSVI(k: float, theta: float, rho_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness (input)
    :param theta: ATM total variance (input)
    :param rho_: spot vol constant correlation (param)
    :param eta_: curvature function parameter (param)
    :param lambda_: curvature function parameter (param)
    :return: total variance
    """
    return 0.5 * theta * (1 + rho_ * SSVI_phi(theta, eta_, lambda_) * k +
                          np.sqrt(pow(SSVI_phi(theta, eta_, lambda_) * k + rho_, 2) + 1 - pow(rho_, 2)))


def SSVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list, 
                               use_durrleman_cond: bool):
    """
    :param params_list: [rho_, eta_, lambda_]
    :param inputs_list: [(k_1, theta_1), (k_2, theta_2), (k_3, theta_3), ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            SSVI(k=inputs_list[i][0], theta=inputs_list[i][1],
                 rho_=params_list[0], eta_=params_list[1], lambda_=params_list[2]) - mktTotVar_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality
    penality = 0
    if use_durrleman_cond:
        theta_list = []
        g_list = []
        for i in range(0, len(inputs_list)):
            if inputs_list[i][1] not in theta_list:
                k_list, g_list = SSVI_Durrleman_Condition(theta=inputs_list[i][1], rho_=params_list[0], eta_=params_list[1],
                                                          lambda_=params_list[2])
                theta_list.append(inputs_list[i][1])
            if min(g_list) < 0:
                penality = penality + 10e5
    return MSVE + penality


def SSVI_calibration(k_list: list, atmfTotVar_list: list, mktTotVar_list: list, weights_list: list, 
                     use_durrleman_cond: bool):
    """
    :param k_list: [k_1, k_2, k_3, ...]
    :param atmfTotVar_list: [atmfTotVar_1, atmfTotVar_2, atmfTotVar_3, ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {rho_, eta_, lambda_}
    """
    init_params_list = [-0.75, 1, 0.5]
    inputs_list = [(k, atmfTotVar) for k, atmfTotVar in zip(k_list, atmfTotVar_list)]
    result = optimize.minimize(
        SSVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "rho_": final_params[0],
        "eta_": final_params[1],
        "lambda_": final_params[2],
    }


def SSVI_skew(strike: float, theta: float, forward: float, maturity: float, rho_: float, eta_: float, lambda_: float):
    """
    :param strike: strike
    :param theta: ATM total variance
    :param forward: forward
    :param maturity: maturity
    :param rho_: SSVI parameter
    :param eta_: SSVI parameter
    :param lambda_: SSVI parameter
    :return: SSVI skew
    """
    phi = SSVI_phi(theta, eta_, lambda_)
    num = 0.5 * theta * ((phi * (rho_ + phi * np.log(strike / forward))) / (
            strike * (np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(strike / forward), 2) + 1))) +
                         (rho_ * phi) / strike)
    den = 2 * maturity * np.sqrt((0.5 * theta * (
            np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(strike / forward), 2) + 1) + rho_ * phi * np.log(
        strike / forward) + 1)) / maturity)
    return num / den


def SSVI_log_forward_skew(k: float, theta: float, rho_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness
    :param theta: ATM total variance
    :param rho_: SSVI parameter
    :param eta_: SSVI parameter
    :param lambda_: SSVI parameter
    :return: SSVI log forward skew
    """
    phi = SSVI_phi(theta, eta_, lambda_)
    return 0.5 * theta * phi * (rho_ + (rho_ + k * phi) / (np.sqrt(k * phi * (2 * rho_ + k * phi) + 1)))


def SSVI_convexity(strike: float, theta: float, forward: float, maturity: float, rho_: float, eta_: float,
                   lambda_: float):
    """
    :param strike: strike
    :param theta: ATM total variance
    :param forward: forward
    :param maturity: maturity
    :param rho_: SSVI parameter
    :param eta_: SSVI parameter
    :param lambda_: SSVI parameter
    :return: SSVI convexity
    """
    phi = SSVI_phi(theta, eta_, lambda_)

    num1 = pow(phi, 2)
    den1 = pow(strike, 2) * np.sqrt(pow(phi * np.log(strike / forward) + rho_, 2) - pow(rho_, 2) + 1)

    num2 = pow(phi, 2) * pow(phi * np.log(strike / forward) + rho_, 2)
    den2 = pow(strike, 2) * pow(pow(phi * np.log(strike / forward) + rho_, 2) - pow(rho_, 2) + 1, 3 / 2)

    num3 = phi * (phi * np.log(strike / forward) + rho_)
    den3 = pow(strike, 2) * np.sqrt(pow(phi * np.log(strike / forward) + rho_, 2) - pow(rho_, 2) + 1)

    num4 = rho_ * phi
    den4 = pow(strike, 2)

    numtot = (theta / 2) * ((num1 / den1) - (num2 / den2) - (num3 / den3) - (num4 / den4))
    dentot = 2 * maturity * np.sqrt(((theta / 2) * (
            np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(strike / forward), 2) + 1) + rho_ * phi * np.log(
        strike / forward) + 1)) / maturity)

    num5 = phi * (phi * np.log(strike / forward) + rho_)
    den5 = strike * np.sqrt(pow(phi * np.log(strike / forward) + rho_, 2) - pow(rho_, 2) + 1)

    numtot2 = pow(theta / 2, 2) * pow((num5 / den5) + (phi * rho_) / strike, 2)
    dentot2 = 4 * pow(maturity, 2) * pow(((theta / 2) * (np.sqrt(-pow(rho_, 2) + pow(rho_ + phi * np.log(
        strike / forward), 2) + 1) + rho_ * phi * np.log(strike / forward) + 1)) / maturity, 3 / 2)

    return (numtot / dentot) - (numtot2 / dentot2)


def SSVI_log_forward_convexity(k: float, theta: float, rho_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness
    :param theta: ATM total variance
    :param rho_: SSVI parameter
    :param eta_: SSVI parameter
    :param lambda_: SSVI parameter
    :return: SSVI log forward convexity
    """
    phi = SSVI_phi(theta, eta_, lambda_)

    num1 = pow(phi, 2)
    den1 = np.sqrt(-pow(rho_, 2) + pow(rho_ + k * phi, 2) + 1)

    num2 = pow(phi, 2) * pow(rho_ + k * phi, 2)
    den2 = pow(-pow(rho_, 2) + pow(rho_ + k * phi, 2) + 1, 3 / 2)

    return 0.5 * theta * (num1 / den1 - num2 / den2)


def SSVI_Durrleman_Condition(theta: float, rho_: float, eta_: float, lambda_: float, min_k=-1, max_k=1, nb_k=200):
    """
    :param theta: ATM total variance
    :param rho_: SSVI parameter
    :param eta_: SSVI parameter
    :param lambda_: SSVI parameter
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    tot_var_list = [SSVI(k=k, theta=theta, rho_=rho_, eta_=eta_, lambda_=lambda_) for k in k_list]
    log_forward_skew_list = [SSVI_log_forward_skew(k=k, theta=theta, rho_=rho_, eta_=eta_, lambda_=lambda_) for k in
                             k_list]
    log_forward_convexity_list = [SSVI_log_forward_convexity(k=k, theta=theta, rho_=rho_, eta_=eta_, lambda_=lambda_)
                                  for k in k_list]
    return k_list, Durrleman_Condition(k_list=k_list, tot_var_list=tot_var_list,
                                       log_forward_skew_list=log_forward_skew_list,
                                       log_forward_convexity_list=log_forward_convexity_list)


def eSSVI_phi(theta: float, eta_: float, lambda_: float):
    """
    :param theta: ATM total variance
    :param eta_: curvature parameter
    :param lambda_: curvature parameter
    :return: curvature result
    """
    return eta_ * pow(theta, -lambda_)


def eSSVI_rho(theta: float, a_: float, b_: float, c_: float):
    """
    :param theta: ATM total variance
    :param a_: spot/vol correlation parameter
    :param b_: spot/vol correlation parameter
    :param c_: spot/vol correlation parameter
    :return: curvature result
    """
    return a_ * np.exp(-b_ * theta) + c_


def eSSVI(k: float, theta: float, a_: float, b_: float, c_: float, eta_: float, lambda_: float):
    """
    :param k: log forward moneyness (input)
    :param theta: ATM total variance (input)
    :param a_: spot/vol correlation function parameter (param)
    :param b_: spot/vol correlation function parameter (param)
    :param c_: spot/vol correlation function parameter (param)
    :param eta_: curvature function parameter (param)
    :param lambda_: curvature function parameter (param)
    :return: total variance
    """
    return 0.5 * theta * (
            1 + eSSVI_rho(theta, a_, b_, c_) * SSVI_phi(theta, eta_, lambda_) * k +
            np.sqrt(pow(SSVI_phi(theta, eta_, lambda_) * k + eSSVI_rho(theta, a_, b_, c_), 2) + 1 -
                    pow(eSSVI_rho(theta, a_, b_, c_), 2)))


def eSSVI_minimisation_function(params_list: list, inputs_list: list, mktTotVar_list: list, weights_list: list, 
                                use_durrleman_cond: bool):
    """
    :param params_list: [a_, b_, c_, eta_, lambda_]
    :param inputs_list: [(k_1, theta_1), (k_2, theta_2), (k_3, theta_3), ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            eSSVI(k=inputs_list[i][0], theta=inputs_list[i][1],
                  a_=params_list[0], b_=params_list[1], c_=params_list[2], eta_=params_list[3], lambda_=params_list[4])
            - mktTotVar_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality
    penality = 0
    if use_durrleman_cond:
        theta_list = []
        g_list = []
        for i in range(0, len(inputs_list)):
            if inputs_list[i][1] not in theta_list:
                k_list, g_list = SSVI_Durrleman_Condition(theta=inputs_list[i][1], rho_=params_list[0], eta_=params_list[1],
                                                          lambda_=params_list[2])
                theta_list.append(inputs_list[i][1])
            if min(g_list) < 0:
                penality = penality + 10e5
    return MSVE + penality


def eSSVI_calibration(k_list: list, atmfTotVar_list: list, mktTotVar_list: list, weights_list: list, 
                      use_durrleman_cond: bool):
    """
    :param k_list: [k_1, k_2, k_3, ...]
    :param atmfTotVar_list: [atmfTotVar_1, atmfTotVar_2, atmfTotVar_3, ...]
    :param mktTotVar_list: [TotVar_1, TotVar_2, TotVar_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {a_, b_, c_, eta_, lambda_}
    """
    init_params_list = [-0.75, 0.5, 0, 1, 0.5]
    inputs_list = [(k, atmfTotVar) for k, atmfTotVar in zip(k_list, atmfTotVar_list)]
    result = optimize.minimize(
        eSSVI_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktTotVar_list, weights_list, use_durrleman_cond),
        tol=1e-8,
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
    """
    :param strike: strike
    :param theta: ATM total variance
    :param forward: forward
    :param maturity: maturity
    :param eta_: eSSVI parameter
    :param lambda_: eSSVI parameter
    :param a_: eSSVI parameter
    :param b_: eSSVI parameter
    :param c_: eSSVI parameter
    :return: eSSVI skew
    """
    rho = eSSVI_rho(theta, a_, b_, c_)
    phi = eSSVI_phi(theta, eta_, lambda_)
    num = 0.5 * theta * ((phi * (rho + phi * np.log(strike / forward))) / (strike * (np.sqrt(-pow(rho, 2) +
                                                                                             pow(rho + phi * np.log(
                                                                                                 strike / forward),
                                                                                                 2) + 1))) + (
                                 rho * phi) / strike)
    den = 2 * maturity * np.sqrt((0.5 * theta * (
            np.sqrt(-pow(rho, 2) + pow(rho + phi * np.log(strike / forward), 2) + 1) + rho * phi * np.log(
        strike / forward) + 1)) / maturity)
    return num / den


def eSSVI_log_forward_skew(k: float, theta: float, eta_: float, lambda_: float, a_: float,
                           b_: float, c_: float):
    """
    :param k: log-forward moneyness
    :param theta: ATM total variance
    :param eta_: eSSVI parameter
    :param lambda_: eSSVI parameter
    :param a_: eSSVI parameter
    :param b_: eSSVI parameter
    :param c_: eSSVI parameter
    :return: eSSVI log forward skew
    """
    rho = eSSVI_rho(theta, a_, b_, c_)
    phi = eSSVI_phi(theta, eta_, lambda_)
    return 0.5 * theta * phi * (rho + (rho + k * phi) / (np.sqrt(k * phi * (2 * rho + k * phi) + 1)))


def eSSVI_convexity(strike: float, theta: float, forward: float, maturity: float, eta_: float, lambda_: float,
                    a_: float, b_: float, c_: float):
    """
    :param strike: strike
    :param theta: log-forward moneyness
    :param forward: forward
    :param maturity: maturity
    :param eta_: eSSVI parameter
    :param lambda_: eSSVI parameter
    :param a_: eSSVI parameter
    :param b_: eSSVI parameter
    :param c_: eSSVI parameter
    :return: eSSVI convexity
    """
    rho = eSSVI_rho(theta, a_, b_, c_)
    phi = eSSVI_phi(theta, eta_, lambda_)

    num1 = pow(phi, 2)
    den1 = pow(strike, 2) * np.sqrt(pow(phi * np.log(strike / forward) + rho, 2) - pow(rho, 2) + 1)

    num2 = pow(phi, 2) * pow(phi * np.log(strike / forward) + rho, 2)
    den2 = pow(strike, 2) * pow(pow(phi * np.log(strike / forward) + rho, 2) - pow(rho, 2) + 1, 3 / 2)

    num3 = phi * (phi * np.log(strike / forward) + rho)
    den3 = pow(strike, 2) * np.sqrt(pow(phi * np.log(strike / forward) + rho, 2) - pow(rho, 2) + 1)

    num4 = rho * phi
    den4 = pow(strike, 2)

    numtot = (theta / 2) * ((num1 / den1) - (num2 / den2) - (num3 / den3) - (num4 / den4))
    dentot = 2 * maturity * np.sqrt(((theta / 2) * (
            np.sqrt(-pow(rho, 2) + pow(rho + phi * np.log(strike / forward), 2) + 1) + rho * phi * np.log(
        strike / forward) + 1)) / maturity)

    num5 = phi * (phi * np.log(strike / forward) + rho)
    den5 = strike * np.sqrt(pow(phi * np.log(strike / forward) + rho, 2) - pow(rho, 2) + 1)

    numtot2 = pow(theta / 2, 2) * pow((num5 / den5) + (phi * rho) / strike, 2)
    dentot2 = 4 * pow(maturity, 2) * pow(((theta / 2) * (np.sqrt(-pow(rho, 2) + pow(rho + phi * np.log(
        strike / forward), 2) + 1) + rho * phi * np.log(strike / forward) + 1)) / maturity, 3 / 2)

    return (numtot / dentot) - (numtot2 / dentot2)


def eSSVI_log_forward_convexity(k: float, theta: float, eta_: float, lambda_: float, a_: float,
                                b_: float, c_: float):
    """
    :param k: log-forward moneyness
    :param theta: ATM total variance
    :param eta_: eSSVI parameter
    :param lambda_: eSSVI parameter
    :param a_: eSSVI parameter
    :param b_: eSSVI parameter
    :param c_: eSSVI parameter
    :return: eSSVI log forward convexity
    """
    rho = eSSVI_rho(theta, a_, b_, c_)
    phi = eSSVI_phi(theta, eta_, lambda_)

    num1 = pow(phi, 2)
    den1 = np.sqrt(-pow(rho, 2) + pow(rho + k * phi, 2) + 1)

    num2 = pow(phi, 2) * pow(rho + k * phi, 2)
    den2 = pow(-pow(rho, 2) + pow(rho + k * phi, 2) + 1, 3 / 2)

    return 0.5 * theta * (num1 / den1 - num2 / den2)


def eSSVI_Durrleman_Condition(theta: float, a_: float, b_: float, c_: float, eta_: float, lambda_: float,
                              min_k=-1, max_k=1, nb_k=200):
    """
    :param theta: ATM total variance
    :param a_: spot/vol correlation parameter
    :param b_: spot/vol correlation parameter
    :param c_: spot/vol correlation parameter
    :param eta_: curvature parameter
    :param lambda_: curvature parameter
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    tot_var_list = [eSSVI(k=k, theta=theta, a_=a_, b_=b_, c_=c_, eta_=eta_, lambda_=lambda_) for k in k_list]
    log_forward_skew_list = [eSSVI_log_forward_skew(k=k, theta=theta, a_=a_, b_=b_, c_=c_, eta_=eta_, lambda_=lambda_)
                             for k in k_list]
    log_forward_convexity_list = [
        eSSVI_log_forward_convexity(k=k, theta=theta, a_=a_, b_=b_, c_=c_, eta_=eta_, lambda_=lambda_) for k in k_list]
    return k_list, Durrleman_Condition(k_list=k_list, tot_var_list=tot_var_list,
                                       log_forward_skew_list=log_forward_skew_list,
                                       log_forward_convexity_list=log_forward_convexity_list)


def SABR(f: float, K: float, T: float, alpha_: float, beta_: float, rho_: float, vega_: float):
    """
    :param f: forward (input)
    :param K: strike (input)
    :param T: maturity (input)
    :param alpha_: instantaneous vol (param)
    :param beta_: CEV component for forward rate (param)
    :param rho_: spot vol constant correlation (param)
    :param vega_: constant vol of vol (param)
    :return: volatility
    """
    z = (vega_ / alpha_) * pow(f * K, (1-beta_)/2) * np.log(f/K)
    xhi = np.log((np.sqrt(1 - 2 * rho_ * z + pow(z, 2)) + z - rho_) / (1-rho_))
    num = alpha_ * (1 + T * ((pow(1-beta_, 2)/24) * (pow(alpha_, 2)/pow(f * K, 1 - beta_)) + (1/4) * ((alpha_*beta_*rho_*vega_)/(pow(f*K, (1 - beta_)/2))) + ((2 - 3*pow(rho_, 2))/24) * pow(vega_, 2)))
    den = (pow(f*K, (1-beta_)/2)) * (1 + (pow(1-beta_, 2) * pow(np.log(f/K), 2)) / 24 + pow(1-beta_, 4) * pow(np.log(f/K), 4) / 1920)
    return (z/xhi) * num / den


def SABR_minimisation_function(params_list: list, inputs_list: list, mktImpVol_list: list, weights_list: list):
    """
    :param params_list: [alpha_, beta_, rho_, vega_]
    :param inputs_list: [(f_1, K_1, T_1), (f_2, K_2, T_2), (f_3, K_3, T_3), ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            SABR(f=inputs_list[i][0], K=inputs_list[i][1], T=inputs_list[i][2], alpha_=params_list[0],
                 beta_=params_list[1], rho_=params_list[2], vega_=params_list[3]) - mktImpVol_list[i], 2)
    MSVE = SVE / len(inputs_list)
    return MSVE


def SABR_calibration(f_list: list, K_list: list, T_list: list, mktImpVol_list: list, weights_list: list):
    """
    :param f_list: [f_1, f_2, f_3, ...]
    :param K_list: [K_1, K_2, K_3, ...]
    :param T_list: [T_1, T_2, T_3, ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :return: calibrated parameters dict {alpha_, beta_, rho_, vega_}
    """
    init_params_list = [0.01, 0.01, 0.01, 0.01]
    inputs_list = [(f, K, T) for f, K, T in zip(f_list, K_list, T_list)]
    result = optimize.minimize(
        SABR_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktImpVol_list, weights_list),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "alpha_": final_params[0],
        "beta_": final_params[1],
        "rho_": final_params[2],
        "vega_": final_params[3],
    }


def SABR_skew(f: float, K: float, T: float, alpha_: float, beta_: float, rho_: float, vega_: float):
    """
    :param f: forward (input)
    :param K: strike (input)
    :param T: maturity (input)
    :param alpha_: instantaneous vol (param)
    :param beta_: CEV component for forward rate (param)
    :param rho_: spot vol constant correlation (param)
    :param vega_: constant vol of vol (param)
    :return: SABR skew
    """

    # SABR = u * v
    # Skew SABR = u' * v + u * v

    z = (vega_ / alpha_) * pow(f * K, (1-beta_)/2) * np.log(f/K)
    xhi = np.log((np.sqrt(1 - 2 * rho_ * z + pow(z, 2)) + z - rho_) / (1-rho_))

    u_num = alpha_ * (1 + T * ((pow(1-beta_, 2)/24) * (pow(alpha_, 2)/pow(f * K, 1 - beta_)) + (1/4) * ((alpha_*beta_*rho_*vega_)/(pow(f*K, (1 - beta_)/2))) + ((2 - 3*pow(rho_, 2))/24) * pow(vega_, 2)))
    u_den = (pow(f*K, (1-beta_)/2)) * (1 + (pow(1-beta_, 2) * pow(np.log(f/K), 2)) / 24 + pow(1-beta_, 4) * pow(np.log(f/K), 4) / 1920)

    u = u_num / u_den
    v = z / xhi

    u_prime_num1 = alpha_ * (1 - beta_) * f * pow(f * K, (beta_ - 1) / 2 - 1) * (T * ((1 / 24) * pow(alpha_, 2) * pow(1 - beta_, 2) *
           pow(f * K, beta_ - 1) + 0.25 * alpha_ * beta_ * rho_ * vega_ * pow(f * K, (beta_ - 1) / 2) + (1 / 24) *
           (2 - 3 * pow(rho_, 2)) * pow(vega_, 2)) + 1)
    u_prime_den1 = 2 * (((pow(1 - beta_, 4) * pow(np.log(f / K), 4)) / 1920) + (1 / 24) * pow(1 - beta_, 2) * pow(np.log(f / K), 2) + 1)

    u_prime_num2 = alpha_ * pow(f * K, (beta_ - 1)/2) * (- (pow(1 - beta_, 4) * pow(np.log(f/K), 3)) / (480 * K) - (pow(1-beta_, 2) * np.log(f/K)) / (12 * K)) \
           * (T*((1/24) * pow(alpha_, 2) * pow(1-beta_, 2) * pow(f*K, beta_-1) + 0.25 *
            alpha_ * beta_ * rho_ * vega_ * pow(f*K, (beta_-1)/2) + (1/24) * (2 - 3 * pow(rho_,2)) * pow(vega_, 2))+1)
    u_prime_den2 = pow((pow(1-beta_, 4) * pow(np.log(f/K), 4) / 1920) + (1/24) * pow(1-beta_, 2) * pow(np.log(f/K), 2) + 1, 2)

    u_prime_num3 = alpha_ * T * pow(f * K, (beta_-1)/2) * ((-1/24) * pow(alpha_, 2) * pow(1 - beta_, 3) * f * pow(f * K, beta_ - 2) -
                0.125 * alpha_ * (1 - beta_) * beta_ * f * rho_ * vega_ * pow(f * K, (beta_ - 1)/2 - 1))
    u_prime_den3 = (pow(1 - beta_, 4) * pow(np.log(f/K), 4) / 1920) + (1/24) * pow(1 - beta_, 2) * pow(np.log(f/K), 2) + 1

    u_prime = - u_prime_num1 / u_prime_den1 - u_prime_num2 / u_prime_den2 + u_prime_num3 / u_prime_den3

    v_prime = 0

    return u_prime * v + u * v_prime
