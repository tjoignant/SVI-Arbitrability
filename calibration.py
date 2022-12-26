import math
import numpy as np
import scipy.special as sc
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
    # Penality (Durrleman)
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
    # Penality (Gatheral/Jacquier)
    penality = 0
    theta_list = sorted(list(set([inputs_list[i][1] for i in range(0, len(inputs_list))])))
    # Calendar Spread Arbitrage
    for i in range(1, len(theta_list)):
        diff = theta_list[i] * SSVI_phi(theta=theta_list[i], eta_=params_list[1], lambda_=params_list[2]) - \
                    theta_list[i-1] * SSVI_phi(theta=theta_list[i-1], eta_=params_list[1], lambda_=params_list[2])
        if not 0 <= diff <= 1 / pow(params_list[0], 2) * (1 + np.sqrt(1 - pow(params_list[0], 2))) * SSVI_phi(theta=theta_list[i-1], eta_=params_list[1], lambda_=params_list[2]):
            penality = penality + 10e5
    # Butterfly Spread Arbitrage
    for i in range(1, len(theta_list)):
        if not theta_list[i] * SSVI_phi(theta=theta_list[i], eta_=params_list[1], lambda_=params_list[2]) * (1 + abs(params_list[0])) < 4 and \
            theta_list[i] * pow(SSVI_phi(theta=theta_list[i], eta_=params_list[1], lambda_=params_list[2]), 2) * (
                        1 + abs(params_list[0])) <= 4:
            penality = penality + 10e5
    # Penality (Durrleman)
    if use_durrleman_cond:
        for theta in theta_list:
            k_list, g_list = SSVI_Durrleman_Condition(theta=theta, rho_=params_list[0], eta_=params_list[1],
                                                      lambda_=params_list[2])
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
    # Penality (Gatheral/Jacquier)
    penality = 0
    theta_list = sorted(list(set([inputs_list[i][1] for i in range(0, len(inputs_list))])))
    # Calendar Spread Arbitrage
    for i in range(1, len(theta_list)):
        diff = theta_list[i] * eSSVI_phi(theta=theta_list[i], eta_=params_list[3], lambda_=params_list[4]) - \
                    theta_list[i-1] * eSSVI_phi(theta=theta_list[i-1], eta_=params_list[3], lambda_=params_list[4])
        if not 0 <= diff <= 1 / pow(eSSVI_rho(theta=theta_list[i], a_=params_list[0], b_=params_list[1], c_=params_list[2]), 2) * \
               (1 + np.sqrt(1 - pow(eSSVI_rho(theta=theta_list[i], a_=params_list[0], b_=params_list[1], c_=params_list[2]), 2))) * eSSVI_phi(theta=theta_list[i-1], eta_=params_list[3], lambda_=params_list[4]):
            penality = penality + 10e5
    # Butterfly Spread Arbitrage
    for i in range(1, len(theta_list)):
        if not theta_list[i] * eSSVI_phi(theta=theta_list[i-1], eta_=params_list[3], lambda_=params_list[4]) * (1 + abs(eSSVI_rho(theta=theta_list[i], a_=params_list[0], b_=params_list[1], c_=params_list[2]))) < 4 and \
            theta_list[i] * pow(eSSVI_phi(theta=theta_list[i-1], eta_=params_list[3], lambda_=params_list[4]), 2) * (
                        1 + abs(eSSVI_rho(theta=theta_list[i], a_=params_list[0], b_=params_list[1], c_=params_list[2]))) <= 4:
            penality = penality + 10e5
    # Penality (Durrleman)
    if use_durrleman_cond:
        for theta in theta_list:
            k_list, g_list = eSSVI_Durrleman_Condition(theta=theta, a_=params_list[0], b_=params_list[1],
                                                       c_=params_list[2], eta_=params_list[3], lambda_=params_list[4])
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


def SABR(f: float, K: float, T: float, alpha_: float, rho_: float, nu_: float):
    """
    :param f: forward (input)
    :param K: strike (input)
    :param T: maturity (input)
    :param alpha_: instantaneous vol (param)
    :param rho_: spot vol constant correlation (param)
    :param nu_: constant vol of vol (param)
    :return: volatility
    """
    z = (nu_ / alpha_) * np.log(f/K)
    xhi = np.log((np.sqrt(1 - 2 * rho_ * z + pow(z, 2)) + z - rho_) / (1-rho_))
    return alpha_ * (1 + T * (rho_ * nu_ * alpha_ / 4 + (2 - 3 * pow(rho_, 2)) / 24 * pow(nu_, 2))) * (z/xhi)


def SABR_minimisation_function(params_list: list, inputs_list: list, mktImpVol_list: list, weights_list: list,
                               use_durrleman_cond: bool):
    """
    :param params_list: [alpha_, rho_, nu_]
    :param inputs_list: [(f_1, K_1, T_1), (f_2, K_2, T_2), (f_3, K_3, T_3), ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            SABR(f=inputs_list[i][0], K=inputs_list[i][1], T=inputs_list[i][2], alpha_=params_list[0],
                 rho_=params_list[1], nu_=params_list[2]) - mktImpVol_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality (Durrleman)
    penality = 0
    if use_durrleman_cond:
        theta_list = []
        g_list = []
        for i in range(0, len(inputs_list)):
            if inputs_list[i][1] not in theta_list:
                k_list, g_list = SABR_Durrleman_Condition(f=inputs_list[i][0], T=inputs_list[i][2],
                                                          alpha_=params_list[0], rho_=params_list[1], nu_=params_list[2])
                theta_list.append(inputs_list[i][1])
            if min(g_list) < 0:
                penality = penality + 10e5
    return MSVE + penality


def SABR_calibration(f_list: list, K_list: list, T_list: list, mktImpVol_list: list, weights_list: list,
                     use_durrleman_cond: bool):
    """
    :param f_list: [f_1, f_2, f_3, ...]
    :param K_list: [K_1, K_2, K_3, ...]
    :param T_list: [T_1, T_2, T_3, ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {alpha_, rho_, nu_}
    """
    init_params_list = [0.25, -0.4, 4]
    inputs_list = [(f, K, T) for f, K, T in zip(f_list, K_list, T_list)]
    result = optimize.minimize(
        SABR_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktImpVol_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "alpha_": final_params[0],
        "rho_": final_params[1],
        "nu_": final_params[2],
    }


def SABR_skew(f: float, K: float, T: float, alpha_: float, rho_: float, nu_: float):
    """
    :param f: forward (input)
    :param K: strike (input)
    :param T: maturity (input)
    :param alpha_: instantaneous vol (param)
    :param rho_: spot vol constant correlation (param)
    :param nu_: constant vol of vol (param)
    :return: SABR skew
    """
    # Numerical Differential
    K_neg_shifted = K - pow(10, -5)
    K_pos_shifted = K + pow(10, -5)
    vol_sabr_neg_shifted = SABR(f=f, K=K_neg_shifted, T=T, alpha_=alpha_, rho_=rho_, nu_=nu_)
    vol_sabr_pos_shifted = SABR(f=f, K=K_pos_shifted, T=T, alpha_=alpha_, rho_=rho_, nu_=nu_)

    # Analytical Differential (not working for now)
    constant = alpha_ * (1 + T * (rho_ * nu_ * alpha_ / 4 + (2 - 3 * pow(rho_, 2)) / 24 * pow(nu_, 2)))
    z = (nu_ / alpha_) * np.log(f / K)
    xhi = np.log((np.sqrt(1 - 2 * rho_ * z + pow(z, 2)) + z - rho_) / (1 - rho_))
    z_prime = - nu_ / (alpha_ * K)
    xhi_prime = ((-2 * z + 2 * rho_ - 1) * z_prime) / (2 * ((2 * rho_ - 1) * z - pow(z, 2) + rho_ - 1))

    return (vol_sabr_pos_shifted - vol_sabr_neg_shifted) / (K_pos_shifted - K_neg_shifted)
    # return constant * (z_prime * xhi - z * xhi_prime) / pow(xhi, 2)


def SABR_Durrleman_Condition(f: float, T: float, alpha_: float, rho_: float, nu_: float, min_k=-1, max_k=1, nb_k=200):
    """
    :param f: forward (input)
    :param T: maturity (input)
    :param alpha_: instantaneous vol (param)
    :param rho_: spot vol constant correlation (param)
    :param nu_: constant vol of vol (param)
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    vol_list = [SABR(f=f, K=f*np.exp(k), T=T, alpha_=alpha_, rho_=rho_, nu_=nu_) for k in k_list]
    tot_var_list = [T * pow(vol, 2) for vol in vol_list]
    log_forward_skew_list = []
    log_forward_convexity_list = []
    new_tot_var_list = []
    new_k_list = []
    k_step = (max_k - min_k) / nb_k
    for i in range(1, len(tot_var_list) - 1):
        new_k_list.append(k_list[i])
        new_tot_var_list.append(tot_var_list[i])
        log_forward_skew_list.append((tot_var_list[i+1] - tot_var_list[i-1]) / (2 * k_step))
        log_forward_convexity_list.append((tot_var_list[i + 1] - 2 * tot_var_list[i] + tot_var_list[i-1]) / pow(k_step, 2))
    return new_k_list, Durrleman_Condition(k_list=new_k_list, tot_var_list=new_tot_var_list,
                                           log_forward_skew_list=log_forward_skew_list,
                                           log_forward_convexity_list=log_forward_convexity_list)


def ZABR_u(x: float, eta_: float, rho_: float):
    return (1/eta_) * np.log((np.sqrt(1 - 2 * rho_ * eta_ * x + pow(eta_, 2) * pow(x, 2)) + eta_ * x - rho_) / (1 - rho_))


def ZABR_simple(X0: float, K: float, vol: float, eta_: float, rho_: float, beta_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied volatility (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta_: constant in local volatility function (param)
    :return: volatility
    """
    def sigma(x):
        return pow(x, beta_)
    integral = (pow(X0, 1-beta_) - pow(K, 1-beta_)) / (1 - beta_)
    delta = vol / sigma(X0)
    return (X0 - K) / (ZABR_u(x=pow(delta, -1) * integral, eta_=eta_, rho_=rho_))


def ZABR_simple_minimisation_function(params_list: list, inputs_list: list, mktImpVol_list: list, weights_list: list,
                                      use_durrleman_cond: bool):
    """
    :param params_list: [eta_, rho_, beta_]
    :param inputs_list: [(XO_1, K_1, vol_1), (XO_2, K_2, vol_2), (XO_3, K_3, vol_3), ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            ZABR_simple(X0=inputs_list[i][0], K=inputs_list[i][1], vol=inputs_list[i][2],
                        eta_=params_list[0], rho_=params_list[1], beta_=params_list[2]) - mktImpVol_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality (Durrleman)
    penality = 0
    return MSVE + penality


def ZABR_simple_calibration(X0_list: list, K_list: list, vol_list: list, mktImpVol_list: list, weights_list: list,
                            use_durrleman_cond: bool):
    """
    :param X0_list: [XO_1, XO_2, XO_3, ...]
    :param K_list: [K_1, K_2, K_3, ...]
    :param vol_list: [vol_1, vol_2, vol_3, ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {eta_, rho_, beta_}
    """
    init_params_list = [0.8, -0.5, -0.5]
    inputs_list = [(X0, K, vol) for X0, K, vol in zip(X0_list, K_list, vol_list)]
    result = optimize.minimize(
        ZABR_simple_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktImpVol_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "eta_": final_params[0],
        "rho_": final_params[1],
        "beta_": final_params[2],
    }


def ZABR_simple_skew(X0: float, K: float, vol: float, eta_: float, rho_: float, beta_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied vol (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta_: constant in local volatility function (param)
    :return: simple ZABR skew
    """
    K_neg_shifted = K - pow(10, -5)
    K_pos_shifted = K + pow(10, -5)
    vol_sabr_neg_shifted = ZABR_simple(X0=X0, K=K_neg_shifted, vol=vol, eta_=eta_, rho_=rho_, beta_=beta_)
    vol_sabr_pos_shifted = ZABR_simple(X0=X0, K=K_pos_shifted, vol=vol, eta_=eta_, rho_=rho_, beta_=beta_)
    return (vol_sabr_pos_shifted - vol_sabr_neg_shifted) / (K_pos_shifted - K_neg_shifted)


def ZABR_simple_Durrleman_Condition(f: float, X0: float, vol: float, T: float, eta_: float, rho_: float, beta_: float,
                                    min_k=-1, max_k=1, nb_k=200):
    """
    :param f: forward (input)
    :param X0: spot (input)
    :param vol: ATM implied vol (input)
    :param T: maturity (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta_: constant in local volatility function (param)
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    vol_list = [ZABR_simple(X0=X0, K=f*np.exp(k), vol=vol, eta_=eta_, rho_=rho_, beta_=beta_) for k in k_list]
    tot_var_list = [T * pow(vol, 2) for vol in vol_list]
    log_forward_skew_list = []
    log_forward_convexity_list = []
    new_tot_var_list = []
    new_k_list = []
    k_step = (max_k - min_k) / nb_k
    for i in range(1, len(tot_var_list) - 1):
        new_k_list.append(k_list[i])
        new_tot_var_list.append(tot_var_list[i])
        log_forward_skew_list.append((tot_var_list[i+1] - tot_var_list[i-1]) / (2 * k_step))
        log_forward_convexity_list.append((tot_var_list[i + 1] - 2 * tot_var_list[i] + tot_var_list[i-1]) / pow(k_step, 2))
    return new_k_list, Durrleman_Condition(k_list=new_k_list, tot_var_list=new_tot_var_list,
                                           log_forward_skew_list=log_forward_skew_list,
                                           log_forward_convexity_list=log_forward_convexity_list)


def ZABR_double_beta(X0: float, K: float, vol: float, eta_: float, rho_: float, beta1_: float, beta2_: float,
                     lambda_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied volatility (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param lambda_: constant in local volatility function (param)
    :return: volatility
    """
    def sigma(x):
        return 1 / (np.exp(-lambda_ * x) * pow(x, -beta1_) + (1 - np.exp(-lambda_ * x)) * pow(x, -beta2_))
    def integral_fct(x):
        return -pow(x, 1-beta2_) / (beta2_ - 1) - \
               (pow(x, -beta1_) * pow(lambda_ * x, beta1_) * sc.gammainc(1 - beta1_, x * lambda_)) / lambda_ + \
               (pow(x, -beta2_) * pow(lambda_ * x, beta2_) * sc.gammainc(1 - beta2_, x * lambda_)) / lambda_
    integral = integral_fct(X0) - integral_fct(K)
    delta = vol / sigma(X0)
    return (X0 - K) / (ZABR_u(x=pow(delta, -1) * integral, eta_=eta_, rho_=rho_))


def ZABR_double_beta_minimisation_function(params_list: list, inputs_list: list, mktImpVol_list: list,
                                           weights_list: list, use_durrleman_cond: bool):
    """
    :param params_list: [eta_, rho_, beta1_, beta2_, lambda_]
    :param inputs_list: [(XO_1, K_1, vol_1), (XO_2, K_2, vol_2), (XO_3, K_3, vol_3), ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            ZABR_double_beta(X0=inputs_list[i][0], K=inputs_list[i][1], vol=inputs_list[i][2],
                             eta_=params_list[0], rho_=params_list[1], beta1_=params_list[2], beta2_=params_list[3],
                             lambda_=params_list[4]) - mktImpVol_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality (Durrleman)
    penality = 0
    return MSVE + penality


def ZABR_double_beta_calibration(X0_list: list, K_list: list, vol_list: list, mktImpVol_list: list, weights_list: list,
                                 use_durrleman_cond: bool):
    """
    :param X0_list: [XO_1, XO_2, XO_3, ...]
    :param K_list: [K_1, K_2, K_3, ...]
    :param vol_list: [vol_1, vol_2, vol_3, ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {eta_, rho_, beta1_, beta2_, lambda_}
    """
    init_params_list = [0.96, -0.45, -1.4, -1.45, 0.001]
    inputs_list = [(X0, K, vol) for X0, K, vol in zip(X0_list, K_list, vol_list)]
    result = optimize.minimize(
        ZABR_double_beta_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktImpVol_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "eta_": final_params[0],
        "rho_": final_params[1],
        "beta1_": final_params[2],
        "beta2_": final_params[3],
        "lambda_": final_params[4],
    }


def ZABR_double_beta_skew(X0: float, K: float, vol: float, eta_: float, rho_: float, beta1_: float, beta2_: float,
                          lambda_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied vol (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param lambda_: constant in local volatility function (param)
    :return: simple ZABR skew
    """
    K_neg_shifted = K - pow(10, -5)
    K_pos_shifted = K + pow(10, -5)
    vol_sabr_neg_shifted = ZABR_double_beta(X0=X0, K=K_neg_shifted, vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_,
                                            beta2_=beta2_, lambda_=lambda_)
    vol_sabr_pos_shifted = ZABR_double_beta(X0=X0, K=K_pos_shifted, vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_,
                                            beta2_=beta2_, lambda_=lambda_)
    return (vol_sabr_pos_shifted - vol_sabr_neg_shifted) / (K_pos_shifted - K_neg_shifted)


def ZABR_double_beta_Durrleman_Condition(f: float, X0: float, vol: float, T: float, eta_: float, rho_: float,
                                         beta1_: float, beta2_: float, lambda_: float, min_k=-1, max_k=1, nb_k=200):
    """
    :param f: forward (input)
    :param X0: spot (input)
    :param vol: ATM implied vol (input)
    :param T: maturity (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param lambda_: constant in local volatility function (param)
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    vol_list = [ZABR_double_beta(X0=X0, K=f*np.exp(k), vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_, beta2_=beta2_,
                                 lambda_=lambda_) for k in k_list]
    tot_var_list = [T * pow(vol, 2) for vol in vol_list]
    log_forward_skew_list = []
    log_forward_convexity_list = []
    new_tot_var_list = []
    new_k_list = []
    k_step = (max_k - min_k) / nb_k
    for i in range(1, len(tot_var_list) - 1):
        new_k_list.append(k_list[i])
        new_tot_var_list.append(tot_var_list[i])
        log_forward_skew_list.append((tot_var_list[i+1] - tot_var_list[i-1]) / (2 * k_step))
        log_forward_convexity_list.append((tot_var_list[i + 1] - 2 * tot_var_list[i] + tot_var_list[i-1]) / pow(k_step, 2))
    return new_k_list, Durrleman_Condition(k_list=new_k_list, tot_var_list=new_tot_var_list,
                                           log_forward_skew_list=log_forward_skew_list,
                                           log_forward_convexity_list=log_forward_convexity_list)


def ZABR_double(X0: float, K: float, vol: float, eta_: float, rho_: float, beta1_: float, beta2_: float,
                phi0_: float, d_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied volatility (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param phi0_: constant in local volatility function (param)
    :param d_: constant in local volatility function (param)
    :return: volatility
    """
    def sigma(x):
        if x <= phi0_:
            return pow(x, beta1_)
        else:
            delta_f = phi0_ * (beta2_/beta1_) * np.exp(-d_)
            return pow(phi0_, beta1_) / delta_f * pow(x - phi0_ + delta_f, beta2_)
    def integral_fct_1(x):
        # sigma(x) = pow(x, beta1_)
        return pow(x, 1-beta1_) / (1 - beta1_)
    def integral_fct_2(x):
        # delta_f = phi0_ * (beta2_/beta1_) * np.exp(-d_)
        # sigma(x) = pow(phi0_, beta1_) / delta_f * pow(x - phi0_ + delta_f, beta2_)
        return (np.exp(-d_) * beta2_ * pow(phi0_, 1 - beta1_) *
                pow(np.exp(-d_) * beta2_ * phi0_ / beta1_ + x - phi0_, -beta2_) *
                (phi0_ * (np.exp(-d_) * beta2_ - beta1_) + beta1_ * x)) / (beta1_ * (beta1_ - beta1_ * beta2_))
    if X0 <= phi0_ and K <= phi0_:
        integral = integral_fct_1(X0) - integral_fct_1(K)
    elif X0 > phi0_ and K > phi0_:
        integral = integral_fct_2(X0) - integral_fct_2(K)
    else:
        integral = integral_fct_1(phi0_) - integral_fct_1(K) + integral_fct_2(X0) - integral_fct_2(phi0_)
    delta = vol / sigma(X0)
    return (X0 - K) / (ZABR_u(x=pow(delta, -1) * integral, eta_=eta_, rho_=rho_))


def ZABR_double_minimisation_function(params_list: list, inputs_list: list, mktImpVol_list: list, weights_list: list,
                                      use_durrleman_cond: bool):
    """
    :param params_list: [eta_, rho_, beta1_, beta2_, phi0, d_]
    :param inputs_list: [(XO_1, K_1, vol_1), (XO_2, K_2, vol_2), (XO_3, K_3, vol_3), ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibration error
    """
    # Mean Squared Error (MSE)
    SVE = 0
    for i in range(0, len(inputs_list)):
        SVE = SVE + weights_list[i] * pow(
            ZABR_double(X0=inputs_list[i][0], K=inputs_list[i][1], vol=inputs_list[i][2],
                        eta_=params_list[0], rho_=params_list[1], beta1_=params_list[2], beta2_=params_list[3],
                        phi0_=params_list[4], d_=params_list[5]) - mktImpVol_list[i], 2)
    MSVE = SVE / len(inputs_list)
    # Penality (Durrleman)
    penality = 0
    return MSVE + penality


def ZABR_double_calibration(X0_list: list, K_list: list, vol_list: list, mktImpVol_list: list, weights_list: list,
                                 use_durrleman_cond: bool):
    """
    :param X0_list: [XO_1, XO_2, XO_3, ...]
    :param K_list: [K_1, K_2, K_3, ...]
    :param vol_list: [vol_1, vol_2, vol_3, ...]
    :param mktImpVol_list: [ImpVol_1, ImpVol_2, ImpVol_3, ...]
    :param weights_list: [w_1, w_2, w_3, ...]
    :param use_durrleman_cond: add penality if Durrleman condition is not respected (no butterfly arbitrage)
    :return: calibrated parameters dict {eta_, rho_, beta1_, beta2_, phi0_, d_}
    """
    init_params_list = [0.96, -0.5, -1, -0.5, 1, 0.1]
    inputs_list = [(X0, K, vol) for X0, K, vol in zip(X0_list, K_list, vol_list)]
    result = optimize.minimize(
        ZABR_double_minimisation_function,
        x0=init_params_list,
        method='nelder-mead',
        args=(inputs_list, mktImpVol_list, weights_list, use_durrleman_cond),
        tol=1e-8,
    )
    final_params = list(result.x)
    return {
        "eta_": final_params[0],
        "rho_": final_params[1],
        "beta1_": final_params[2],
        "beta2_": final_params[3],
        "phi0_": final_params[4],
        "d_": final_params[5],
    }


def ZABR_double_skew(X0: float, K: float, vol: float, eta_: float, rho_: float, beta1_: float, beta2_: float,
                     phi0_: float, d_: float):
    """
    :param X0: spot (input)
    :param K: strike (input)
    :param vol: ATM implied vol (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param phi0_: constant in local volatility function (param)
    :param d_: constant in local volatility function (param)
    :return: simple ZABR skew
    """
    K_neg_shifted = K - pow(10, -5)
    K_pos_shifted = K + pow(10, -5)
    vol_sabr_neg_shifted = ZABR_double(X0=X0, K=K_neg_shifted, vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_,
                                       beta2_=beta2_, phi0_=phi0_, d_=d_)
    vol_sabr_pos_shifted = ZABR_double(X0=X0, K=K_pos_shifted, vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_,
                                       beta2_=beta2_, phi0_=phi0_, d_=d_)
    return (vol_sabr_pos_shifted - vol_sabr_neg_shifted) / (K_pos_shifted - K_neg_shifted)


def ZABR_double_Durrleman_Condition(f: float, X0: float, vol: float, T: float, eta_: float, rho_: float, beta1_: float,
                                    beta2_: float, phi0_: float, d_:float, min_k=-1, max_k=1, nb_k=200):
    """
    :param f: forward (input)
    :param X0: spot (input)
    :param vol: ATM implied vol (input)
    :param T: maturity (input)
    :param eta_: constant vol of vol (param)
    :param rho_: spot vol constant correlation (param)
    :param beta1_: constant in local volatility function (param)
    :param beta2_: constant in local volatility function (param)
    :param phi0_: constant in local volatility function (param)
    :param d_: constant in local volatility function (param)
    :param min_k: first log forward moneyness
    :param max_k: last log forward moneyness
    :param nb_k: number of log forward moneyness
    :return: g list [g1, g2, g3, ...]
    """
    k_list = np.linspace(min_k, max_k, nb_k)
    vol_list = [ZABR_double(X0=X0, K=f*np.exp(k), vol=vol, eta_=eta_, rho_=rho_, beta1_=beta1_, beta2_=beta2_,
                            phi0_=phi0_, d_=d_) for k in k_list]
    tot_var_list = [T * pow(vol, 2) for vol in vol_list]
    log_forward_skew_list = []
    log_forward_convexity_list = []
    new_tot_var_list = []
    new_k_list = []
    k_step = (max_k - min_k) / nb_k
    for i in range(1, len(tot_var_list) - 1):
        new_k_list.append(k_list[i])
        new_tot_var_list.append(tot_var_list[i])
        log_forward_skew_list.append((tot_var_list[i+1] - tot_var_list[i-1]) / (2 * k_step))
        log_forward_convexity_list.append((tot_var_list[i + 1] - 2 * tot_var_list[i] + tot_var_list[i-1]) / pow(k_step, 2))
    return new_k_list, Durrleman_Condition(k_list=new_k_list, tot_var_list=new_tot_var_list,
                                           log_forward_skew_list=log_forward_skew_list,
                                           log_forward_convexity_list=log_forward_convexity_list)
