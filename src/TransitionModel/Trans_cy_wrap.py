"""

This file contains small wrappers for the underlying cython modules
"""

import src.C_Extensions.Trans_cy as Tr_KM03
import src.C_Extensions.Trans_cy_TKF91 as Tr_TKF91
import src.SamplerSettings.Prior as Prior


def Trans_KM03(a, r, time):
    return Tr_KM03.Transition_km03(a=a, r=r, time=time, dmax_r=Prior.dmax_trans_r, dmax_a=Prior.dmax_trans_a)


def Trans_TKF91(mu, lambd, time):
    return Tr_TKF91.Transition_TKF91(time=time, lambd=lambd, mu=mu, dmax=Prior.dmax_trans)
