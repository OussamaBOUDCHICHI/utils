# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# Multiprocessing implementation of example_1 results code.


import multiprocessing as mp
import numpy as np
from robbins_monro import RobbinsMonro
import os
import yaml


# Open "params.yaml" and change the numerical setup if u wish so.
with open('params.yaml', 'r') as f:
    S_0, K, sig, T, r, P_0 = yaml.safe_load(f)['exog_params'].values()

a = sig * np.sqrt(T)

def phi(x):
    S_T = S_0 * np.exp((r - .5 * sig**2) * T + sig * np.sqrt(T) * x)
    return np.maximum(K - S_T, 0.) - np.exp(r * T) * P_0


norm = np.linalg.norm
def H(Theta, alpha, x, phase = 'I'):
    xi, c, thet, mu = Theta

    match phase:
        case 'I':
            H_1 = 1. - (1./ (1. - alpha)) * (phi(x) >= xi) 
        case 'II':
            H_1 =  np.exp(-norm(thet)**2)* ( 1. - (phi(x + thet) >= xi) * np.exp(-np.dot(x, thet) - .5 * norm(thet)**2) / (1. - alpha))
    
    H_2 = c - (xi + np.maximum(phi(x + mu) - xi, 0.) * np.exp(-np.dot(x, mu) - .5 * norm(mu)**2)/ (1. - alpha))

    K_1 = (phi(x - thet) >= xi) * (2. * thet - x)  
    K_2 =   np.exp(- a * (norm(mu)**2 + 1.)) * (2 * mu - x) * (np.maximum(phi(x - mu) - xi, 0.))**2 
    out = np.empty(4, dtype=object)
    out[0], out[1], out[2], out[3] = H_1, H_2, K_1, K_2
    return out

def main(args):
    n_iter, M, alphas = args
    x_0 = np.empty(4, dtype = object)
    x_0[0] = 10.
    x_0[1] = 0.
    x_0[2]= 1.
    x_0[3] = 1.

    np.random.seed(n_iter)
    rng = np.random.randn
    
    gamma = lambda n:  10. / (n**(3/4) + 200.)

    with open('./results/res_' + str(n_iter) + '.txt', "w") as f:
        f.write('RM_steps\talpha\tVaR\tCVaR\tVR_VaR\tVR_CVaR\n')
        
        for alpha in alphas:
            print('alpha: ', alpha)
            f.write(str(n_iter) + '\t' + str(alpha) + '\t')
            alpha_1 = lambda m: .5 * ((m >= 1) & (m <= int(M / 3))) + .8 * ((m > int(M / 3)) & (m <= int(2 * M / 3))) + alpha * (m > int(2 * M / 3)) 
            alpha_2  = lambda n: alpha

            func_1 = lambda theta, x, alpha_1: H(theta, alpha_1, x, 'I')
            func_2 = lambda theta, x, alpha_2: H(theta, alpha_2, x, 'II')

            # % PHASE I:
            RM = RobbinsMonro(x_0, gamma, func_1, rng, alpha_1)
            RM.get_target(M)

            # % PHASE II:
            RM.init_value = RM.x
            RM.init_value[1] = 0.
            RM.H = func_2
            RM.extra_args = alpha_2
            RM.reset()
            RM.get_target(n_iter)

            # % Results
            xi_star, C_star, theta_star, mu_star = RM.x
            
            # F_1, F_2 : functions used in the MC simulation to estimate asymptotic variances of both procedures (i.e. the naive one and the RM-IS proc.)
            F_1 = lambda x, theta : (phi(x + theta) >= xi_star) * np.exp(-x * theta - .5 * theta**2)
            F_2 = lambda x, mu: np.maximum(phi(x + mu) - xi_star, 0.) * np.exp(-x * mu - .5 * mu**2)

            v_var_is = np.var(F_1(rng(10_000), theta_star), ddof = 1)
            v_var = np.var(F_1(rng(10_000), 0.), ddof = 1)

            v_cvar_is = np.var(F_2(rng(10_000), mu_star), ddof = 1)
            v_cvar = np.var(F_2(rng(10_000), 0.), ddof = 1)

            vr_VaR = v_var / v_var_is
            vr_CVaR = v_cvar / v_cvar_is
            f.write(str(xi_star) + '\t' + str(C_star) + '\t' + str(vr_VaR) + '\t' + str(vr_CVaR) + '\n')

    f.close()

if __name__ == '__main__':

    root = os.getcwd()
    project_root = os.path.join(root, 'results')
    os.makedirs(project_root, exist_ok = True)
    
    with open('params.yaml', 'r') as f:
        pars = list(yaml.safe_load(f).values())
    
    args = []
    for par in pars[1:]:
        args.append(list(par.values()))
    
    nb_pools = 6
    with mp.Pool(nb_pools) as p:
        p.map(main, args)



