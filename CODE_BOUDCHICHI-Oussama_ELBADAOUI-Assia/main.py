# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# Recursive Computation of VaR and CVaR using MC and QMC
# Driver code

# % Python version test (to be ignored if version >= 3.10)
import sys

v = sys.version_info[:2]
if float('.'.join(str(x) for x in v)) < 3.10:
    print('Please upgrade python version to >= 3.10')


# % Install necessary packages. To be ignored if the packages are already installed.
# import subprocess
# packages = ['numpy', 'scipy', 'cython', 'yaml']
# for package in packages:
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# % =========================================================================

# % Code begins here:
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from robbins_monro import RobbinsMonro
from _sobol_cy import Sobol

plt.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.linewidth'] = 2 


# % Example 1: =========================================================
# % Exog. params.
K = 110.
S_0 = 100.
sig = .2
P_0 = 10.7
r = .05
T = 1.
alpha = .95
a = sig * np.sqrt(T)

def phi(x):
    """ 
    The loss function

    Parameters
    ----------
    x: numpy.ndarray or number
    """
    S_T = S_0 * np.exp((r - .5 * sig**2) * T + sig * np.sqrt(T) * x)
    return np.maximum(K - S_T, 0.) - np.exp(r * T) * P_0



norm = np.linalg.norm
def H(Theta, alpha, x, phase = 'I'):
    """ 
    The function appearing in the expectation representation of the mean func.

    Parameters
    ----------

    Theta: numpy.ndarray
           RM procedure vector
    alpha: float
           confidence level
    x    : np.ndarray
           innovation
    phase: str {'I', 'II'}
           phase of the approx. algorithm
    """
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

x_0 = np.empty(4, dtype = object)
x_0[0] = 300.
x_0[1] = 0.
x_0[2]= 1.
x_0[3] = 1.

rm_steps = [20_000, 40_000, 60_000]
Ms = [1500, 3000, 4500]
alphas = [.95, .99, .995, .999]
rng = np.random.randn
#rng = s.get_1dgaussian # If one wishes to use a Sobol sequence instead of pseudo-random numbers

# % ⚠️ ⚠️ ⚠️ ⚠️ 
# % the following chunck takes a lot of time to run!!
# % ➞ We suggest to run file: "example1_mp.py" instead, as it's implemented using a parallelization procedure.
# % ===========================================================================
gamma = lambda n:  1. / (n**(3/4) + 100.)
with open('res' + '.txt', "w") as f:
    f.write('RM_steps\talpha\tVaR\tCVaR\tVR_VaR\tVR_CVaR\n')
    for M, n_iter in zip(Ms, rm_steps):
        print('RM_steps: ', n_iter)

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
            F_1 = lambda x, theta : (phi(x + theta) >= xi_star) * np.exp(-x * theta - .5 * theta**2)
            F_2 = lambda x, mu: np.maximum(phi(x + mu) - xi_star, 0.) * np.exp(-x * mu - .5 * mu**2)

            v_var_is = np.var(F_1(rng(10_000), theta_star), ddof = 1)
            v_var = np.var(F_1(rng(10_000), 0), ddof = 1)

            v_cvar_is = np.var(F_2(rng(10_000), mu_star), ddof = 1)
            v_cvar = np.var(F_2(rng(10_000), 0.), ddof = 1)

            vr_VaR = v_var / v_var_is
            vr_CVaR = v_cvar / v_cvar_is
            f.write(str(xi_star) + '\t' + str(C_star) + '\t' + str(vr_VaR) + '\t' + str(vr_CVaR) + '\n')

f.close()
# % ============================================================================

# Function to estimate the valuye of the potential V (obj. func) at xi using MC
def compute_objective(n_simul, xi, alpha):
    """ 
    a function to evaluate the objective function value at xi, using MC (n_simul samples) 
    """
    x = rng(n_simul)
    return xi + (1. / (1. - alpha)) * np.mean(np.maximum(phi(x) - xi, 0.))


# % Example 2. ================================================================= 
T, K_C, K_P, S_0 = .25, 130., 110., 120.
norm_cdf = stats.norm.cdf

def bs_closed(flag = 'C'):
    """ 
    Black-Scholes pricing function. 
    """
    match flag:
        case 'C': d_1 = (np.log(S_0 / K_C) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        case 'P': d_1 = (np.log(S_0 / K_P) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d_2 = d_1 - sig * np.sqrt(T)
    match flag:
        case 'C': return S_0 * norm_cdf(d_1) - K_C * np.exp(-r * T) * norm_cdf(d_2)
        case 'P': return K_P * np.exp(-r * T) * norm_cdf(-d_2) - S_0 * norm_cdf(-d_1)

def phi(x):
    X_T = S_0 * np.exp((r - .5 * sig**2) * T + sig * np.sqrt(T) * x)
    out = (np.maximum(K_P - X_T, 0.) + np.maximum(X_T - K_C, 0.)).sum()
    return 10 * (out - 5 * np.exp(r * T) * (bs_closed('C') + bs_closed('P')))


x_0 = np.empty(4, dtype = object)
x_0[0] = 300.
x_0[1] = 0.
x_0[2]= np.zeros(5)
x_0[3] = np.zeros(5)

print('\nStoch. Approx. : ')
n_iter = 50_000
M = 1500

gamma = lambda n:  10. / (n**(3/4) + 200.)

alpha_1 = lambda m: .5 * ((m >= 1) & (m <= int(M / 3))) + .8 * ((m > int(M / 3)) & (m <= int(2 * M / 3))) + alpha * (m > int(2 * M / 3)) 
alpha_2  = lambda n: alpha

func_1 = lambda theta, x, alpha_1: H(theta, alpha_1, x, 'I')
func_2 = lambda theta, x, alpha_2: H(theta, alpha_2, x, 'II')

rng = lambda n = 1: np.random.multivariate_normal(np.zeros(5), np.identity(5), size=n).reshape((5,))

# % PHASE I:
RM = RobbinsMonro(x_0, gamma, func_1, rng, alpha_1)
RM.get_target(M)
RM.print_vals()

# % PHASE II:
RM.init_value = RM.x
RM.init_value[1] = 0.
RM.H = func_2
RM.extra_args = alpha_2
RM.reset()

RM.get_target(n_iter)
RM.print_vals()

xi_mc = RM.history_x


def compute_objective(n_simul, xi, alpha):
    sample = np.random.multivariate_normal(np.zeros(5), np.identity(5), size=n_simul)
    y = np.array([phi(x) for x in sample])
    return xi + (1. / (1. - alpha)) * np.mean(np.maximum(y - xi, 0.))

xi = np.linspace(90., 400., 1000)
V = np.array([compute_objective(1000, i, .95) for i in xi])
plt.rcParams.update({
  "text.usetex": True
})


plt.plot(xi, V, '-', color = 'navy')
# plt.plot(xi_phase2[:, 0], xi_phase2[:, 1], '.--', color = 'firebrick', label = 'batch')
# plt.plot(polyak[:, 0], polyak[:, 1], '--', color = 'navy', label = 'polyak')
# plt.plot(xi_phase1[:, 0], xi_phase1[:, 1], '.--', color = '#28629d')
# plt.plot([xi_phase1[-1, 0], xi_phase1[-1, 0]], [xi_phase1[-1, 1], 0.])
# plt.xlabel(r'$\xi_n$')
# plt.ylabel(r'$C_n$')
# plt.scatter(24.6, 29.9, s = 50, color = '#B42C4E', label = r'$(\xi^\star, C^\star)$')
# plt.legend()
plt.axvline(x = 290, linestyle = '--', color = '#B42C4E')
plt.axvline(x = 200, linestyle = '--', color = '#B42C4E')
plt.axhline(y = 260, linestyle = '--', color = '#B42C4E')
plt.axhline(y = 350, linestyle = '--', color = '#B42C4E')

plt.xlabel(r'$\xi$')
plt.ylabel(r'$V(\xi)$')
plt.tight_layout()
plt.savefig('V.pdf', format = 'pdf')
plt.show()

print('=======================================================')
print('Quasi Stoch.approx. :')
# % ============================================================================

# % QMC framework ============================================================


def H_qmc(Theta, x, alpha):
    """  
    The function appearing in the exp. repre. of the mean function in
    the QMC framework
    """
    xi, c = Theta
    H_1 = 1. - (1./ (1. - alpha)) * (phi(x) >= xi)
    H_2 = c - (xi + np.maximum(phi(x) - xi, 0.) / (1. - alpha))
    out = np.empty(2, dtype = object)
    out[0], out[1] = H_1, H_2
    return out


RM.init_value
n_iter = 50_000
x_0 = np.array([RM.init_value[0], 0.])

#s = Sobol(2)  ; For example 1, i.e where X id 1d
s = Sobol(6) # ; For example 2, i.e where X id 5d

#rng_qmc = s.get_1dgaussian # rng For example 1.

# rng for Example 2.
def rng_qmc(n = 1):
    sobol_seq = s.get(n)
    out = np.zeros_like(sobol_seq)
    for k in range(out.shape[0]):
        for j in range(0, out.shape[1] - 1, 2):
            out[k, j] = np.sqrt(-2 * np.log(sobol_seq[k, j])) * np.cos(2 * np.pi * sobol_seq[k, j + 1])
            out[k, j + 1] = np.sqrt(-2 * np.log(sobol_seq[k, j])) * np.sin(2 * np.pi * sobol_seq[k, j + 1])
    return out[:, :-1]


#gamma = lambda n: 1. / (n + 100.)
gamma = lambda n:  10. / (n**(3/4) + 200.)

RM_qmc = RobbinsMonro(x_0, gamma, H_qmc, rng_qmc, lambda n: alpha)
RM_qmc.get_target(n_iter)
RM_qmc.print_vals()

hist = RM_qmc.history_x[:, 0]
differ_2 = np.abs(hist - RM_qmc.x[0])[:-1]


xi_qmc = RM_qmc.history_x


plt.plot(np.arange(xi_mc[M-1:, 0].shape[0]), xi_mc[M-1:, 0], '--', color = 'firebrick', label = r'$\mathrm{VaR}_\mathrm{IS}$')
plt.plot(np.arange(xi_qmc.shape[0]), xi_qmc[:, 0], color = 'navy', label = r'$\mathrm{VaR}_\mathrm{QMC}$')
plt.axhline(y = 226., linestyle = 'dashed', color = 'k', label = r'$\xi_\alpha^\star$')
plt.legend()
plt.xlabel(r'$n$')
plt.ylabel(r'$\left|\xi_n - \xi_\alpha^\star\right|$')
plt.tight_layout()
plt.savefig('conv_ex2.pdf', format = 'pdf')
plt.show()

# % Code ends here.
# % ============================================================================
