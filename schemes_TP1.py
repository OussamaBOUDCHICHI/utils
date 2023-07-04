# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama & EL BADAOUI Assia
# Finite difference method for a European option


import numpy as np
import scipy.stats as stats
import time
import yaml
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2 
from scipy.sparse import csr_matrix as sparse
from scipy.sparse.linalg import spsolve
from tabulate import tabulate


def create_A(upper, lower, diag, I, SPARSE = False):
    A = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            if i == j: A[i, j] = diag[i]
            elif j == i + 1: A[i, j] = upper[i]
            elif j == i - 1: A[i, j] = lower[j]
            
    if SPARSE:
        return sparse(A)
    else:
        return A




def theta_scheme(x_min, x_max, t_min, t_max, 
                 u_left, u_right, u_0, q, I, N, theta,
                 alphas, betas, r, SPARSE = False, return_A = False):
    

    dt = (t_max - t_min) / N 
    dx = (x_max - x_min) / (I + 1)

    x = x_min + dx * np.arange(0, I + 2)
    t = t_min + dt * np.arange(0, N + 1)

    U = np.zeros((I + 2, N + 1))
    # INIT
    U[1:-1, 0] = u_0(x[1:-1])

    # BOUNDARY VALUES
    U[0, :] = u_left(t)
    U[-1, :] = u_right(t)

    upper = -(alphas[:-1] + betas[:-1])
    lower =  betas[1:] - alphas[1:]
    diag = 2 * alphas + r

    A = create_A(upper, lower, diag, I, SPARSE)

    if SPARSE:
        SOLVE = spsolve
        I_d = sparse(np.identity(I))
    else:
        SOLVE = np.linalg.solve
        I_d = np.identity(I)
    
    for n in range(1, N + 1):
        b = (I_d - (1. - theta) * dt * A)@U[1:-1, n - 1] - dt * (theta * q(t[n]) + (1. - theta) * q(t[n - 1]))
        U[1:-1, n] = SOLVE(I_d + theta * dt * A, b)

    if return_A:
        return U, A
    else:
        return U






def FD_BS(I, N, SCHEME = "EI", SPARSE = False, verbose = False, return_A = False):
    assert((SCHEME == "EE") | (SCHEME == "EI") | (SCHEME == "CN"))
    start_time = time.time()
    dic = {'EE': 0., "EI": 1., "CN": .5}

    # LOADING PARAMS
    with open('params.yaml', 'r') as f:
        pars = yaml.safe_load(f)['params']
    
    S_min, S_max, K, sigma, T, r = pars.values()
    dt = T / N
    ds = (S_max - S_min) / (I + 1)

    s = S_min + ds * np.arange(0, I + 2)
    t = dt * np.arange(0, N + 1)

    # alphas = .5 * dt * (sigma * s[1:-1] / ds) ** 2
    # betas = .5 * r * s[1:-1] / ds
    alphas = (0.5 * (sigma / ds)**2) * (s**2)
    betas = r / (2 * ds) * s
    

    # BORDER CONDITIONS
    def u_left(t):
        return K * np.exp(-r * t) - S_min

    def u_right(t):
        return 0.

    def u_0(x):
        return np.maximum(K - x, 0.)

    def q(t):
        y = np.zeros(I)
        y[0] = (-alphas[0] + betas[0]) * u_left(t)
        y[-1] = -(alphas[-1] + betas[-1]) * u_right(t)
        return y
    if return_A:
        U, A = theta_scheme(S_min, S_max, 0., T, u_left, u_right, u_0, q, 
                 I, N, dic[SCHEME], alphas, betas, r, SPARSE, return_A)
    else:
        U = theta_scheme(S_min, S_max, 0., T, u_left, u_right, u_0, q, 
                 I, N, dic[SCHEME], alphas, betas, r, SPARSE, return_A)
    esl_time = time.time() - start_time
    if verbose:
        print("%s seconds" % esl_time)
    if return_A:
        return U, t, s, dt, ds, u_0, esl_time, A
    else:
        return U, t, s, dt, ds, u_0, esl_time





def INTERPOL(n, Sval, s, U):
    idx = (np.abs(s - Sval)).argmin()
    ds = s[1] - s[0]
    out = ((s[idx + 1] - Sval) * U[idx, n] / ds) + (Sval - s[idx]) * U[idx + 1, n]/ ds
    return out
    
norm_cdf = stats.norm.cdf
def BS_closed(t, S):

    with open('params.yaml', 'r') as f:
        pars = yaml.safe_load(f)['params']
    
    _, _, K, sigma, T, r = pars.values()
    d_1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d_2 = d_1 - sigma * np.sqrt(t)

    return K * np.exp(-r * t) * norm_cdf(-d_2) - S * norm_cdf(-d_1)


I, N = 10, 30
U, t, s, _, _, u_0, _ = FD_BS(I, N, "CN")
plt.rcParams.update({
  "text.usetex": True
})
plt.plot(s, U[:, -1], ".--", color = 'firebrick', label = 'Crank-Nicholson Approximation ' + r'($\theta = 1/2$)')
plt.plot(s, u_0(s), color = 'navy', label = 'Payoff function')
plt.plot(s, BS_closed(1., s), '.--', color = '#28629d', label = 'Exact solution')
plt.xlabel(r'$S$')
plt.legend()
#plt.savefig('CN.pdf', format = 'pdf')
plt.show()






# ======================================================
# 3. Error/order analysis
# ======================================================

def compute_approx_errors(I, N, SCHEME = "EI", SPARSE = False, norm = '2'):
    assert((norm =='2') | (norm == 'inf'))

    U, t, s, _, _, _, _ = FD_BS(I, N, SCHEME, SPARSE)
    errors = np.zeros((I + 2, N + 1))
    for i in range(len(t)):
        errors[:, i] = BS_closed(t[i], s) - U[:, i]
    if norm == '2':
        return t, s, errors, np.linalg.norm(errors, 2, axis = 0).max()
    elif norm =='inf':
        return t, s, errors, np.linalg.norm(errors, np.inf, axis = 0).max()
    

def compute_error(I, N, Sval, n, SCHEME = "EI", SPARSE = False):
    U, t, s, _, _, _,_ = FD_BS(I, N, SCHEME, SPARSE)
    
    return  BS_closed(t[n], Sval) - INTERPOL(n, Sval, s, U)



Ilist = np.array([10, 20, 40, 80, 160, 320])
#Nlist = np.round(Ilist ** 2 / 10.).astype(int)
Nlist = Ilist
#Nlist = (Ilist / 10).astype(int)
SCHEME = "EI"
Sval = 80.
# for I, N in zip(Ilist, Nlist):
#     print(compute_error(I, N, Sval, -1))


norm_2 = lambda x: np.linalg.norm(x, 2)
norm_inf = lambda x: np.linalg.norm(x, np.inf)

norms = []
for I, N in zip(Ilist, Nlist):
    _, _, _, dt, _, _, _, A = FD_BS(I, N, SCHEME, return_A=True)
    B = np.identity(I) - dt * A
    norms.append([I, N, norm_2(B), norm_inf(B), np.all(B.diagonal() > 0), np.any(abs(B.diagonal()) >= 1)])

headers = ['I', 'N', '||B||_2', '||B||_oo', 'All diag positive?', 'coeff of modulus > 1?']
print(tabulate(norms, headers=headers))


tab = {'vals' : [], 'e_k': ['-'], 'alpha_k': ['-', '-']}
for I, N in zip(Ilist, Nlist):
    U, t, s,_, ds, _, esl_t = FD_BS(I, N, SCHEME)
    tab['vals'].append((INTERPOL(-1, Sval, s, U), ds, esl_t, I, N))


for i in range(1, len(tab['vals'])):
    tab['e_k'].append(tab['vals'][i][0] - tab['vals'][i - 1][0])

for i in range(2, len(tab['e_k'])):
    tab['alpha_k'].append(np.log(tab['e_k'][i]/ tab['e_k'][i - 1]) / np.log(tab['vals'][i][1] / tab['vals'][i - 1][1]))

err = {'ds': [], 'e_k':[]}

space_steps = [x[1] for x in tab['vals'][1:]]
errors = tab['e_k'][1:]
space_steps.reverse()
errors.reverse()
err['ds'].append(space_steps)
err['e_k'].append(errors)

plt.rcParams.update({
  "text.usetex": True
})
plt.loglog(err['ds'][0], np.abs(err['e_k'][0]), '.--', color = 'firebrick', label = 'CN')
plt.loglog(err['ds'][1], np.abs(err['e_k'][1]), '.-', color = 'navy', label = 'EI')
plt.xlabel(r'$\log(h)$')
plt.ylabel(r'$\log(e_k)$')
plt.legend()
#plt.savefig('err_EE.pdf', format = 'pdf')
plt.show()
x, y = np.log(err['ds'][0]), np.log(np.abs(err['e_k'][0]))
x, y
err['e_k']

def print_results(tab, wr = False, file_name = 'res'):
    headers = ['I', 'N', 'vals', 'e_k', 'alpha_k', 'eslaped_time']
    rows = []
    for i in range(len(tab['vals'])):
        rows.append([tab['vals'][i][-2], tab['vals'][i][-1],tab['vals'][i][0], tab['e_k'][i], tab['alpha_k'][i], tab['vals'][i][2]])
    print(tabulate(rows, headers, tablefmt="grid"))

    if wr:
        with open(file_name + '.txt', "w") as f:
            for col in headers:
                f.write(col + '\t')
            f.write('\n')
            for row in rows:
                for i in range(len(row)):
                    f.write(str(row[i]) + '\t')
                f.write('\n')
    


print_results(tab, wr= True, file_name="res_CN_N=Idiv10")



# ======================================================
# 4. SPARSE implementation
# ======================================================

I, N = 20, 40
_, _, _, _, _, _, es_t_0 = FD_BS(I, N, SCHEME='EE', SPARSE=True)
_, _, _, _, _, _, es_t_1 = FD_BS(I, N, SCHEME='EE')

es_t_0, es_t_1

# ======================================================
# 5. Exact errors analysis
# ======================================================

Ilist = np.array([10, 20, 40, 80, 160, 320])
#Nlist = np.round(Ilist ** 2 / 10.).astype(int)
Nlist = Ilist
#Nlist = (Ilist / 10).astype(int)
SCHEME = "CN"
Sval = 80.

tab = {'vals' : [], 'e_k': [], 'alpha_k': ['-']}
for I, N in zip(Ilist, Nlist):
    U, t, s,_, ds, _, esl_t = FD_BS(I, N, SCHEME)
    tab['vals'].append((INTERPOL(-1, Sval, s, U), ds, esl_t, I, N))
    tab['e_k'].append(compute_error(I, N, Sval, -1, SCHEME))

for i in range(1, len(tab['e_k'])):
    tab['alpha_k'].append(np.log(tab['e_k'][i]/ tab['e_k'][i - 1]) / np.log(tab['vals'][i][1] / tab['vals'][i - 1][1]))
tab

print_results(tab, wr = True, file_name='ex_err_CN')

t, s, errors, err_norm = compute_approx_errors(20, 20, 'CN', False)
c = plt.imshow(errors, cmap='viridis', interpolation='nearest')
plt.colorbar(c)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
T, X = np.meshgrid(t, s)


plt.rcParams["figure.autolayout"] = False
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, np.abs(errors), cmap='viridis', label = 'Errors')
ax.view_init(20, 55)
ax.set_xlabel(r'$S$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$|e_j^n|$')
fig.savefig('err.pdf', format = 'pdf', )
fig.show()