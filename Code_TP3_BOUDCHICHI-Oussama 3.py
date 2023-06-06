# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama 
# Finite difference method for HJB equations


# % Python version test (to be ignored if version >= 3.10)
import sys

v = sys.version_info[:2]
if float('.'.join(str(x) for x in v)) < 3.10:
    print('Please upgrade python version to >= 3.10')

# % Install necessary packages. To be ignored if the packages are already installed.

# import subprocess
# packages = ['scipy', 'numba', 'tabulate']
# for package in packages:
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
# % =========================================================================

# % Code begins here:

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
# from scipy.sparse import csr_matrix as sparse
# from scipy.sparse.linalg import spsolve
from numba import jit

plt.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.linewidth'] = 2 
plt.rcParams.update({"text.usetex": True})
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 

# % Eikonal Equation ====================================

# % Params : 
c, T, x_min, x_max = 1., 1., -3., 3.

u_right = lambda t: 0.
u_left = lambda t: 0.

u_0 = lambda x: -((np.maximum(1. - x**2, 0.))**2).reshape(-1, 1)

def u_exact(t, x):
    def temp_f(t, x):
        if x <= -c * t: return u_0(x + c * t)
        elif (x >= -c * t) & (x <= c * t): return u_0(0.)
        elif x >= c * t: return u_0(x - c * t)
    
    return np.vectorize(temp_f)(t, x)

u_exact_2 = lambda t, x: np.minimum(u_0(x - c * t), u_0(x + c * t))

# % Utils:
def create_mesh(I, N):
    dt = T / N 
    h = (x_max - x_min) / (I + 1)
    x = x_min + h * np.arange(1, I + 1)
    t = dt * np.arange(0, N + 1)

    return dt, h, t, x

def interpolate(x_new, x, y):
    idx = (np.abs(x - x_new)).argmin()
    dx = x[1] - x[0]
    out = ((x[idx + 1] - x_new) * y[idx] / dx) + (x_new - x[idx]) * y[idx + 1] / dx
    return out

def print_results(tab, wr = False, file_name = 'res'):
    headers = ['I', 'N', 'vals', 'e_k', 'alpha_k_x', 'alpha_k_t', 'eslaped_time']
    rows = []
    for i in range(len(tab['vals'])):
        rows.append([tab['vals'][i][-2], tab['vals'][i][-1], np.round(tab['vals'][i][0], 6), np.round(tab['e_k'][i], 6), np.round(tab['alpha_k_x'][i], 2), np.round(tab['alpha_k_t'][i], 2), np.round(tab['vals'][i][3], 3)])
    
    if wr:
        with open(file_name + '.txt', "w") as f:
            for col in headers:
                f.write(col + '\t')
            f.write('\n')
            for row in rows:
                for i in range(len(row)):
                    f.write(str(row[i]) + '\t')
                f.write('\n')

    print(tabulate(rows, headers, tablefmt="grid"))


 ## % Stiffness matrix assembly
@jit(nopython = True)
def create_A(upper, lower, diag, I):
    A = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            if i == j      : A[i, j] = diag[i]
            elif j == i + 1: A[i, j] = upper[i]
            elif j == i - 1: A[i, j] = lower[j]
    return A

@jit(nopython = True)
def create_D_tilde(up, uup, low, llow, diag):
    I = diag.shape[0]
    D = np.zeros((I, I))
    for i in range(I):
        for j in range(I):
            if i == j      : D[i, j] = diag[i]
            elif j == i + 1: D[i, j] = up[i]
            elif j == i - 1: D[i, j] = low[j]
            elif j == i + 2: D[i, j] = uup[i]
            elif j == i - 2: D[i, j] = llow[j]
    return D

@jit(nopython = True)
def newton(B, b, C, c, x_0, tol, maxIter):
    x = x_0
    it = 0
    err = 1. + tol
    F_tol = 1. + tol
    norm_inf = lambda x: np.linalg.norm(x, np.inf)
    while((it < maxIter) & (err > tol) & (F_tol > tol)):
        x_old = x
        it += 1
        F = np.maximum(B@x - b, C@x - c)

        F_p = C.copy()
        mask = (((B@x - b) -  (C@x - c)) >= 0.).reshape(x.shape[0])
        F_p[mask, :] = B[mask, :]
        x = x - np.linalg.inv(F_p)@F
        
        err = norm_inf(x - x_old)
        F_tol = norm_inf(np.maximum(B@x - b, C@x - c))
      
    return x, err, it, F_tol


SCHEME = 'EIK-BDF'  #'EIK-EI-1' #'EIK-RK2' #"EIK-EE-3" #"EIK-EE-1" # 'EIK-EE-2'

def check_cfl(I, N):
    dt = T / N 
    h = (x_max - x_min) / (I + 1)
    return c * dt / h


x_val = 1.5
Ilist = np.array([10 * 2**k for k in range(8)])
#Nlist = Ilist
Nlist = (Ilist / 10).astype(int)
errors = []
tab = {'vals' : [], 'e_k': [0.], 'alpha_k_x': [0., 0.], 'alpha_k_t': [0., 0.]}
tol, maxIter = 1e-10, 1000

print('Using Scheme: ', SCHEME)
for I, N in zip(Ilist, Nlist):
    start_time = time.time()
    dt, h, _, x = create_mesh(I, N)
    diag = (c / h) * np.ones(I)
    I_d = np.identity(I)

    match SCHEME:
        case 'EIK-EE-1':
            D_m = create_A(np.zeros(I - 1), -diag[1:], diag, I)
            D_p =  create_A(diag[1:], np.zeros(I - 1), -diag, I)
            def q_m(t):
                y = np.zeros((I, 1))
                y[0] =  -(c / h) * u_left(t)
                return y

            def q_p(t):
                y = np.zeros((I, 1))
                y[-1] =  -(c / h) * u_right(t)
                return y
        
        case 'EIK-EE-2':
            D = create_A(.5 * diag[1:], -.5 * diag[1:], np.zeros(I), I)
            def q(t):
                y = np.zeros((I, 1))
                y[0] =  -.5 * (c / h) * u_left(t)
                y[-1] = .5 * (c / h) * u_right(t)
                return y
            
        case 'EIK-EE-3':
            D_tilde_m = create_D_tilde(up  = np.zeros(I - 1),
                                       uup = np.zeros(I - 2),
                                       low = -2. * diag[1:],
                                       llow= .5 * diag[2:], 
                                       diag = 1.5 * diag)
            
            D_tilde_p = create_D_tilde(low  = np.zeros(I - 1),
                                       llow = np.zeros(I - 2),
                                       up = 2. * diag[1:],
                                       uup = -.5 * diag[2:], 
                                       diag = -1.5 * diag)
        
        case 'EIK-RK2':
            D_tilde_m = create_D_tilde(up  = np.zeros(I - 1),
                                       uup = np.zeros(I - 2),
                                       low = -2. * diag[1:],
                                       llow= .5 * diag[2:], 
                                       diag = 1.5 * diag)
            
            D_tilde_p = create_D_tilde(low  = np.zeros(I - 1),
                                       llow = np.zeros(I - 2),
                                       up = 2. * diag[1:],
                                       uup = -.5 * diag[2:], 
                                       diag = -1.5 * diag)
        case 'EIK-EI-1':
            D_m = create_A(np.zeros(I - 1), -diag[1:], diag, I)
            D_p =  create_A(diag[1:], np.zeros(I - 1), -diag, I)
            def q_m(t):
                y = np.zeros((I, 1))
                y[0] =  -(c / h) * u_left(t)
                return y

            def q_p(t):
                y = np.zeros((I, 1))
                y[-1] =  -(c / h) * u_right(t)
                return y
            
        case 'EIK-BDF':
            D_m = create_A(np.zeros(I - 1), -diag[1:], diag, I)
            D_p =  create_A(diag[1:], np.zeros(I - 1), -diag, I)
            D_tilde_m = create_D_tilde(up  = np.zeros(I - 1),
                                       uup = np.zeros(I - 2),
                                       low = -2. * diag[1:],
                                       llow= .5 * diag[2:], 
                                       diag = 1.5 * diag)
            
            D_tilde_p = create_D_tilde(low  = np.zeros(I - 1),
                                       llow = np.zeros(I - 2),
                                       up = 2. * diag[1:],
                                       uup = -.5 * diag[2:], 
                                       diag = -1.5 * diag)
            B = I_d + (2 / 3.) * dt * D_tilde_m
            C = I_d - (2 / 3.) * dt * D_tilde_p

    g = u_0(x)
    U = g.copy()
    if SCHEME == 'EIK-BDF':
        U_0 = g.copy() # U_old, i.e. U^{n-1}
    
    
    for n in range(N):
        match SCHEME:
            case 'EIK-EE-1':
                U = U - dt * np.maximum(D_m@U + q_m(n * dt), -(D_p@U + q_p(n * dt)))
            
            case 'EIK-EE-2':
                U = U - dt * np.abs(D@U + q(n * dt))

            case 'EIK-EE-3':
                U = U - dt * np.maximum(D_tilde_m@U, -D_tilde_p@U)
            
            case 'EIK-RK2':
                #u_old = U.copy()
                U_1 = U - dt * np.maximum(D_tilde_m@U, -D_tilde_p@U)
                U = U - .5 * dt * (np.maximum(D_tilde_m@U, -D_tilde_p@U) + np.maximum(D_tilde_m@U_1, -D_tilde_p@U_1))

            case 'EIK-EI-1':
                U, _, _, _ = newton(I_d + dt * D_m, U, I_d - dt * D_p, U, g, tol, maxIter)

            case 'EIK-BDF':
                if n == 0:
                    # 1st iteration using EI
                    U, _, _, _ = newton(I_d + dt * D_m, U_0, I_d - dt * D_p, U_0, g, tol, maxIter)
                
                else:
                    b = ((4. * U  - U_0) / 3.).copy()
                    U_0 = U
                    U, _, _, _ = newton(B, b, C, b, g, tol, maxIter)


    tcpu = time.time() - start_time
    tab['vals'].append((interpolate(x_val, x, U)[0], h, dt, tcpu, I, N))
    errors.append([dt, h, np.abs(u_exact(1., x_val) - interpolate(x_val, x, U))[0, 0]])

errors = np.array(errors)

for i in range(1, len(tab['vals'])):
    tab['e_k'].append(np.abs(tab['vals'][i][0] - tab['vals'][i - 1][0]))

for i in range(2, len(tab['e_k'])):
    tab['alpha_k_x'].append(np.log(tab['e_k'][i - 1]/ tab['e_k'][i]) / np.log(tab['vals'][i - 1][1] / tab['vals'][i][1]))
    tab['alpha_k_t'].append(np.log(tab['e_k'][i - 1]/ tab['e_k'][i]) / np.log(tab['vals'][i - 1][2] / tab['vals'][i][2]))
    
print_results(tab, True, 'EIK-BDF_case2')

u_exact(1., 1.5)



# % Scheme plots =======
plt.plot(x, U, '.--', color = 'firebrick', label = r'$v_{\mathrm{scheme}}$')
#plt.plot(x_2, U_2, '-', color = 'k', label = r'$v_{\mathrm{scheme}} (I = N = 800)$')
plt.plot(x, u_0(x), '--', color = 'darkmagenta', label = r'$v_0$')
plt.plot(x, u_exact(T, x), color = 'navy', label = r'$v_{\mathrm{exact}}$')
plt.legend(edgecolor = 'k')
plt.grid(linestyle = 'dotted')
plt.xlabel(r'$x$', fontdict={'fontsize': 18})
plt.ylabel(r'$v(t = 1, \ x)$', fontdict={'fontsize': 18})
plt.tight_layout()
#plt.savefig('v_sh_3.pdf', format = 'pdf')
plt.show()

# % Errors  plots =======

fig, ax1 = plt.subplots()
ax1.plot(np.log(errors[:, 0]), np.log(errors[:, -1]), '.--', color = 'k', label = r'$\Delta t \mapsto e_{\Delta t}$')
ax2 = ax1.twiny()
ax2.plot(np.log(errors[:, 1]), np.log(errors[:, -1]), '.--', color = 'darkmagenta', label = r'$h \mapsto  e_h$')

plt.grid(linestyle = 'dotted')
ax1.set_xlabel(r'$\Delta t$', fontdict={'fontsize': 18})
ax2.set_xlabel(r'$h$', fontdict={'fontsize': 18}, color = 'darkmagenta')
ax1.set_ylabel(r'$e_{\Delta t, h}(t = 1, x = 1.5)$', fontdict={'fontsize': 18})
ax2.tick_params(axis='x', colors='darkmagenta')

fig.legend(edgecolor = 'k')
plt.tight_layout()
#plt.savefig('err.pdf', format = 'pdf')
plt.show()

# % Orders approx. using exact errors
from scipy.stats import linregress
linregress(np.log(errors[:, 0]), np.log(errors[:, -1]))



# % Uncertain vol. model: ===================================

# % Exog. params:
T, x_min, x_max = .5, -3., 3.


@jit(nopython = True)
def newton_2(B, b, C, c, x_0, tol, maxIter):
    x = x_0
    it = 0
    err = 1. + tol
    F_tol = 1. + tol
    norm_inf = lambda x: np.linalg.norm(x, np.inf)
    
    while((it < maxIter) & (err > tol) & (F_tol > tol)):
        x_old = x
        it += 1
        F = np.minimum(B@x - b, C@x - c)

        F_p = C.copy()
        mask = (((B@x - b) -  (C@x - c)) <= 0.).reshape(x.shape[0])
        F_p[mask, :] = B[mask, :]
        x = x - np.linalg.inv(F_p)@F
        
        err = norm_inf(x - x_old)
        F_tol = norm_inf(np.minimum(B@x - b, C@x - c))
        
    return x, err, it, F_tol

def sgn(x):
    return np.array([1. * (x_ > 0.) - 1. * (x_ <= 0.) for x_ in x])

def check_cfl_uv(I, N):
    dt = T / N 
    h = (x_max - x_min) / (I + 1)
    return  .5 * dt / (h**2)

u_0 = lambda x:(sgn(x) * ((np.maximum(1. - np.abs(x), 0.))**4 - 1.)).reshape(-1, 1)
u_left = lambda x: 1.
u_right = lambda x: -1.

check_cfl_uv(159, 51200)
check_cfl_uv(79, 12800)

Ilist = np.array([10 * 2**k for k in range(8)]) 
#Nlist = (2. * (Ilist + 1)**2).astype(int)
#Nlist = Ilist 
Nlist = (Ilist / 10).astype(int)

tab = {'vals' : [], 'e_k': [0.], 'alpha_k_x': [0., 0.], 'alpha_k_t': [0., 0.]}
tol, maxIter, verbose = 1e-10, 1000, True
x_val = 1.5
print('Using Scheme: ', SCHEME)
for I, N in zip(Ilist, Nlist):
    start_time = time.time()
    dt, h, _, x = create_mesh(I, N)
    up = ((1. / h)**2) * np.ones(I)
    D = create_A(up[1:], up[1:], -2. * up, I)
    I_d = np.identity(I)

    def q(t):
        y = np.zeros((I, 1))
        y[0] =   (1. / h**2) * u_left(t)
        y[-1] = (1. / h**2) * u_right(t)
        return y

    g = u_0(x)
    U = g.copy()
    if SCHEME == "UV-BDF2":
        B = I_d - (dt / 3.) * D
        U_0 = g.copy()

    for n in range(N):
        match SCHEME:
            case 'UV-EE':
                U = U - dt * np.minimum(-.5 * (D@U + q(n * dt)), 0.)

            case 'UV-EI':
                U, _, _, _ = newton_2(I_d - .5 * dt * D, U + .5 * dt * q((n  + 1) * dt), I_d, U, g, tol, maxIter)

            case 'UV-BDF2':
                    if n == 0:
                        # 1st iteration using EI
                        U, _, _, _ = newton_2(I_d - .5 * dt * D, U_0 + .5 * dt * q((n  + 1) * dt), I_d, U_0, g, tol, maxIter)
                    
                    else:
                        c = ((4. * U  - U_0) / 3.).copy()
                        b = ((4. * U  - U_0 + dt * q((n + 1) * dt)) / 3.).copy()
                        U_0 = U
                        U, _, _, _ = newton_2(B, b, I_d, c, g, tol, maxIter)

    tcpu = time.time() - start_time
    tab['vals'].append((interpolate(x_val, x, U)[0], h, dt, tcpu, I, N))


for i in range(1, len(tab['vals'])):
    tab['e_k'].append(np.abs(tab['vals'][i][0] - tab['vals'][i - 1][0]))

for i in range(2, len(tab['e_k'])):
    tab['alpha_k_x'].append(np.log(tab['e_k'][i - 1]/ tab['e_k'][i]) / np.log(tab['vals'][i - 1][1] / tab['vals'][i][1]))
    tab['alpha_k_t'].append(np.log(tab['e_k'][i - 1]/ tab['e_k'][i]) / np.log(tab['vals'][i - 1][2] / tab['vals'][i][2]))
    
print_results(tab, True, 'UV-BDF2_case2')




# % Plots: 
plt.plot(x, U, '.--', color = 'firebrick', label = r'$v_{\mathrm{scheme}}$')
plt.plot(x, u_0(x), '--', color = 'navy', label = r'$v_0$')
plt.legend(edgecolor = 'k')
plt.grid(linestyle = 'dotted')
plt.xlabel(r'$x$', fontdict={'fontsize': 18})
plt.ylabel(r'$v(t = T, \ x)$', fontdict={'fontsize': 18})
plt.tight_layout()
#plt.savefig('uv_v_sch.pdf', format = 'pdf')
plt.show()