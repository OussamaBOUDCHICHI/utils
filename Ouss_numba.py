
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


T, s_0, sig_0, y_0, rho, gamma, kappa = 1., 100., .15, 0., -.5, .5, 1.
sig_loc = .15


n_steps = 100
N_1 = 10_000
dt = 1. / n_steps
n_xgrid = 30
leverage = np.zeros((n_steps, n_xgrid))


Kern = lambda x: ((x + 1.) ** 2) * ((1. - x)**2) * ((x <= 1.) & (x >= -1.))


def kern_reg(x, x_data, y_data, bw, kern = Kern):
  weights = np.array([Kern((x - x_i) / bw) / bw for x_i in x_data])

  return np.dot(weights, y_data) / weights.sum()



rho_bar = rho * np.sqrt(2. * (1. - np.exp(-kappa * dt)) / (kappa * dt * (1. + np.exp(-kappa * dt))))


rng = lambda n, d=2: np.random.randn(n, d)


q_ln = lambda q, k:  stats.lognorm.ppf(q, loc = s_0, s = sig_loc * np.sqrt(k * dt))



y = y_0 * np.ones(N_1)
ls = np.log(s_0) * np.ones(N_1)
leverage[0, :] = (sig_loc / (sig_0 * np.exp(y_0))) * np.ones(n_xgrid)
x_grid = np.zeros((n_steps, n_xgrid))
x_grid[0, :] = s_0 * np.ones(n_xgrid)

for i in range(1, n_steps):
  z = rng(N_1)
  z_1, z_2 = z[:, 0], z[:, 1]
  if i == 1:
    lev_inter = lambda x: leverage[0, 0]
  else:
    lev_inter = interp1d(x_grid[i - 1, :], leverage[i - 1, :], kind = 'linear', fill_value='extrapolate')
  
  ls = ls - .5 * dt * (sig_0 * np.exp(y) * lev_inter(np.exp(ls)))**2  + sig_0 * np.exp(y) * lev_inter(np.exp(ls)) * np.sqrt(dt) * (np.sqrt(1 - rho_bar**2) * z_1 + rho_bar * z_2)
  s = np.exp(ls)
  y = y * np.exp(-kappa * dt) + gamma * np.sqrt((1. - np.exp(-2. * kappa * dt)) / (2. * kappa)) * z_2
  
  a = sig_0 * np.exp(y)
  bw = kappa * sig_loc * s_0 * np.sqrt(np.maximum(i * dt, .15)) / (N_1**(.2))
  x_min, x_max = q_ln(.05, i), q_ln(.95, i)
  x_grid[i, :] = np.linspace(x_min, x_max, n_xgrid)
  leverage[i, :] = sig_loc / np.array([np.sqrt(kern_reg(x, s, a**2, bw)) for x in x_grid[i, :]])


print(leverage)


N_2 = 1_000
strikes = np.array([70., 80., 90., 100., 110., 120., 130., 140.])
Z = np.random.randn(N_2, n_steps - 1, 2)


def get_price(K):
  m = 0.
  
  for j in range(N_2):
    y = y_0
    ls = np.log(s_0)
    #z = rng(n_steps - 1)
    
    for i in range(1, n_steps):
      #z = rng()
      z_1, z_2 = Z[j, i - 1, :]
      
      if i == 1:
         lev_inter = lambda x: leverage[0, 0]
      else:
         lev_inter = interp1d(x_grid[i - 1, :], leverage[i - 1, :], kind='linear', fill_value='extrapolate')
      
      ls = ls - .5 * dt * (sig_0 * np.exp(y) * lev_inter(np.exp(ls)))**2  + sig_0 * np.exp(y) * lev_inter(np.exp(ls)) * np.sqrt(dt) * (np.sqrt(1 - rho_bar**2) * z_1 + rho_bar * z_2)
      y = y * np.exp(-kappa * dt) + gamma * np.sqrt((1. - np.exp(-2. * kappa * dt)) / (2. * kappa)) * z_2
    
    m += np.maximum(np.exp(ls) - K, 0.) 

  return m / N_2

# from statistics import NormalDist
# N = NormalDist().cdf
# n = NormalDist().pdf

N = stats.norm.cdf
n = stats.norm.pdf

def PriceBS(S, t, K, r, T, sig, flag='C'):

    d_1  = (np.log(S/K) + (r+(sig**2/2))*np.sqrt(T-t))  / (sig * np.sqrt(T-t))
    d_2 = d_1 - sig * np.sqrt(T-t)
    C = S * N(d_1) - K * np.exp(-r*(T-t)) * N(d_2)
    if flag =='C':
        return C
    if flag=='P':
        return  C - S + K * np.exp(-r*(T-t))
    if flag not in ['C', 'P']:
        raise ValueError('Enter C or P')  


# Greeks :

class Greeks:
    
    def __init__(self, S, t, K, r, T, sig):
        self.S = S
        self.t = t
        self.K = K
        self.r = r
        self.T = T
        self.sig = sig
        return None

    def __d_1(self):
        
        d_1  = (np.log(self.S / self.K) + (self.r+(self.sig**2/2))*np.sqrt(self.T-self.t))  / (self.sig * np.sqrt(self.T-self.t))
        return d_1
    def __d_2(self):
        
        d_2 = Greeks.__d_1(self) - self.sig * np.sqrt(self.T-self.t)
        return d_2
    

    def Delta(self):
        
        return N(Greeks.__d_1(self))

    def Gamma(self):
        
        return n(Greeks.__d_1(self)) / (self.S * self.sig * np.sqrt(self.T-self.t))

    def Theta(self):
        
        return (self.S * self.sig * n(Greeks.__d_1(self)))/ (2 * np.sqrt(self.T - self.t)) + self.r * self.K * np.exp(-self.r * (self.T - self.t)) * N(Greeks.__d_2(self))

    def Vega(self):
        
        return self.S * np.sqrt(self.T - self.t) *    n(Greeks.__d_1(self))
    
    def Rho(self):
        
        return self.K * (self.T - self.t) *  np.exp(-self.r * (self.T - self.t)) * N(Greeks.__d_2(self))

    def dCdK(self):
        
        return - np.exp(-self.r * (self.T - self.t)) * N(Greeks.__d_2(self))    


def implidVolatility(S, t, K, r, T, sig_0, 
                     C_market, maxIter = 50, 
                     tolerance = 1e-5, 
                     method = 'N-R', 
                     flag = 'C', 
                     a = .0001, b = 2.0):
    
    if method == 'N-R':

        sig = sig_0
        C = PriceBS(S, t, K, r, T, sig, flag)

        stopping_criterion = np.abs(C - C_market)
        iter = 0

        while((stopping_criterion > tolerance) & (iter < maxIter)):
            iter += 1
            Vega = Greeks(S, t, K, r, T, sig).Vega()
            
            if Vega == float(0):
                message = 'Vega equals ', 0 , 'at iteration :', iter, '. I Suggest another method. Sigma will be put to 0.'
                sig = 0.
                break
            else :
                message = 'Algorithm Converged in : ', iter, ' iterations' 

            sig = sig - (C - C_market)  / Vega
            
            C = PriceBS(S, t, K, r, T, sig, flag)
            stopping_criterion = np.abs(C - C_market)
        
        print(message)
        return sig    

    if method == 'Dichotomy':
        C_min = PriceBS(S, t, K, r, T, a, flag)
        C_max = PriceBS(S, t,K, r, T, b, flag)

        
        try:
            assert((C_min <= C_market) & (C_market <= C_max))

        except AssertionError:
            eps = 0.1
            a = np.maximum(a - eps, 0.001)
            b = np.minimum(b + eps, 3.0)
            

        sig_min = a
        sig_max = b

        sig = (sig_min + sig_max)  / 2
        C = PriceBS(S, t, K, r, T, sig, flag)
        stopping_criterion = np.abs(C - C_market)
        iter = 0

        while((stopping_criterion > tolerance) & (iter < maxIter)):
            iter += 1

            if C - C_market > 0 :
                sig_max = sig
                sig = (sig_min + sig_max) / 2
            else :
                sig_min = sig
                sig = (sig_min + sig_max) / 2
            C = PriceBS(S, t, K, r, T, sig, flag)
            stopping_criterion = np.abs(C - C_market)

        print('Algorithm Converged in : ', iter, ' iterations.')
        return sig    


imp_vol = []


for i in range(8):
  imp_vol.append(implidVolatility(s_0, 0., strikes[i], 0., 1., .1, get_price(strikes[i]), tolerance = 1e-11, method = 'Dichotomy'))


imp_vol


gammas = [0., .25, .50, .75]


T, s_0, sig_0, y_0, rho, _, kappa = 1., 100., .15, 0., 0., .5, 1.
sig_loc = .15

n_steps = 100
N_1 = 10_000
dt = 1. / n_steps
n_xgrid = 30
rho_bar = rho * np.sqrt(2. * (1. - np.exp(-kappa * dt)) / (kappa * dt * (1. + np.exp(-kappa * dt))))


results = {}

for gamma in gammas:
  leverage = np.zeros((n_steps, n_xgrid))
  leverage[0, :] = (sig_loc / (sig_0 * np.exp(y_0))) * np.ones(n_xgrid)

  y = y_0 * np.ones(N_1)
  ls = np.log(s_0) * np.ones(N_1)

  x_grid = np.zeros((n_steps, n_xgrid))
  x_grid[0, :] = s_0 * np.ones(n_xgrid)
  
  for i in range(1, n_steps):
    z = rng(N_1)
    z_1, z_2 = z[:, 0], z[:, 1]

    if i == 1:
      lev_inter = lambda x: leverage[0, 0]
    else:
      lev_inter = interp1d(x_grid[i - 1, :], leverage[i - 1, :], kind = 'linear', fill_value='extrapolate')

    ls = ls - .5 * dt * (sig_0 * np.exp(y) * lev_inter(np.exp(ls)))**2  + sig_0 * np.exp(y) * lev_inter(np.exp(ls)) * np.sqrt(dt) * (np.sqrt(1 - rho_bar**2) * z_1 + rho_bar * z_2)
    s = np.exp(ls)
    y = y * np.exp(-kappa * dt) + gamma * np.sqrt((1. - np.exp(-2. * kappa * dt)) / (2. * kappa)) * z_2

    a = sig_0 * np.exp(y)
    bw = kappa * sig_loc * s_0 * np.sqrt(np.maximum(i * dt, .15)) / (N_1**(.2))
    x_min, x_max = q_ln(.05, i), q_ln(.95, i)
    x_grid[i, :] = np.linspace(x_min, x_max, n_xgrid)
    leverage[i, :] = sig_loc / np.array([np.sqrt(kern_reg(x, s, a**2, bw)) for x in x_grid[i, :]])
  
  results[str(gamma)] = {'x_grid': x_grid, 'leverage': leverage}



import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.linewidth'] = 2 
plt.rcParams.update({"text.usetex": True})
S = np.linspace(70., 150., 500)


colors = ['firebrick', 'navy', 'purple', 'k']
for color, key in zip(colors, results.keys()):
  x, y = results[key]['x_grid'][-1, :], results[key]['leverage'][-1, :]
  f_0 = interp1d(x, y, kind = 'linear', fill_value='extrapolate')
  plt.plot(S, f_0(S), '--', color = color, label = r'$\gamma=' + str(key) + '$')

plt.xlabel(r'$S$')
plt.ylabel(r'$l(T, S)$')
plt.legend()


Ks = np.linspace(80., 150., 150)
Ks = np.append(Ks, 100.)
Ks = np.sort(Ks)

idx, = np.where(Ks == 100.)[0]
Ks[idx - 1], Ks[idx + 1]

N_2 = 10_000
Z = np.random.randn(N_2, n_steps - 1, 2)


gammas
import time
from functools import wraps


from numba import jit
def _timer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Eslaped time :  {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@_timer
@jit(nopython = True)
def get_price1(K, gamma):
  m = 0.
  for j in range(N_2):
    y = y_0
    ls = np.log(s_0)
    #z = rng(n_steps - 1)
    for i in range(1, n_steps):
      #z = rng()
      z_1, z_2 = Z[j, i - 1, :]
      
      ls = ls - .5 * dt * (sig_0 * np.exp(y) )**2  + sig_0 * np.exp(y) * np.sqrt(dt) * (np.sqrt(1 - rho_bar**2) * z_1 + rho_bar * z_2)
      y = y * np.exp(-kappa * dt) + gamma * np.sqrt((1. - np.exp(-2. * kappa * dt)) / (2. * kappa)) * z_2
    
    m += np.maximum(np.exp(ls) - K, 0.) 

  return m / N_2



get_price1(100., .75)



smile = {}
for gamma in gammas:
  smile[str(gamma)] = np.array([implidVolatility(s_0, 0., k, 0., 1., .1, get_price1(k, gamma), tolerance=1e-10, method = 'Dichotomy') for k in Ks])



colors = ['firebrick', 'navy', 'purple', 'darkgreen']
for color, key in zip(colors, smile.keys()):
  plt.plot(np.log(Ks / s_0), smile[key], '--', color = color, label = r'$\gamma=' + str(key) + '$')

plt.xlabel(r'$x = \log(K / S_0)$', fontdict={'fontsize': 15}, color = 'darkblue')
plt.ylabel(r'$\sigma_{\mathrm{BS}}(t = T, x)$', fontdict={'fontsize': 15}, color = 'darkblue')
plt.legend()
plt.grid(linestyle = 'dotted')
plt.tight_layout()
#plt.savefig('iv_smile.pdf', format = 'pdf')
plt.show()

x = np.log(Ks / s_0)
# ATM skew (gamma = .75)
(smile['0.75'][idx + 1] - smile['0.75'][idx - 1]) / (2. *(x[idx] - x[idx - 1]))
