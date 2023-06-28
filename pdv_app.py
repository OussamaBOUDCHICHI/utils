import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABCMeta, abstractmethod

plt.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.linewidth'] = 2 
set_tex = lambda tex=True: plt.rcParams.update({"text.usetex": tex})
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 
import streamlit as st

st.markdown('# Path-Dependent Local Volatility Models ')
st.markdown('**Author : BOUDCHICHI Oussama**')

st.markdown(' ')

## Code 
np.random.seed(0)
rng = lambda n=1: np.random.randn(n)
dt = 1. / 252.

@st.cache_data
def get_samples(x_init, mdl_params, matur, n_samples = 10_000):
    betas, lambds = mdl_params['betas'], mdl_params['lambdas']
    sigma = lambda r_1, r_2: betas[0] + betas[1] * r_1 + betas[2] * np.sqrt(r_2)
    disc_steps = int(matur / dt)
    spot = np.empty((n_samples, disc_steps + 1))
    spot[:, 0] = x_init[0]
    r_1, r_2 = np.full(n_samples, x_init[1]), np.full(n_samples, x_init[2])

    for n in range(1, disc_steps + 1):
        z = rng(n_samples)
        sig_ = sigma(r_1, r_2)
        spot[:, n] = spot[:, n - 1] * (1. + sig_ * np.sqrt(dt) * z)
        r_1 = r_1 * (1. - lambds[0] * dt) + lambds[0] * sig_ * np.sqrt(dt) * z
        r_2 = r_2 + lambds[1] * (sig_**2 - r_2) * dt
    return spot

@st.cache_data
def price_(samples, strikes, forward, disc_factor):
    return disc_factor * (np.maximum(np.subtract.outer(strikes, samples), .0).mean(axis = 1) + forward - strikes)
#% ----------------------


models = ['twofPDLV', 'fourfPDLV', 'fourfPDSV']
option = st.sidebar.selectbox('Choose a Model', models)
st.sidebar.markdown("---")

st.sidebar.markdown('$$\mathrm{d}S_t = S_t \sigma(R_{1,t}, R_{2,t})a_t\mathrm{d}W_t$$')
st.sidebar.markdown(r"$$\sigma: (R_1, R_2) \mapsto \beta_0 + \beta_1 R_1 + \beta_2 \sqrt{R_2}$$")
st.sidebar.markdown("---")

st.sidebar.markdown(r"$$ R_{1,t} := \displaystyle \int_{-\infty}^t K_1(t - s) \sigma_s \mathrm{d}W_s$$")
st.sidebar.markdown(r"$$ R_{2,t} := \displaystyle \int_{-\infty}^t K_2(t - s) \sigma_s^2 \mathrm{d}s$$")

st.sidebar.markdown("---")
st.markdown("## Model's parameters")


s_0, r_1, r_2 = None, None, None
betas, lambdas, thetas = [], [], []
kappa, nu = None, None

col1, col2, col3, col4 = st.columns(4)
with col1:
        s_0 = st.number_input(r"$\color{darkmagenta}{S_0}$", min_value=0., value = 1.)
        betas.append(st.number_input(r'$\color{blue}{\beta_0}$', min_value=0., value = .013)) 
        betas.append(st.number_input(r'$\color{blue}{\beta_1}$', max_value=0., value = -.013))
        betas.append(st.number_input(r'$\color{blue}{\beta_2}$', min_value=0., max_value=1., value = .9)) 

with col2:
    r_1 = st.number_input(r"$\color{darkmagenta}{R_{1,0}}$")

with col3:
    r_2 = st.number_input(r"$\color{darkmagenta}{R_{2,0}}$")
match option:
    case "twofPDLV":
        st.sidebar.markdown(r"$$K_i : \tau \mapsto \lambda_i e^{-\lambda_i \tau} \quad \forall\ i \in \{1, 2\}$$")
        st.sidebar.markdown(r"$$\forall \ t\in [0, T]: \ a_t = 1$$")
        
        with col2:
            lambdas.append(st.number_input(r'$$\lambda_1$$', min_value=0.)) 
            lambdas.append(st.number_input(r'$$\lambda_2$$', min_value=0.))  

    case "fourfPDLV":
        st.sidebar.markdown(r"$$ K_i : \tau \mapsto (1 - \theta_i)\lambda_{i,0} e^{-\lambda_{i,0}\tau} + \theta_i\lambda_{i,1} e^{-\lambda_{i,1}\tau}\quad \forall\ i \in \{1, 2\}$$")
        st.sidebar.markdown(r"$$\forall \ t\in [0, T]: \ a_t = 1$$")

        with col2:
            thetas.append(st.number_input(r'$$\theta_1$$', min_value=0., max_value = 1.)) 
            lambdas.append(st.number_input(r'$$\lambda_{1, 0}$$', min_value=0.))  
            lambdas.append(st.number_input(r'$$\lambda_{1, 1}$$', min_value=0.))  
        
        with col3:
            thetas.append(st.number_input(r'$$\theta_2$$', min_value=0., max_value = 1.)) 
            lambdas.append(st.number_input(r'$$\lambda_{2, 0}$$', min_value=0.))  
            lambdas.append(st.number_input(r'$$\lambda_{2, 1}$$', min_value=0.))

    case "fourfPDSV":
        st.sidebar.markdown(r"$$ K_i : \tau \mapsto (1 - \theta_i)\lambda_{i,0} e^{-\lambda_{i,0}\tau} + \theta_i\lambda_{i,1} e^{-\lambda_{i,1}\tau}\quad \forall\ i \in \{1, 2\}$$")
        st.sidebar.markdown(r"$$\forall \ t\in [0, T]: \ a_t = e^{\nu X_t}$$")
        st.sidebar.markdown(r"$$\mathrm{d}X_t = -\kappa X_t \mathrm{d}t + \mathrm{d}W_t$$")
        
        with col2:
            thetas.append(st.number_input(r'$$\theta_1$$', min_value=0., max_value = 1.)) 
            lambdas.append(st.number_input(r'$$\lambda_{1, 0}$$', min_value=0.))  
            lambdas.append(st.number_input(r'$$\lambda_{1, 1}$$', min_value=0.))  
        
        with col3:
            thetas.append(st.number_input(r'$$\theta_2$$', min_value=0., max_value = 1.)) 
            lambdas.append(st.number_input(r'$$\lambda_{2, 0}$$', min_value=0.))  
            lambdas.append(st.number_input(r'$$\lambda_{2, 1}$$', min_value=0.))

        with col4:
            kappa = st.number_input(r'$$\kappa$$', min_value=0.) 
            nu = st.number_input(r'$$\nu$$', min_value=0.)  



st.markdown('---')
form = st.form("input_data")
number_maturities = None
strikes, maturities = [], []

col1_, col2_ = st.columns(2)
# with col1_:
#     number_strikes = st.number_input("Number of strikes", min_value=0)

# with col2_:
#     number_maturities = st.number_input("Number of maturities", min_value=0)

with col1_:
    #number_maturities = st.number_input("Number of maturities", min_value=0, value = 1)
    maturities_manual = st.checkbox("Enter maturities manually")
    maturities_grid = st.checkbox("Generate grid of maturities", value = True)
with col2_:
    strikes_manual = st.checkbox("Enter strikes manually")
    strikes_grid = st.checkbox("Generate grid of strikes", value = True)
    same_strikes = st.checkbox("Same strikes for all maturities", value = True)

if (maturities_manual & maturities_grid) or (strikes_grid & strikes_manual):
    st.write("**Please choose one method of input**")

if maturities_manual and not maturities_grid:
    with col1_:
        number_maturities = st.number_input("Number of maturities", min_value=0, value = 1)
        for k in range(number_maturities):
            maturities.append(st.number_input(f'$T_{k}$')) 

        maturities = np.array(maturities)


if maturities_grid:
    with col1_:
        t_min = st.number_input(r"$ T_{\text{min}}$", min_value=0.)
        t_max = st.number_input(r"$ T_{\text{max}}$", min_value=1.)
        t_step = st.number_input(r"$\Delta T$", min_value=0., value = (t_max - t_min))
        maturities = np.arange(t_min, t_max + t_step, t_step)


#number_maturities = mat_grid.shape[0]
K_grid = None

if same_strikes & strikes_grid:
    with col2_:
        K_min = st.number_input(r"$ K_{\text{min}}$", min_value=0.)
        K_max = st.number_input(r"$ K_{\text{max}}$", min_value=1.)
        K_step = st.number_input(r"$\Delta K$", min_value=0., value = (K_max - K_min))
        K_grid = np.arange(K_min, K_max + K_step, K_step)

mdl_params = {'betas': betas, 'lambdas': lambdas}
x_init = np.array([s_0, r_1, r_2])

spot_samples, prices = {}, {}
for mat in maturities:
    spot_samples[str(mat)] = get_samples(x_init, mdl_params, matur=mat)


max_matur = maturities.max()
t_grid = np.arange(0., max_matur + dt, dt)

#spot_samples[str(maturities[0])]

# spot_ = get_samples(x_init, mdl_params, matur=1.)
# fig, axs = plt.subplots()
# axs.plot(t_grid, spot_samples[str(maturities[0])][1, :])
# st.pyplot(fig = fig)

# maturities

ouss = spot_samples[str(maturities[0])][:, -1]
prices = price_(ouss, K_grid, s_0, 1.)

fig, axs = plt.subplots()
axs.plot(K_grid, prices)
st.pyplot(fig = fig)