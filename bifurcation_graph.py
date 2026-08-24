import math
import numpy as np
import matplotlib.pyplot as plt

def A_hat(m: int, rho: float) -> float:
    """Fourier cosine coefficient hatA(m) for the band kernel at bandwidth rho."""
    if m == 0:
        return 1.0
    x = 2 * math.pi * m * rho
    if abs(x) < 1e-12:
        return 1.0  # numerical safeguard (limit as x->0)
    return math.sin(x) / x

'''
def A_hat(m: int, rho: float, p: float) -> float:
    """Fourier cosine coefficient hatA(m) for the band kernel at bandwidth rho."""
    if m == 0:
        return -p
    x = math.pi * m * rho
    if abs(x) < 1e-12:
        return -p  # numerical safeguard (limit as x->0)
    return (1 - 2*p)*math.sin(2*x) / x - 2*r - p + 4*r*p
'''

# ======== EDIT YOUR PARAMETERS HERE ========
r = 0.1        # logistic growth rate
eta_n = 0.05
n0 = 8.0
eta_c = 0.04
k = 7
alpha = 0.2
beta = 0.1
rho_list = np.linspace(0.01, 0.47, 1000)     # bandwidth of the band kernel
p_list = np.linspace(0.01, 1, 1000)     # bandwidth of the band kernel

w_star_list = np.zeros(len(rho_list))
w_star_w_extreme_list = np.zeros(len(rho_list))
kappa_list = np.zeros(len(rho_list))
gamma_list = np.zeros(len(rho_list))

w_star_list = np.zeros(len(p_list))
w_star_w_extreme_list = np.zeros(len(p_list))
kappa_list = np.zeros(len(p_list))
gamma_list = np.zeros(len(p_list))
# ==========================================

c0 = alpha*n0/beta

k_discrete = np.arange(1, 1000) 

for i in range(len(rho_list)):
    rho = rho_list[i]

    # Logistic derivatives at n0
    # f(n) = r*n(1 - n/n0)  
    fp   = -r          # f'(S)
    fpp  = -2*r/n0      # f''(S)
    fppp = 0.0  # f'''(S)

    k_star = 1
    # Fourier coefficients
    A0  = A_hat(0, rho)
    Ak  = A_hat(k_star, rho)
    A2k = A_hat(2*k_star, rho)

    # Useful combinations
    Delta  = A0 - Ak
    Delta2 = A0 - A2k
    C2     = A0 - 2*Ak + A2k
    Delta_extreme = A0 
    w_star_min = (eta_n*eta_c*(Delta**2) + (beta*eta_n - 2*eta_c*fp)*Delta - 2*beta*fp)/(alpha*eta_n*Delta*n0)

    for k in k_discrete:
        # Fourier coefficients
        A0  = A_hat(0, rho)
        Ak  = A_hat(k, rho)
        A2k = A_hat(2*k, rho)

        # Useful combinations
        Delta  = A0 - Ak
        Delta2 = A0 - A2k
        C2     = A0 - 2*Ak + A2k
        Delta_extreme = A0 
    
        # w
        num_w = eta_n*eta_c*(Delta**2) + (beta*eta_n - 2*eta_c*fp)*Delta - 2*beta*fp
        den_w = alpha*eta_n*Delta*n0
        if abs(den_w) < 1e-14:
            raise ZeroDivisionError("alpha*eta_n*Delta is too small; adjust rho or parameters.")
        w_star_temp = num_w / den_w
        if w_star_temp < w_star_min:
            w_star_min = w_star_temp
            k_star = k
        
    #print("\\rho = ", rho, ": ", "w_star_min = ", w_star_min, "k_star = ", k_star)

    num_w_extreme = eta_n*eta_c*(Delta_extreme**2) + (beta*eta_n - 2*eta_c*fp)*Delta_extreme - 2*beta*fp
    den_w_extreme = alpha*eta_n*Delta_extreme*n0
    if abs(den_w_extreme) < 1e-14:
        raise ZeroDivisionError("alpha*eta_n*Delta is too small; adjust rho or parameters.")
    w_star_w_extreme = num_w_extreme / den_w_extreme

    w_star_list[i] = w_star_min
    w_star_w_extreme_list[i] = w_star_w_extreme

    w = w_star_min

    # Fourier coefficients (again)
    A0  = A_hat(0, rho)
    Ak  = A_hat(k_star, rho)
    A2k = A_hat(2*k_star, rho)

    # Useful combinations (again)
    Delta  = A0 - Ak
    Delta2 = A0 - A2k
    C2     = A0 - 2*Ak + A2k
    Delta_extreme = A0 
    w_star_min = (eta_n*eta_c*(Delta**2) + (beta*eta_n - 2*eta_c*fp)*Delta - 2*beta*fp)/(alpha*eta_n*Delta*n0)

    # p and q
    q_n = 1.0
    q_c = alpha/(beta + eta_c*Delta)
    q = np.array([[q_n], [q_c]])

    p_n = (2*alpha*beta + 2*alpha*eta_c*Delta)/(alpha*(eta_n + 2*eta_c)*Delta + 2*alpha*(beta - fp))
    p_c = ((beta + eta_c*Delta)*(eta_n*Delta - 2*fp))/(alpha*(eta_n + 2*eta_c)*Delta + 2*alpha*(beta - fp))

    p = np.array([[p_n], [p_c]])

    # s_0, s_2, t_0 and t_2
    s_0n = -fpp*q_n**2/4
    s_0c = 0.0

    s_2n = -eta_n*w*q_n*q_c*Delta2/8 - fpp*q_n**2/4
    s_2c = 0.0

    s_0 = np.array([[s_0n], [s_0c]])
    s_2 = np.array([[s_2n], [s_2c]])

    L_0 = np.array([[fp, 0], [alpha, -beta]])
    L_2k = np.array([[-eta_n*Delta2/2 + fp, eta_n*w*n0*Delta2/2], [alpha, -eta_c*Delta2 - beta]])
    L_0_inv = np.linalg.inv(L_0)
    L_2k_inv = np.linalg.inv(L_2k)

    t_0 = L_0_inv @ s_0
    t_2 = L_2k_inv @ s_2

    t_0n = t_0[0][0]
    t_0c = t_0[1][0]
    t_2n = t_2[0][0]
    t_2c = t_2[1][0]

    # kappa
    kappa = (eta_n*n0/2.0) * p_n * q_c * (A0 - Ak)
    kappa_list[i] = kappa

    # gamma
    T1 = eta_n*w*p_n*q_c*t_0n*(A0 - Ak)/2
    T2 = eta_n*w*p_n*q_c*t_2n*(A0 + A2k - 2*Ak)/8
    T3 = eta_n*w*p_n*q_n*t_2c*(A0 - A2k)/8
    T4 = eta_n*w**3*n0*p_n*q_c**3*(4*Ak - 3*A0 - A2k)/32
    T5 = fpp * p_n * q_n * (t_0n + 0.5*t_2n)
    gamma = T1 + T2 + T3 + T4 + T5
    gamma_list[i] = gamma


plt.figure(figsize=(8, 6))
plt.plot(rho_list, w_star_list, label = "$\\chi^{*}$", color = "#FF5F05", linewidth = 5)
plt.plot(rho_list, w_star_w_extreme_list, label = "$\\chi^{**}$", color = "#13294B", linewidth = 5, linestyle='--')
plt.grid(True)
plt.grid(which = 'both')
plt.grid(alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', linewidth = 0.7) 
plt.ylim([0.38, 0.4936])
plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$\\chi^{*}$ and $\\chi^{**}$', fontsize=24)
plt.legend(fontsize=24)
plt.show()

plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='k', linestyle='--', linewidth = 5)
#plt.plot(rho_list, kappa_list, label = "$\\kappa$")
plt.plot(rho_list, kappa_list, label = "$\\kappa$", color = "#FF5F05", linewidth = 5)
plt.grid(True)
plt.grid(which = 'both')
plt.grid(alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', linewidth = 0.7) 
plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$\\kappa$', fontsize=24)
plt.ylim(-0.25, 0.25)
plt.legend(fontsize=24)
plt.show()


plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='k', linestyle='--', linewidth = 5)
#plt.plot(rho_list, kappa_list, label = "$\\kappa$")
plt.plot(rho_list, gamma_list, label = "$\\gamma$", color = "#FF5F05", linewidth = 5)
plt.grid(True)
plt.grid(which = 'both')
plt.grid(alpha = 0.5)
plt.grid(color = 'gray', linestyle = ':', linewidth = 0.7) 
plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$\\gamma$', fontsize=24)
plt.ylim(-0.02, 0.02)
plt.legend(fontsize=24)
plt.show()
