import math
import numpy as np
import matplotlib.pyplot as plt

def A_hat(m: int, rho: float) -> float:
    """Fourier cosine coefficient hatA(m) for the band kernel at bandwidth rho."""
    if m == 0:
        return 1.0
    x = 2 * math.pi * m * rho
    if abs(x) < 1e-12:
        return 1.0
    return math.sin(x) / x


# ======== PARAMETERS ========
r = 0.1
eta_n = 0.05
n0 = 8.0
eta_c = 0.04
alpha = 0.2
beta = 0.1

rho_list = np.linspace(0.01, 0.47, 1000)
k_discrete = np.arange(1, 1000)

#theta_values = [0.0, 0.5/n0, 1.4/n0, 3/n0]
theta_values = [0.0, 0.5/n0, 1.1/n0]
'''
theta_labels = [
    r'$\theta = 0$',
    r'$\theta = 0.5/\bar{n}$',
    r'$\theta = 1.0/\bar{n}$',
    r'$\theta = 1.1/\bar{n}$'
]
'''
theta_labels = [
    r'$\theta = 0$',
    r'$\theta = 0.5/\bar{n}$',
    r'$\theta = 1.1/\bar{n}$'
]
#colors = ["#13294B", "#FF5F05", "#00A5CF", "#C800CF"]
colors = ["#13294B", "#FF5F05", "#00A5CF"]
# ============================


c0 = alpha*n0/beta

w_star_list = np.zeros(len(rho_list))
w_star_w_extreme_list = np.zeros(len(rho_list))
kappa_list = np.zeros(len(rho_list))
gamma_all = []


for theta_index, theta in enumerate(theta_values):

    gamma_list = np.zeros(len(rho_list))

    for i, rho in enumerate(rho_list):

        # Reaction derivatives at n0
        fp   = -r
        fpp  = 2*r*(theta - 1/n0)
        fppp = 3*r*theta*(2/n0 - theta)

        # Initial guess for critical mode
        k_star = 1

        A0 = A_hat(0, rho)
        Ak = A_hat(k_star, rho)

        Delta = A0 - Ak

        w_star_min = (
            eta_n*eta_c*(Delta**2)
            + (beta*eta_n - 2*eta_c*fp)*Delta
            - 2*beta*fp
        ) / (alpha*eta_n*Delta*n0)

        # Search over modes
        for k in k_discrete:
            A0 = A_hat(0, rho)
            Ak = A_hat(k, rho)

            Delta = A0 - Ak

            num_w = (
                eta_n*eta_c*(Delta**2)
                + (beta*eta_n - 2*eta_c*fp)*Delta
                - 2*beta*fp
            )
            den_w = alpha*eta_n*Delta*n0

            if abs(den_w) < 1e-14:
                continue

            w_star_temp = num_w / den_w

            if w_star_temp < w_star_min:
                w_star_min = w_star_temp
                k_star = k

        # Extreme threshold w**
        Delta_extreme = A_hat(0, rho)

        num_w_extreme = (
            eta_n*eta_c*(Delta_extreme**2)
            + (beta*eta_n - 2*eta_c*fp)*Delta_extreme
            - 2*beta*fp
        )
        den_w_extreme = alpha*eta_n*Delta_extreme*n0

        w_star_w_extreme = num_w_extreme / den_w_extreme

        # w* and w** do not depend on theta, so store them only once
        if theta_index == 0:
            w_star_list[i] = w_star_min
            w_star_w_extreme_list[i] = w_star_w_extreme

        w = w_star_min

        # Fourier coefficients for selected k_star
        A0  = A_hat(0, rho)
        Ak  = A_hat(k_star, rho)
        A2k = A_hat(2*k_star, rho)

        Delta  = A0 - Ak
        Delta2 = A0 - A2k

        # p and q
        q_n = 1.0
        q_c = alpha / (beta + eta_c*Delta)

        p_n = (
            2*alpha*beta + 2*alpha*eta_c*Delta
        ) / (
            alpha*(eta_n + 2*eta_c)*Delta
            + 2*alpha*(beta - fp)
        )

        p_c = (
            (beta + eta_c*Delta)*(eta_n*Delta - 2*fp)
        ) / (
            alpha*(eta_n + 2*eta_c)*Delta
            + 2*alpha*(beta - fp)
        )

        # s_0, s_2
        s_0n = -fpp*q_n**2/4
        s_0c = 0.0

        s_2n = -eta_n*w*q_n*q_c*Delta2/8 - fpp*q_n**2/4
        s_2c = 0.0

        s_0 = np.array([[s_0n], [s_0c]])
        s_2 = np.array([[s_2n], [s_2c]])

        # L_0 and L_2k
        L_0 = np.array([
            [fp, 0],
            [alpha, -beta]
        ])

        L_2k = np.array([
            [-eta_n*Delta2/2 + fp, eta_n*w*n0*Delta2/2],
            [alpha, -eta_c*Delta2 - beta]
        ])

        t_0 = np.linalg.inv(L_0) @ s_0
        t_2 = np.linalg.inv(L_2k) @ s_2

        t_0n = t_0[0][0]
        t_0c = t_0[1][0]
        t_2n = t_2[0][0]
        t_2c = t_2[1][0]

        # kappa does not depend on theta, so store it only once
        kappa = (eta_n*n0/2.0) * p_n * q_c * Delta

        if theta_index == 0:
            kappa_list[i] = kappa

        # gamma
        T1 = eta_n*w*p_n*q_c*t_0n*(A0 - Ak)/2
        T2 = eta_n*w*p_n*q_c*t_2n*(A0 + A2k - 2*Ak)/8
        T3 = eta_n*w*p_n*q_n*t_2c*(A0 - A2k)/8
        T4 = eta_n*w**3*n0*p_n*q_c**3*(4*Ak - 3*A0 - A2k)/32
        T5 = fpp * p_n * q_n * (t_0n + 0.5*t_2n)

        gamma = T1 + T2 + T3 + T4 + T5
        gamma_list[i] = gamma

    gamma_all.append(gamma_list)


# ======== Figure 1: w* and w** ========

plt.figure(figsize=(8, 6))
plt.plot(
    rho_list,
    w_star_list,
    label="$w^{*}$",
    color="#FF5F05",
    linewidth=5
)
plt.plot(
    rho_list,
    w_star_w_extreme_list,
    label="$w^{**}$",
    color="#13294B",
    linewidth=5,
    linestyle='--'
)

plt.grid(True)
plt.grid(which='both')
plt.grid(alpha=0.5)
plt.grid(color='gray', linestyle=':', linewidth=0.7)

plt.ylim([0.38, 0.4936])
plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$w^{*}$ and $w^{**}$', fontsize=24)
plt.legend(fontsize=24)
plt.show()


# ======== Figure 2: kappa ========

plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='k', linestyle='--', linewidth=5)

plt.plot(
    rho_list,
    kappa_list,
    label=r'$\kappa$',
    color="#FF5F05",
    linewidth=5
)

plt.grid(True)
plt.grid(which='both')
plt.grid(alpha=0.5)
plt.grid(color='gray', linestyle=':', linewidth=0.7)

plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$\\kappa$', fontsize=24)
plt.ylim(-0.25, 0.25)
plt.legend(fontsize=24)
plt.show()


# ======== Figure 3: gamma for different theta ========

plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='k', linestyle='--', linewidth=5)

for gamma_list, label, color in zip(gamma_all, theta_labels, colors):
    plt.plot(
        rho_list,
        gamma_list,
        label=label,
        color=color,
        linewidth=5
    )

plt.grid(True)
plt.grid(which='both')
plt.grid(alpha=0.5)
plt.grid(color='gray', linestyle=':', linewidth=0.7)

plt.xlabel('Band width $\\rho$', fontsize=24)
plt.ylabel('$\\gamma$', fontsize=24)
#plt.ylim(-0.006, 0.006)
plt.ylim(-0.02, 0.02)
plt.legend(fontsize=18)
plt.show()