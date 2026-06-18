import numpy as np

def alpha_supg(Pe: float) -> float:
    if np.abs(Pe) < 1e-8:
        return 0.0
    coth_pe = 1 / np.tanh(Pe)
    alpha_opt = coth_pe - 1.0 / Pe
    return alpha_opt

def element_matrix(kappa: float, v: float, le: float, alpha: float):
    kappa_bar = kappa + alpha * v * le / 2.0
    K_diff = kappa_bar / le * np.array([[1, -1], [-1, 1]])
    K_conv = v / 2.0 * np.array([[-1, 1], [-1, 1]])
    Ke = K_diff + K_conv
    return Ke

def solve_advection_diffusion(nel: int, L: float, v: float, kappa: float, alpha: float):
    le = L / nel
    x = np.linspace(0, L, nel + 1)
    nnodes = nel + 1
    K_global = np.zeros((nnodes, nnodes))
    rhs = np.zeros(nnodes)

    for e in range(nel):
        i0 = e
        i1 = e + 1
        Ke = element_matrix(kappa, v, le, alpha)
        K_global[i0:i1+1, i0:i1+1] += Ke

    # 边界条件 θ(0)=0, θ(L)=1
    K_global[0, :] = 0.0
    K_global[0, 0] = 1.0
    rhs[0] = 0.0

    K_global[-1, :] = 0.0
    K_global[-1, -1] = 1.0
    rhs[-1] = 1.0

    theta_num = np.linalg.solve(K_global, rhs)
    Pe_global = v * L / kappa
    theta_exact = (np.expm1(v * x / kappa)) / np.expm1(Pe_global)
    return x, theta_num, theta_exact, K_global