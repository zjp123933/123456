import numpy as np
from fem_core import alpha_supg, solve_advection_diffusion
from plot_utils import calc_max_error, plot_solution_comparison, print_error_table, plot_convergence_curve

def main():
    # 全局固定参数
    L = 1.0
    nel = 20
    v = 1.0
    Pe_cases = [0.1, 3.0]
    le = L / nel

    err_galerkin = []
    err_upwind = []
    err_supg = []
    all_results = {}

    print("========== 开始计算两种Pe工况 ==========")
    for Pe in Pe_cases:
        kappa = (v * le) / (2 * Pe)
        print(f"\n【Pe = {Pe:.2f}】 kappa = {kappa:.6e}")

        # 1. 标准Galerkin α=0
        x_g, theta_g, theta_ex, K_gal = solve_advection_diffusion(nel, L, v, kappa, alpha=0.0)
        err_g = calc_max_error(theta_g, theta_ex)

        # 2. 迎风格式 α=1
        x_u, theta_u, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=1.0)
        err_u = calc_max_error(theta_u, theta_ex)

        # 3. SUPG
        alpha_opt = alpha_supg(Pe)
        x_s, theta_s, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=alpha_opt)
        err_s = calc_max_error(theta_s, theta_ex)
        print(f"SUPG最优alpha = {alpha_opt:.6f}")

        all_results[Pe] = {
            "x": x_g,
            "exact": theta_ex,
            "galerkin": theta_g,
            "upwind": theta_u,
            "supg": theta_s,
            "K_galerkin": K_gal
        }
        err_galerkin.append(err_g)
        err_upwind.append(err_u)
        err_supg.append(err_s)

    # 输出误差总表
    print("\n========== 各格式最大节点误差汇总 ==========")
    print_error_table(Pe_cases, err_galerkin, err_upwind, err_supg)

    # 绘图：Pe=0.1、Pe=3.0
    print("\n绘制 Pe=0.1 结果图...")
    res01 = all_results[0.1]
    plot_solution_comparison(res01["x"], res01["galerkin"], res01["upwind"], res01["supg"], res01["exact"], 0.1)

    print("绘制 Pe=3.0 结果图...")
    res3 = all_results[3.0]
    plot_solution_comparison(res3["x"], res3["galerkin"], res3["upwind"], res3["supg"], res3["exact"], 3.0)

    # 任务4：Pe=3 标准Galerkin矩阵性质分析
    print("\n========== Pe=3.0 标准Galerkin全局刚度矩阵分析 ==========")
    K = all_results[3.0]["K_galerkin"]
    print(f"矩阵尺寸：{K.shape}")
    is_symmetric = np.allclose(K, K.T, atol=1e-10)
    print(f"矩阵是否对称：{is_symmetric}")

    eig_vals = np.linalg.eigvals(K)
    min_eig = np.min(np.real(eig_vals))
    is_pos_def = min_eig > -1e-10
    print(f"最小特征值实部：{min_eig:.4e}")
    print(f"矩阵是否正定：{is_pos_def}")
    print("\n矩阵左上角6×6子矩阵：")
    print(np.round(K[:6, :6], 4))

    # 附加题：网格加密收敛测试 nel=10,20,40,80
    print("\n========== 附加题：网格加密收敛测试 ==========")
    nel_list = [10, 20, 40, 80]
    Pe_target = 3.0
    err_gal_conv = []
    err_supg_conv = []
    for n in nel_list:
        le_curr = L / n
        kap = (v * le_curr) / (2 * Pe_target)
        _, thg, thex, _ = solve_advection_diffusion(n, L, v, kap, alpha=0)
        eg = calc_max_error(thg, thex)
        ao = alpha_supg(Pe_target)
        _, ths, _, _ = solve_advection_diffusion(n, L, v, kap, alpha=ao)
        es = calc_max_error(ths, thex)
        err_gal_conv.append(eg)
        err_supg_conv.append(es)
        print(f"nel={n:3d} | Galerkin误差={eg:.6e} | SUPG误差={es:.6e}")
    plot_convergence_curve(nel_list, err_gal_conv, err_supg_conv, Pe_target)

if __name__ == "__main__":
    main()