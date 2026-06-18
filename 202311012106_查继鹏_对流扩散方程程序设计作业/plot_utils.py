import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

# 仅保留Windows自带中文字体，删除Linux专属文泉驿
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 普通坐标轴负号兼容
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'SimHei'

# 屏蔽两类警告
# 1. 中文汉字Glyph缺失警告
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from font")
# 2. 数学负号\u2212缺失警告（本次红色刷屏来源）
warnings.filterwarnings("ignore", category=UserWarning, message="Font 'default' does not have a glyph for '\\u2212'")
# 3. 屏蔽matplotlib字体查找日志（找不到文泉驿的提示）
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
# ============================================================================

def calc_max_error(theta_num, theta_exact):
    """计算节点最大绝对误差"""
    return np.max(np.abs(theta_num - theta_exact))

def plot_solution_comparison(x, sol_gal, sol_upwind, sol_supg, sol_exact, Pe_val):
    """
    绘制单张图：精确解、标准Galerkin、迎风格式、SUPG
    输入同一Pe下四组解，图标题标注Pe数值
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, sol_exact, 'k-', linewidth=2, label='精确解 Exact')
    plt.plot(x, sol_gal, 'r--', linewidth=1.5, label='标准Galerkin α=0')
    plt.plot(x, sol_upwind, 'g-.', linewidth=1.5, label='迎风格式 α=1')
    plt.plot(x, sol_supg, 'b:', linewidth=2, label='SUPG α=α_opt')
    plt.xlabel('x')
    plt.ylabel(r'$\theta(x)$')
    plt.title(f'对流扩散方程数值解对比，单元Peclet数 Pe = {Pe_val:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 保存高清图片到本地，方便报告使用
    plt.savefig(f"Pe_{Pe_val}.png", dpi=300, bbox_inches="tight")
    plt.show()

def print_error_table(pe_list, err_gal, err_upwind, err_supg):
    """打印误差表格，输出每种格式最大误差"""
    print("="*60)
    print(f"{'Pe':<8}{'标准Galerkin最大误差':<22}{'迎风格式最大误差':<20}{'SUPG最大误差':<18}")
    print("-"*60)
    for i, pe in enumerate(pe_list):
        print(f"{pe:<8.2f}{err_gal[i]:<22.6e}{err_upwind[i]:<20.6e}{err_supg[i]:<18.6e}")
    print("="*60)

def plot_convergence_curve(nel_list, err_gal_list, err_supg_list, Pe_target):
    plt.figure(figsize=(8,5))
    plt.loglog(nel_list, err_gal_list, 'ro-', label="Galerkin最大误差")
    plt.loglog(nel_list, err_supg_list, 'bs-', label="SUPG最大误差")
    plt.xlabel("单元数量 nel")
    plt.ylabel("Max 绝对误差")
    plt.title(f"网格加密收敛曲线 Pe={Pe_target:.2f}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    # 保存收敛曲线图片
    plt.savefig(f"convergence_Pe_{Pe_target}.png", dpi=300, bbox_inches="tight")
    plt.show()