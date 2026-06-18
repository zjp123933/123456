# ==========================================================
# 文件名：equilibrium_api.py
# 功能：自由度分块 + 作业标准求解接口
# 求解器：自研LDL^T + Intel MKL PARDISO（通过PyPardiso调用，作业指定求解器）
# 作业接口：solve_equilibrium(K_FF, rhs, method="ldlt", **options)
# 依赖：pypardiso、numpy、scipy
# ==========================================================
from env_config import *
from ldlt_solver import *
# 导入PyPardiso（封装Intel MKL PARDISO）
from pypardiso import spsolve as pardiso_solve

def split_dof(model, K):
    """
    【子程序】自由度分块模块，修复force_dof字符串索引报错
    功能：划分自由/约束自由度，生成缩减方程 K_FF · d_F = rhs
    输入：有限元模型、总体刚度矩阵
    输出：free, fixed, K_FF, K_EF, K_EE, f_F, f_E, d_E, rhs
    """
    ndof_total = model['nnp'] * model['ndof']
    fixed = model['fixed_dof']
    free = [i for i in range(ndof_total) if i not in fixed]

    # 提取刚度分块矩阵
    K_FF = K[np.ix_(free, free)]
    K_EF = K[np.ix_(fixed, free)]
    K_EE = K[np.ix_(fixed, fixed)]

    # 初始化载荷向量，强制浮点类型
    f = np.zeros(ndof_total, dtype=np.float64)
    # 循环时强制把dof、val转为数字，解决字符串下标报错
    for dof_text, val_text in zip(model['force_dof'], model['force_value']):
        dof = int(dof_text)
        val = float(val_text)
        f[dof] = val

    f_F = f[free]
    f_E = f[fixed]
    d_E = np.array(model['fixed_value'], dtype=np.float64)

    # 计算缩减方程右端项
    rhs = f_F - K_EF.T @ d_E
    return free, fixed, K_FF, K_EF, K_EE, f_F, f_E, d_E, rhs

def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    """
    【作业标准求解接口】
    method = "ldlt"  : 自研稠密LDL^T求解（桁架、小矩阵）
    method = "sparse": Intel MKL PARDISO（大规模稀疏矩阵，Poisson算例）
    返回：位移解 + 求解信息（耗时、残差、求解器名称等）
    """
    solve_info = {}
    t_start = time.perf_counter()

    # 分支1：自研 LDL^T 稠密求解（原有功能不变）
    if method == "ldlt":
        try:
            L, D = ldlt_factor(K_FF)
            solve_info["ldlt_decomp_success"] = True
        except ValueError as e:
            solve_info["ldlt_decomp_success"] = False
            solve_info["error_msg"] = str(e)
            raise e

        d_F = ldlt_solve(L, D, rhs)
        _, norm_r, rel_res = residual_norm(K_FF, d_F, rhs)
        cond = calc_condition_number(K_FF)

        solve_info["cond_number"] = cond
        solve_info["res_norm"] = norm_r
        solve_info["rel_residual"] = rel_res

    # 分支2：Intel MKL PARDISO 稀疏求解（作业要求）
    elif method == "sparse":
        # 转为PARDISO标准CSR稀疏格式
        K_sparse = csr_matrix(K_FF)
        # 调用MKL PARDISO求解器
        d_F = pardiso_solve(K_sparse, rhs)

        # 计算残差指标
        _, norm_r, rel_res = residual_norm(K_sparse, rhs)
        solve_info["res_norm"] = norm_r
        solve_info["rel_residual"] = rel_res
        solve_info["sparse_format"] = "CSR"
        solve_info["solver_name"] = "Intel MKL PARDISO"  # 报告/输出统一标注

    else:
        raise NotImplementedError(f"不支持求解方法: {method}")

    # 统计总耗时
    t_end = time.perf_counter()
    solve_info["solve_time_s"] = t_end - t_start
    return d_F, solve_info

print("【04_equilibrium_api】加载完成 | 稀疏求解器：Intel MKL PARDISO")