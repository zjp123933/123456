# ==========================================================
# 文件名：ldlt_solver.py
# 功能：作业任务1核心 — 手写对称正定矩阵LDL^T分解求解器
# 包含：分解、前代、对角求解、回代、非正主元检测、残差、条件数计算
# 作业强制要求：禁止调用numpy/scipy高阶求解器，核心算法纯手工实现
# 理论：K = L * D * L^T ，L为单位下三角，D为对角阵
# 依赖文件：01_env_config.py
# ==========================================================

# 导入全局环境
from env_config import *

def ldlt_factor(K):
    """
    【子程序】对称矩阵 LDL^T 分解
    输入参数：
        K (np.array)：对称方阵（有限元缩减刚度矩阵K_FF）
    返回值：
        L (np.array)：单位下三角矩阵
        D (np.array)：对角向量（存储对角矩阵对角元）
    异常处理：
        检测 D[j] <= 1e-12 判定为非正主元，抛出异常（作业要求）
    适用范围：仅针对对称正定矩阵（有限元缩减刚度矩阵特性）
    """
    n = K.shape[0]          # 矩阵阶数
    L = np.eye(n, dtype=np.float64)  # 初始化单位下三角矩阵
    D = np.zeros(n, dtype=np.float64)# 初始化对角向量

    # 逐列计算D对角元与L下三角元素
    for j in range(n):
        # 第一步：计算D[j]，累加前面已计算项
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - sum_d

        # ========== 非正主元检测（作业核心要求） ==========
        # 主元接近0或负数：矩阵奇异/非正定，无法使用LDL^T求解
        if D[j] <= 1e-12:
            raise ValueError(f"【分解失败】第 {j+1} 个主元 D={D[j]:.6e}，矩阵非正定或存在零主元！")

        # 第二步：计算L矩阵第j列下方的下三角元素
        for i in range(j+1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_l) / D[j]
    return L, D


def ldlt_solve(L, D, R):
    """
    【子程序】基于LDL^T分解求解线性方程组 L*D*L^T * x = R
    求解三步法（作业要求完整实现）：
        1. 前代：求解 L * y = R
        2. 对角求解：求解 D * z = y
        3. 回代：求解 L^T * x = z
    输入参数：
        L (np.array)：单位下三角阵（ldlt_factor输出）
        D (np.array)：对角向量（ldlt_factor输出）
        R (np.array)：方程组右端向量
    返回值：
        x (np.array)：方程组解向量
    """
    n = len(R)
    # ===================== 1. 前代计算 L*y = R =====================
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = R[i] - s

    # ===================== 2. 对角求解 D*z = y =====================
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        z[i] = y[i] / D[i]

    # ===================== 3. 回代计算 L^T*x = z =====================
    x = np.zeros(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        s = 0.0
        for k in range(i+1, n):
            s += L[k, i] * x[k]
        x[i] = z[i]
    return x


def residual_norm(K, x, R):
    """
    【子程序】残差计算（作业任务2误差分析核心）
    公式：残差 r = R - K·x
    计算：残差2-范数、相对残差 = ||r|| / ||R||
    输入参数：
        K (np.array)：系数矩阵
        x (np.array)：数值解
        R (np.array)：右端向量
    返回值：
        r (np.array)：残差向量
        norm_r (float)：残差二范数
        rel_res (float)：相对残差
    作用：评估求解精度、分析病态矩阵特性
    """
    r = R - K @ x
    norm_r = np.linalg.norm(r, 2)   # 残差2-范数
    norm_R = np.linalg.norm(R, 2)   # 右端向量2-范数
    rel_res = norm_r / (norm_R + 1e-15)  # 加极小值防止除零错误
    return r, norm_r, rel_res


def calc_condition_number(K):
    """
    【子程序】计算矩阵2-范数条件数 cond(K)
    公式：cond(K) = ||K||₂ · ||K⁻¹||₂
    作用：判断矩阵病态程度（作业任务2病态分析必备）
    输入参数：
        K (np.array)：系数矩阵
    返回值：
        float：矩阵条件数
    """
    return np.linalg.cond(K, 2)

# 模块加载提示
print("【03_ldlt_solver】LDL^T求解器 & 误差分析工具加载完成")