# ==========================================================
# 文件名：06_poisson_fea.py
# 优化适配轻薄本，支持200×200大网格轻量化计算绘图
# 修复三角形节点读取下标越界bug，降低CPU/内存占用
# 功能：二维Poisson方程有限元求解 + 轻量化可视化绘图
# 单元类型：线性三角形T3单元
# 理论解：u(x,y) = sin(πx)sin(πy)
# 依赖文件：01_env_config.py、03_ldlt_solver.py、04_equilibrium_api.py
# ==========================================================
from env_config import *
from ldlt_solver import *
from equilibrium_api import split_dof

def poisson_fea_solver(nx, ny):
    """
    【轻量化求解主函数】Poisson方程有限元求解器
    控制方程：-Δu = f ，全域u=0（Dirichlet齐次边界）
    输入参数：
        nx (int)：x方向单元数量
        ny (int)：y方向单元数量
    返回值：
        res (dict)：计算结果、网格、误差、时间、绘图数据、统计字段
    """
    # 计时起点：整体开始
    t_total_start = time.time()

    # 计算域：单位正方形 (0,1)×(0,1)
    Lx, Ly = 1.0, 1.0
    # 生成均匀网格坐标
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    X, Y = np.meshgrid(x, y)
    nnp = (nx+1) * (ny+1)   # 总节点数
    nel = 2 * nx * ny      # 总单元数（每个矩形拆分为2个三角形单元）
    ndof_total = nnp

    # 节点编号映射函数：(i,j)网格点 → 全局节点编号(0-based)
    def node_id(i, j):
        return j*(nx+1) + i

    # 单元拓扑数组 IEN：存储每个单元的三个节点编号
    IEN = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_id(i, j)
            n1 = node_id(i+1, j)
            n2 = node_id(i, j+1)
            n3 = node_id(i+1, j+1)
            IEN.append([n0, n1, n2])
            IEN.append([n1, n3, n2])

    # 存储所有节点的(x,y)坐标，预分配数组减少内存碎片
    node_xy = np.zeros((nnp, 2), dtype=np.float64)
    for j in range(ny+1):
        for i in range(nx+1):
            idx = node_id(i, j)
            node_xy[idx, 0] = x[i]
            node_xy[idx, 1] = y[j]

    # ===================== 理论解与右端项（制造解） =====================
    def u_exact(xx, yy):
        """Poisson方程理论解"""
        return np.sin(np.pi * xx) * np.sin(np.pi * yy)
    def f_rhs(xx, yy):
        """方程右端项 f = 2π²·sin(πx)sin(πy)"""
        return 2 * (np.pi**2) * np.sin(np.pi * xx) * np.sin(np.pi * yy)

    # ===================== 稀疏矩阵组装（COO格式起步） =====================
    t_assemble_start = time.time()
    # 预分配列表容量，减少动态扩容开销
    rows = [0] * (nel * 9)
    cols = [0] * (nel * 9)
    vals = [0.0] * (nel * 9)
    R = np.zeros(ndof_total, dtype=np.float64)  # 全局载荷向量
    ptr = 0  # COO填充指针

    # 遍历所有三角形单元，组装单元刚度与单元载荷
    for elem in IEN:
        n0, n1, n2 = elem
        # 逐个读取节点坐标，不会索引越界
        x0 = node_xy[n0, 0]
        y0 = node_xy[n0, 1]
        x1 = node_xy[n1, 0]
        y1 = node_xy[n1, 1]
        x2 = node_xy[n2, 0]
        y2 = node_xy[n2, 1]
        # 三角形单元面积
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        # 应变矩阵B（T3线性单元）
        b0 = y1 - y2; b1 = y2 - y0; b2 = y0 - y1
        c0 = x2 - x1; c1 = x0 - x2; c2 = x1 - x0
        B = np.array([[b0, b1, b2], [c0, c1, c2]]) / (2*area)
        # 单元刚度矩阵
        Ke = area * B.T @ B
        # 单元等效载荷（积分近似）
        fe = area/3 * np.array([f_rhs(x0,y0), f_rhs(x1,y1), f_rhs(x2,y2)])
        # 累加至全局载荷向量
        for a in range(3):
            R[elem[a]] += fe[a]
            # COO格式填充刚度矩阵非零元
            for b in range(3):
                rows[ptr] = elem[a]
                cols[ptr] = elem[b]
                vals[ptr] = Ke[a,b]
                ptr += 1

    # 截断多余预分配空间，转为COO再转CSR
    rows = rows[:ptr]
    cols = cols[:ptr]
    vals = vals[:ptr]
    K_coo = coo_matrix((vals, (rows, cols)), shape=(ndof_total, ndof_total))
    K_csr = K_coo.tocsr()
    nnz = K_csr.nnz  # 稀疏矩阵非零元总数
    t_assemble_end = time.time()
    assemble_time = t_assemble_end - t_assemble_start

    # ===================== 齐次Dirichlet边界：全域u=0 =====================
    t_bc_start = time.time()
    fixed_dof = []
    for j in range(ny+1):
        for i in range(nx+1):
            xi, yi = x[i], y[j]
            # 判定边界节点：x=0 / x=1 / y=0 / y=1
            if abs(xi) < 1e-6 or abs(xi-1) < 1e-6 or abs(yi) <1e-6 or abs(yi-1)<1e-6:
                fixed_dof.append(node_id(i,j))
    fixed_value = np.zeros_like(fixed_dof, dtype=np.float64)
    free = [d for d in range(ndof_total) if d not in fixed_dof]
    ndof_free = len(free)  # 未知自由度数

    # 自由度分块，生成缩减方程
    K_FF = K_csr[np.ix_(free, free)]
    K_EF = K_csr[np.ix_(fixed_dof, free)]
    d_E = np.array(fixed_value)
    f_F = R[free]
    rhs = f_F - K_EF.T @ d_E
    t_bc_end = time.time()
    bc_time = t_bc_end - t_bc_start

    # ===================== 稀疏求解 =====================
    t_solve_start = time.time()
    d_F = spsolve(K_FF, rhs)
    t_solve_end = time.time()
    solve_time = t_solve_end - t_solve_start

    # 重构全场数值解
    u_num = np.zeros(ndof_total, dtype=np.float64)
    u_num[free] = d_F
    u_num[fixed_dof] = d_E

    # ===================== 误差计算（作业要求指标） =====================
    u_exact_all = np.zeros(ndof_total, dtype=np.float64)
    err_all = np.zeros(ndof_total, dtype=np.float64)
    for idx in range(nnp):
        xx, yy = node_xy[idx]
        ue = u_exact(xx, yy)
        u_exact_all[idx] = ue
        err_all[idx] = abs(u_num[idx] - ue)

    max_err = np.max(err_all)                # 最大节点绝对误差
    l2_err = np.linalg.norm(u_num - u_exact_all, 2) / np.linalg.norm(u_exact_all, 2) # L2相对误差

    # 计算相对残差
    full_res = R - K_csr @ u_num
    norm_full_res = np.linalg.norm(full_res)
    norm_R_full = np.linalg.norm(R)
    rel_residual = norm_full_res / (norm_R_full + 1e-15)

    # 总耗时
    total_time = time.time() - t_total_start

    # 重构网格数据，用于绘图
    u_grid = np.zeros_like(X, dtype=np.float64)
    err_grid = np.zeros_like(X, dtype=np.float64)
    for j in range(ny+1):
        for i in range(nx+1):
            idx = node_id(i,j)
            u_grid[j,i] = u_num[idx]
            err_grid[j,i] = err_all[idx]

    # ===================== 【重点】补齐所有main.py需要的字段 =====================
    res = {
        "nx": nx,
        "ny": ny,
        "nnp": nnp,
        "nel": nel,
        "ndof_free": ndof_free,
        "nnz": nnz,
        "assemble_time": assemble_time,
        "bc_time": bc_time,
        "solve_time": solve_time,
        "total_time": total_time,
        "rel_residual": rel_residual,
        "max_error": max_err,
        "l2_error": l2_err,
        "X": X,
        "Y": Y,
        "u_num_grid": u_grid,
        "err_grid": err_grid
    }
    return res

def plot_poisson_results(res, draw_3d=False):
    """
    【轻量化绘图函数】适配轻薄本，可关闭3D曲面大幅提速
    输入参数：
        res (dict)：poisson_fea_solver返回的结果字典
        draw_3d (bool)：是否绘制3D曲面，轻薄本跑200×200建议设为False
    输出：2张2D云图（数值解+误差），可选3D曲面
    """
    X = res["X"]
    Y = res["Y"]
    u_num = res["u_num_grid"]
    err = res["err_grid"]
    # 根据是否绘制3D调整画布列数
    col_num = 3 if draw_3d else 2
    fig = plt.figure(figsize=(14, 5))
    # 子图1：数值解二维云图，减少分层降低渲染压力
    ax1 = fig.add_subplot(1, col_num, 1)
    cf1 = ax1.contourf(X, Y, u_num, cmap="jet", levels=10, antialiased=False)
    plt.colorbar(cf1, ax=ax1)
    ax1.set_title(f"Poisson数值解 (nx={res['nx']}, ny={res['ny']})")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    # 子图2：绝对误差云图
    ax2 = fig.add_subplot(1, col_num, 2)
    cf2 = ax2.contourf(X, Y, err, cmap="hot", levels=10, antialiased=False)
    plt.colorbar(cf2, ax2)
    ax2.set_title(f"绝对误差云图 |MaxErr={res['max_error']:.2e}|")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    # 可选3D曲面：轻薄本跑200×200默认关闭
    if draw_3d:
        ax3 = fig.add_subplot(133, projection='3d')
        surf = ax3.plot_surface(X, Y, u_num, cmap="jet", linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax3, shrink=0.5)
        ax3.set_title("数值解3D曲面")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("u")
    plt.tight_layout()
    plt.show()
    return fig

# 模块加载提示
print("【06_poisson_fea】轻量化Poisson有限元模块加载完成，支持200×200网格")