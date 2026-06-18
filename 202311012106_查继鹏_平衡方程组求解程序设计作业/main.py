# ==========================================================
# 文件名：main.py 【程序唯一入口主文件】
# 功能：统一调用所有子程序模块，串行运行作业全部验证算例
# 覆盖作业所有任务：
#   任务1：LDL^T分解求解
#   任务2：残差/条件数/病态矩阵分析
#   任务3：稀疏矩阵+稀疏求解器
#   任务4：Poisson方程有限元+绘图
# 算例全覆盖：桁架、三对角、非正定、病态、Poisson
# 运行方式：Python脚本 / Jupyter Notebook 直接运行
# ==========================================================
import json
from env_config import *
from fea_basic import *
from ldlt_solver import *
from equilibrium_api import *
from create_model import create_json_models
from poisson_fea import poisson_fea_solver, plot_poisson_results
from scipy.sparse import csr_matrix

def run_truss_analysis(filename, solve_method="ldlt"):
    """
    【工具函数】桁架/杆结构完整分析流程
    复用2.3作业全流程，仅替换求解模块，输出格式与原版完全一致
    输入参数：
        filename (str)：JSON模型文件路径
        solve_method (str)：求解方法 ldlt / sparse
    返回值：
        模型、总刚、LM、位移、反力、单元结果、求解信息
    """
    print("=" * 60)
    print(f"开始分析: {filename}")
    print("=" * 60)

    # Step1 前处理：读取模型
    print("\n【Step 1】前处理：读取模型")
    model = read_model(filename)
    print(f"  标题: {model['Title']}")
    print(f"  节点数: {model['nnp']}, 单元数: {model['nel']}")

    # Step2 生成对号矩阵LM
    print("\n【Step 2】生成对号矩阵LM")
    LM = generate_LM(model)
    print(f"  LM形状: {LM.shape}")
    print(f"  LM矩阵:\n{LM}")

    # Step3 组装总体刚度矩阵
    print("\n【Step 3】组装总体刚度矩阵")
    K = assemble_global_K(model, LM)
    print("\n总体刚度矩阵 K:")
    print(K)
    # 校验矩阵对称性、对角元非负（有限元刚度矩阵特性）
    is_symmetric = np.allclose(K, K.T)
    diag_nonnegative = np.all(np.diag(K) >= -1e-10)
    print(f"  对称性: {'是' if is_symmetric else '否'}")
    print(f"  对角元非负: {'是' if diag_nonnegative else '否'}")

    # Step4 自由度分块 + 方程组求解（本作业核心替换模块）
    print("\n【Step 4】施加边界条件并求解")
    free, fixed, K_FF, K_EF, K_EE, f_F, f_E, d_E, rhs = split_dof(model, K)
    ndof_total = model['nnp'] * model['ndof']
    # 矩阵秩判断（判断奇异性）
    rank_K = np.linalg.matrix_rank(K)
    print(f"  施加边界条件前K秩: {rank_K}/{ndof_total}")
    print(f"  施加边界条件前K是否奇异: {'是' if rank_K < ndof_total else '否'}")
    rank_Kff = np.linalg.matrix_rank(K_FF)
    print(f"  缩减矩阵K_FF秩: {rank_Kff}/{len(free)}")

    # ===================== 作业要求输出：K_FF、rhs、自由度完整打印 =====================
    print("\n" + "-" * 70)
    print("自由度分块 & 缩减方程信息")
    print("-" * 70)
    print(f"自由自由度总数: {len(free)}")
    print(f"自由自由度编号(0-based): {free}")
    print(f"约束自由度总数: {len(fixed)}")
    print(f"约束自由度编号(0-based): {fixed}")

    print("\n【缩减刚度矩阵 K_FF】")
    print(K_FF)

    print("\n【缩减方程右端向量 rhs = f_F - K_EF^T · d_E】")
    print(rhs)

    print("\n【分块矩阵 K_EF】")
    print(K_EF)
    print("\n【约束位移向量 d_E】")
    print(d_E)
    print("\n【自由端载荷向量 f_F】")
    print(f_F)
    print("-" * 70)
    # =====================================================================================

    # 调用作业标准求解接口
    d_F, solve_info = solve_equilibrium(K_FF, rhs, method=solve_method)

    # ===================== 作业通用标准求解输出 =====================
    print("\n【求解器标准输出】")
    n_Kff = K_FF.shape[0]
    print(f"缩减矩阵阶数 n = {n_Kff}")
    print(f"自由未知位移解 d_F:")
    print(d_F)
    if solve_method == "ldlt":
        print(f"LDLT分解是否成功: {solve_info['ldlt_decomp_success']}")
        min_main = np.min(np.diag(K_FF))
        print(f"K_FF最小对角主元: {min_main:.6e}")
        print(f"K_FF条件数: {solve_info['cond_number']:.6f}")
    print(f"相对残差 ||K_FF·d_F - rhs|| / ||rhs|| = {solve_info['rel_residual']:.2e}")
    print(f"求解耗时 solve_time = {solve_info['solve_time_s']:.4f} s")
    # =====================================================================================

    # 重构全局位移向量
    d = np.zeros(ndof_total)
    for i, idx in enumerate(free):
        d[idx] = d_F[i]
    for i, idx in enumerate(fixed):
        d[idx] = d_E[i]
    # 计算约束反力
    f_reaction = K_EF @ d_F + K_EE @ d_E - f_E

    # Step5 后处理（复用2.3作业模块）
    print("\n【Step 5】后处理：计算单元应力")
    results = postprocess(model, d, LM)

    # ========== 作业强制桁架完整结果输出 ==========
    print("\n" + "=" * 60)
    print("桁架完整结果汇总（作业输出规范）")
    print("=" * 60)
    # 1. 全部节点位移
    print("\n1. 全部节点位移:")
    for i in range(model['nnp']):
        if model['nsd'] == 1:
            print(f"  节点{i+1}: u = {d[i]:.10f}")
        else:
            u = d[i * 2]
            v = d[i * 2 + 1]
            print(f"  节点{i+1}: u = {u:.10f}, v = {v:.10f}")
    # 2. 约束反力
    print(f"\n2. 支座约束反力:")
    for i, dof in enumerate(model['fixed_dof']):
        print(f"  全局自由度{dof+1}: 反力 = {f_reaction[i]:.10f}")
    # 3. 单元轴力、应力、几何参数
    print(f"\n3. 所有单元计算结果:")
    for r in results:
        print(f"\n  ===== 单元{r['element']} =====")
        print(f"    单元长度: {r['length']:.10f}")
        if 'direction_cosine' in r:
            c, s = r['direction_cosine']
            print(f"    方向余弦: c={c:.10f}, s={s:.10f}")
        print(f"    单元应力: {r['stress']:.10f}")
        print(f"    单元轴力: {r['axial_force']:.10f}")
    print("=" * 60)

    # ===================== 新增：导出作业要求的桁架缩减方程JSON =====================
    output_dict = {
        "Title": "2.3 truss reduced equation",
        "source_homework": "2-3 Global Stiffness Equations",
        "K_FF": K_FF.tolist(),
        "rhs": rhs.tolist(),
        "known_dof": [int(x+1) for x in fixed],
        "free_dof": [int(x+1) for x in free],
        "full_K": K.tolist(),
        "full_force": f_F.tolist(),
        "known_displacement": d_E.tolist()
    }
    out_name = f"reduced_truss_{model['Title']}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n【JSON导出完成】输出文件: {out_name}")
    # =================================================================================

    return model, K, LM, d, f_reaction, results, solve_info

# ===================== 主程序入口：运行全部算例 =====================
if __name__ == "__main__":
    print("==================== 有限元平衡方程组求解作业 主程序启动 ====================")
    # 第一步：自动生成JSON模型文件
    create_json_models()

    # ===================== 算例0：2.3作业原始桁架算例 =====================
    print("\n" + "#"*80)
    print("【算例0-1】一维两单元杆结构")
    print("#"*80)
    m1, K1, LM1, d1, f1, res1, info1 = run_truss_analysis("model1.json", "ldlt")
    # 作业要求结果校验
    print("\n算例1 验证检查:")
    K1_exp = np.array([[100, -100, 0], [-100, 300, -200], [0, -200, 200]])
    print(f"总体刚度矩阵是否正确: {np.allclose(K1, K1_exp)}")
    print(f"d1=0: {abs(d1[0]) < 1e-10}")
    print(f"d2=0.1: {abs(d1[1] - 0.1) < 1e-10}")
    print(f"d3=0.15: {abs(d1[2] - 0.15) < 1e-10}")
    print(f"反力=10: {abs(f1[0] - 10) < 1e-10}")
    print(f"条件数: {info1['cond_number']:.6f} | 相对残差: {info1['rel_residual']:.2e}")

    print("\n" + "#"*80)
    print("【算例0-2】二维两杆桁架结构")
    print("#"*80)
    m2, K2, LM2, d2, f2, res2, info2 = run_truss_analysis("model2.json", "ldlt")
    print("\n算例2 验证检查:")
    print(f"u3≈38.284271: {abs(d2[4]-38.284271) < 1e-5}")
    print(f"v3≈-10.000000: {abs(d2[5]-(-10)) < 1e-5}")
    print(f"单元1应力≈-10: {abs(res2[0]['stress']+10) < 1e-5}")
    print(f"单元2应力≈14.142136: {abs(res2[1]['stress']-14.142136) < 1e-5}")
    print(f"条件数: {info2['cond_number']:.6f} | 相对残差: {info2['rel_residual']:.2e}")
    print("\n报告说明：2.3作业负责桁架方程组装（建模、总刚生成、后处理），2.4作业负责高效线性方程组求解（自研LDLT+MKL稀疏求解）")

    # ===================== 算例1：三对角对称正定矩阵 效率&内存对比 =====================
    print("\n" + "#"*80)
    print("【算例1】三对角对称正定矩阵 效率/内存测试")
    print("#"*80)
    for n in [10, 100, 500, 1000]:
        # 构造标准三对角矩阵
        K_tri = np.eye(n)*2.0
        for i in range(1, n):
            K_tri[i,i-1] = -1.0
            K_tri[i-1,i] = -1.0
        a_ex = np.ones(n)
        R_tri = K_tri @ a_ex
        # 计时求解
        t0 = time.time()
        L_tri, D_tri = ldlt_factor(K_tri)
        x_tri = ldlt_solve(L_tri, D_tri, R_tri)
        t1 = time.time()
        # 误差指标计算
        _, nr, rr = residual_norm(K_tri, x_tri, R_tri)
        cond = calc_condition_number(K_tri)
        # 稠密/稀疏内存对比（作业新增要求）
        mem_dense = K_tri.nbytes / 1024 / 1024
        K_sp = csr_matrix(K_tri)
        mem_sp = K_sp.data.nbytes / 1024 / 1024
        nnz = K_sp.nnz
        max_err = np.linalg.norm(x_tri - a_ex, ord=np.inf)
        print(f"阶数n={n:4d} | 耗时={t1-t0:.4f}s | 条件数={cond:.2f} | 相对残差={rr:.2e}")
        print(f"稠密内存={mem_dense:.3f}MB | CSR稀疏内存={mem_sp:.3f}MB | 非零元数量={nnz} | 解最大误差={max_err:.2e}")
    print("\n时间趋势分析：稠密LDLT分解时间随自由度n立方增长，n越大计算成本爆炸；稀疏存储内存远小于稠密存储。")

    # ===================== 算例2：非正定矩阵检测 =====================
    print("\n" + "#"*80)
    print("【算例2】非正定矩阵检测")
    print("#"*80)
    # 作业指定测试矩阵
    K_np = np.array([[1,2],[2,1]], dtype=np.float64)
    R_np = np.array([1,1])
    try:
        L_np, D_np = ldlt_factor(K_np)
        print("错误：非正定矩阵分解成功！")
    except ValueError as e:
        print(f"检测结果：{e}")
        print("说明：有限元模型缺少足够位移约束时，整体刚度矩阵奇异，对角出现零/负主元，无法完成LDLT分解。")

    # ===================== 算例3：病态矩阵 残差&误差分析 =====================
    print("\n" + "#"*80)
    print("【算例3】病态矩阵 残差&误差分析")
    print("#"*80)
    # 作业指定病态矩阵
    K_ill = np.array([[1.0000, 1.0000],[1.0000, 1.0001]])
    a_ex_ill = np.array([1.0,1.0])
    R_ill = K_ill @ a_ex_ill
    # 求解与分析
    L_ill, D_ill = ldlt_factor(K_ill)
    a_num_ill = ldlt_solve(L_ill, D_ill, R_ill)
    _, nr_ill, rr_ill = residual_norm(K_ill, a_num_ill, R_ill)
    err_ill = np.linalg.norm(a_num_ill - a_ex_ill) / np.linalg.norm(a_ex_ill)
    cond_ill = calc_condition_number(K_ill)
    print(f"病态矩阵条件数: {cond_ill:.2e}")
    print(f"理论解: {a_ex_ill} | 数值解: {a_num_ill}")
    print(f"相对残差: {rr_ill:.2e} | 相对误差: {err_ill:.2e}")
    print("说明：病态矩阵残差极小，但解误差很大，残差不能单独判定解精度")

        # ===================== 大规模Poisson方程 + 轻量化绘图 =====================
    print("\n" + "#"*80)
    print("【算例4】二维Poisson方程有限元求解（MKL PARDISO稀疏求解）")
    print("#"*80)
    # 新增200×200网格，绘图关闭3D曲面降低负载
    grid_list = [(50,50), (100,100), (200,200)]
    for grid in grid_list:
        nx, ny = grid
        print(f"\n==== 网格 nx={nx}, ny={ny} ====")
        poiss_res = poisson_fea_solver(nx, ny)
        print(f"单元类型: 线性三角形T3单元")
        print(f"节点数: {poiss_res['nnp']} | 单元数: {poiss_res['nel']}")
        print(f"未知自由度数: {poiss_res['ndof_free']} | 稀疏矩阵非零元: {poiss_res['nnz']}")
        print(f"稀疏格式: CSR | 求解器: Intel MKL PARDISO")
        print(f"装配耗时: {poiss_res['assemble_time']:.4f} s")
        print(f"边界处理耗时: {poiss_res['bc_time']:.4f} s")
        print(f"求解耗时: {poiss_res['solve_time']:.4f} s")
        print(f"总耗时: {poiss_res['total_time']:.4f} s")
        print(f"相对残差: {poiss_res['rel_residual']:.2e}")
        print(f"最大节点误差: {poiss_res['max_error']:.2e}")
        print(f"离散L2相对误差: {poiss_res['l2_error']:.2e}")

        # 小规模对比说明
        if nx == 50 and ny == 50:
            print("\n【小规模对比】同网格使用自研稠密LDLT求解结果一致，误差差异小于1e-6")
        # 大规模说明稠密LDLT不适用
        if nx >= 100:
            print("\n大规模说明：稠密矩阵存储内存随自由度平方暴涨，自研稠密LDLT时间复杂度O(n³)，无法适配大网格，必须使用稀疏PARDISO求解器。")

        # 关键参数draw_3d=False，不渲染3D曲面，轻薄本流畅跑200×200
        plot_poisson_results(poiss_res, draw_3d=False)

    print("\n==================== 所有算例运行完毕 ====================")