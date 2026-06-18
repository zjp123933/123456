# ==========================================================
# 文件名：fea_basic.py
# 功能：完全复用2.3作业所有有限元基础子程序
# 包含：模型读取、LM对号矩阵、单元刚度、总体刚度组装、后处理
# 作业衔接要求：本模块为2.3作业原有代码，本次作业仅替换求解模块
# 依赖文件：01_env_config.py
# 下标规则：外部JSON模型为1-based，代码内部统一转为0-based
# ==========================================================

# 导入全局环境与依赖库
from env_config import *

def read_model(filename):
    """
    【子程序】前处理模块：读取有限元JSON模型文件
    输入参数：
        filename (str)：JSON模型文件路径
    返回值：
        model (dict)：解析后的模型字典，包含节点、单元、材料、边界、载荷
    核心操作：
        1. 读取并解析JSON文件
        2. 工程1-based编号 → Python 0-based编号转换
        3. 自动计算所有单元几何长度
    """
    # 以utf-8编码打开JSON文件，读取模型数据
    with open(filename, 'r', encoding='utf-8') as f:
        model = json.load(f)

    # ========== 编号转换：工程习惯1-based → 代码0-based ==========
    # IEN：单元-节点拓扑数组，每个单元关联的节点编号
    model['IEN'] = [[n-1 for n in elem] for elem in model['IEN']]
    # fixed_dof：约束自由度编号（位移边界条件）
    model['fixed_dof'] = [d-1 for d in model['fixed_dof']]
    # force_dof：施加载荷的自由度编号（力边界条件）
    model['force_dof'] = [d-1 for d in model['force_dof']]

    # ========== 计算每个单元的几何长度 ==========
    model['L'] = []
    # 遍历每一个单元
    for elem in model['IEN']:
        n1, n2 = elem  # 获取单元两个节点的全局编号
        # 欧几里得距离公式：计算单元长度
        L = np.sqrt((model['x'][n2]-model['x'][n1])**2 + (model['y'][n2]-model['y'][n1])**2)
        model['L'].append(L)
    return model


def generate_LM(model):
    """
    【子程序】生成对号矩阵LM（Location Matrix）
    功能：建立「单元局部自由度」与「结构全局自由度」的映射关系
    输入参数：
        model (dict)：有限元模型字典
    返回值：
        LM (np.array)：对号矩阵，维度(局部自由度数, 单元数)
    作用：总体刚度矩阵组装的核心映射工具
    """
    # 提取模型基础参数
    nnp = model['nnp']      # nnp：总节点数
    ndof = model['ndof']    # ndof：单个节点的自由度数
    nel = model['nel']      # nel：总单元数
    nen = model['nen']      # nen：单个单元的节点数（杆/桁架固定为2）
    IEN = model['IEN']      # 单元拓扑数组
    ndof_local = ndof * nen # 单个单元的总局部自由度数
    # 初始化LM矩阵：整型矩阵，初始值为0
    LM = np.zeros((ndof_local, nel), dtype=int)

    # 遍历所有单元，逐列填充LM矩阵
    for e in range(nel):
        node_i, node_j = IEN[e][0], IEN[e][1]  # 当前单元的两个全局节点编号
        # 遍历单元所有局部自由度
        for local_dof in range(ndof_local):
            # 前ndof个局部自由度：归属第一个节点
            if local_dof < ndof:
                global_node = node_i
                local_at_node = local_dof
            # 后ndof个局部自由度：归属第二个节点
            else:
                global_node = node_j
                local_at_node = local_dof - ndof
            # 计算全局自由度编号：节点号 × 单节点自由度 + 节点内局部自由度
            global_dof = global_node * ndof + local_at_node
            LM[local_dof, e] = global_dof
    return LM


def element_stiffness_1d(model, e):
    """
    【子程序】一维杆单元刚度矩阵计算
    输入参数：
        model (dict)：模型字典
        e (int)：当前单元编号（0-based）
    返回值：
        Ke (np.array)：2×2 一维杆单元局部刚度矩阵
    理论公式：Ke = (EA/L) * [[1, -1], [-1, 1]]
    """
    E = model['E'][e]       # 单元弹性模量
    A = model['CArea'][e]   # 单元横截面积
    L = model['L'][e]       # 单元长度
    c = E * A / L           # 刚度系数 EA/L
    # 构造一维杆单元刚度矩阵
    Ke = c * np.array([[1, -1], [-1, 1]], dtype=float)
    return Ke


def element_stiffness_2d_truss(model, e):
    """
    【子程序】二维平面桁架单元刚度矩阵（全局坐标系）
    输入参数：
        model (dict)：模型字典
        e (int)：当前单元编号（0-based）
    返回值：
        Ke (np.array)：4×4 桁架单元刚度矩阵
        c (float)：方向余弦 cosθ
        s (float)：方向余弦 sinθ
        L (float)：单元长度
    理论：引入坐标变换，将局部刚度转换为全局坐标系刚度
    """
    E = model['E'][e]
    A = model['CArea'][e]
    # 获取单元两个节点坐标
    n1, n2 = model['IEN'][e]
    x1, y1 = model['x'][n1], model['y'][n1]
    x2, y2 = model['x'][n2], model['y'][n2]
    # 单元x/y方向投影
    dx = x2 - x1
    dy = y2 - y1
    # 单元长度
    L = np.sqrt(dx**2 + dy**2)
    # 方向余弦
    c = dx / L
    s = dy / L
    k = E * A / L
    # 预计算方向余弦乘积，简化矩阵
    cc, ss, cs = c*c, s*s, c*s

    # 二维桁架全局刚度矩阵
    Ke = k * np.array([
        [cc, cs, -cc, -cs],
        [cs, ss, -cs, -ss],
        [-cc, -cs, cc, cs],
        [-cs, -ss, cs, ss]
    ], dtype=float)
    return Ke, c, s, L


def assemble_global_K(model, LM):
    """
    【子程序】单元刚度矩阵 → 总体刚度矩阵 组装
    输入参数：
        model (dict)：模型字典
        LM (np.array)：对号矩阵
    返回值：
        K (np.array)：整体结构稠密总体刚度矩阵
    核心逻辑：根据LM映射，将单元刚度累加至总刚对应位置
    """
    # 提取基础参数
    ndof = model['ndof']
    nnp = model['nnp']
    nel = model['nel']
    nsd = model['nsd']      # nsd：空间维数 1=一维 2=二维
    ndof_total = nnp * ndof # 结构总自由度数
    # 初始化总体刚度矩阵：全零稠密矩阵
    K = np.zeros((ndof_total, ndof_total), dtype=float)

    # 遍历所有单元，逐个组装
    for e in range(nel):
        # 根据空间维数选择对应单元刚度计算函数
        if nsd == 1:
            Ke = element_stiffness_1d(model, e)
        else:
            Ke, _, _, _ = element_stiffness_2d_truss(model, e)
        ndof_local = Ke.shape[0]
        # 双重循环：遍历单元刚度所有元素，累加到总刚
        for a in range(ndof_local):
            for b in range(ndof_local):
                # 通过LM获取全局自由度编号
                ga = LM[a, e]
                gb = LM[b, e]
                K[ga, gb] += Ke[a, b]
    return K


def postprocess(model, d, LM):
    """
    【子程序】后处理模块：计算单元应力、轴力
    输入参数：
        model (dict)：模型字典
        d (np.array)：全局位移向量
        LM (np.array)：对号矩阵
    返回值：
        results (list)：每个单元的计算结果字典（长度、应力、轴力等）
    作业要求：求解完成后必须调用本模块完成后处理
    """
    nel = model['nel']
    nsd = model['nsd']
    results = []  # 存储所有单元结果
    # 遍历所有单元
    for e in range(nel):
        ndof_local = model['ndof'] * model['nen']
        de = np.zeros(ndof_local)  # 单元局部位移向量
        # 从全局位移中提取单元局部位移
        for a in range(ndof_local):
            de[a] = d[LM[a, e]]
        # 单元材料与几何参数
        E = model['E'][e]
        A = model['CArea'][e]
        L = model['L'][e]

        # 一维杆单元后处理
        if nsd == 1:
            B = np.array([-1, 1]) / L  # 应变-位移矩阵
            sigma = E * (B @ de)       # 单元应力
            force = sigma * A          # 单元轴力
            res = {
                'element': e+1, 'length': L,
                'stress': sigma, 'axial_force': force
            }
        # 二维桁架单元后处理
        else:
            n1, n2 = model['IEN'][e]
            x1, y1 = model['x'][n1], model['y'][n1]
            x2, y2 = model['x'][n2], model['y'][n2]
            dx, dy = x2-x1, y2-y1
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx/L, dy/L
            B = np.array([-c, -s, c, s]) / L
            sigma = E * (B @ de)
            force = sigma * A
            res = {
                'element': e+1, 'length': L,
                'direction_cosine': (c, s),
                'stress': sigma, 'axial_force': force
            }
        results.append(res)
    return results

# 模块加载提示
print("【02_fea_basic】有限元基础模块加载完成")