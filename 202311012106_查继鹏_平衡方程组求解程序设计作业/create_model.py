# ==========================================================
# 文件名：create_model.py
# 功能：自动生成2.3作业标准JSON模型文件
# 生成文件：model1.json（一维杆）、model2.json（二维桁架）
# 作业要求：使用JSON作为模型输入格式
# 依赖文件：01_env_config.py
# ==========================================================

# 导入全局环境
from env_config import *

def create_json_models():
    """
    【主函数】生成两个标准算例的JSON模型文件
    算例1：一维两单元杆结构（作业验证算例0-1）
    算例2：二维两杆桁架结构（作业验证算例0-2）
    JSON格式说明：所有编号采用工程1-based格式
    """
    # ===================== 算例1：一维两单元杆 =====================
    model1_data = {
        "Title": "1D bar example",       # 模型标题
        "nsd": 1,                        # 空间维数：1=一维
        "ndof": 1,                       # 单节点自由度数：1个轴向位移
        "nnp": 3,                        # 总节点数：3
        "nel": 2,                        # 总单元数：2
        "nen": 2,                        # 单单元节点数：2（杆单元）
        "E": [100.0, 200.0],             # 两个单元的弹性模量
        "CArea": [1.0, 1.0],             # 两个单元的横截面积
        "x": [0.0, 1.0, 2.0],            # 节点x坐标（一维y全为0）
        "y": [0.0, 0.0, 0.0],
        "IEN": [[1, 2], [2, 3]],         # 单元拓扑：单元1(1-2)、单元2(2-3)
        "fixed_dof": [1],                # 约束自由度：节点1位移固定
        "fixed_value": [0.0],            # 约束位移值：0
        "force_dof": [2, 3],             # 施加载荷的自由度
        "force_value": [0.0, 10.0]       # 载荷大小：节点3受10单位拉力
    }
    # 写入JSON文件，indent=2 格式化缩进，便于阅读
    with open('model1.json', 'w', encoding='utf-8') as f:
        json.dump(model1_data, f, indent=2)
    print("【05_create_model】model1.json 创建成功")

    # ===================== 算例2：二维两杆桁架 =====================
    model2_data = {
        "Title": "2D truss example",     # 模型标题
        "nsd": 2,                        # 空间维数：2=二维
        "ndof": 2,                       # 单节点自由度：x/y两个位移
        "nnp": 3,                        # 总节点数：3
        "nel": 2,                        # 总单元数：2
        "nen": 2,                        # 单单元节点数：2
        "E": [1.0, 1.0],                 # 弹性模量
        "CArea": [1.0, 1.0],             # 横截面积
        "x": [1.0, 0.0, 1.0],            # 节点x坐标
        "y": [0.0, 0.0, 1.0],            # 节点y坐标
        "IEN": [[1, 3], [2, 3]],         # 单元拓扑
        "fixed_dof": [1, 2, 3, 4],       # 节点1、2完全固定
        "fixed_value": [0.0, 0.0, 0.0, 0.0],
        "force_dof": [5, 6],             # 节点3施加载荷
        "force_value": [10.0, 0.0]
    }
    with open('model2.json', 'w', encoding='utf-8') as f:
        json.dump(model2_data, f, indent=2)
    print("【05_create_model】model2.json 创建成功")

# 独立运行本文件时，自动生成模型
if __name__ == "__main__":
    create_json_models()