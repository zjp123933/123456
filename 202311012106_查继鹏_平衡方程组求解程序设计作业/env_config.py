# ==========================================================
# 文件名：env_config.py
# 功能：全局环境初始化、第三方依赖库导入、打印/绘图格式统一配置
# 说明：本文件为所有模块的公共依赖，所有.py文件均优先引用此文件
# 适配环境：原生Python脚本 / Jupyter Notebook 双环境
# 数组下标约定：代码内部统一使用 0-based（Python标准索引）
# ==========================================================

# 数值计算核心库：矩阵、数组、数学运算
import numpy as np
# JSON文件读写：用于有限元模型输入输出（作业指定模型格式）
import json
# 系统时间模块：统计装配、求解耗时，完成效率分析
import time
# 操作系统模块：文件路径操作
import os
# 稀疏矩阵相关：COO/CSR/CSC格式，作业任务3稀疏存储要求
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
# 稀疏方程组求解：替代稠密求解器，用于大规模算例
from scipy.sparse.linalg import spsolve
# 绘图库：绘制云图、误差图、3D曲面（作业Poisson算例绘图要求）
import matplotlib.pyplot as plt
# 3D绘图工具：配合matplotlib实现三维曲面展示
from mpl_toolkits.mplot3d import Axes3D

# ===================== 全局格式配置 =====================
# 设置numpy矩阵打印格式：保留4位小数、关闭科学计数法
# 与2.3作业原始代码输出格式保持完全一致
np.set_printoptions(precision=4, suppress=True)

# 兼容Jupyter Notebook内嵌绘图：Notebook环境自动开启行内绘图
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
# 原生Python脚本环境跳过该配置
except NameError:
    # 设置中文字体，防止绘图中文乱码
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
    # 解决坐标轴负号显示异常问题
    plt.rcParams["axes.unicode_minus"] = False

# 模块加载提示
print("【01_env_config】全局环境 & 依赖库加载完成")