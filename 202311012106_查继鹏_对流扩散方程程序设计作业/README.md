# 一维对流扩散有限元作业
## 文件结构
- fem_core.py：有限元核心计算（单元矩阵、组装、求解、SUPG参数）
- plot_utils.py：绘图函数、误差计算工具，内置中文字体配置，消除Glyph警告
- main.py：主程序入口，一键运行全部任务1~4+附加题

## 环境依赖
pip install numpy matplotlib

## 运行方式
Jupyter Notebook 中使用：
新建空白ipynb，执行：
%run main.py

## 执行输出内容
1. Pe=0.1、Pe=3.0 数值解对比两张高清png图片
2. 网格收敛曲线png图片
3. 三种格式最大误差表格打印
4. Pe=3时Galerkin刚度矩阵对称/正定判定、矩阵局部输出