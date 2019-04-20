# -*- coding: utf-8 -*-
"""
   Description : about numpy
   Author :        xxm
"""
import numpy as np

# 使用list初始化
nparr = np.array([i for i in range(10)])
print(nparr.dtype)

# 零矩阵 (默认浮点数)
zeros = np.zeros(10)
print(zeros)

print(
    np.zeros((3, 5), dtype=int)
)

# 单位矩阵
print(
    np.ones((3, 5))
)

# 填充矩阵
print(
    np.full(shape=(3, 5), fill_value=666)
)

# 步长可为浮点数
print(
    np.arange(0, 1, 0.2)
)

# arg3:在[0,20]中截取10个数
print(
    np.linspace(0, 20, 10)
)

# 随机数
print(
    np.random.randint(0, 10)
)
# 随机向量
print(
    np.random.randint(0, 10, size=10)
)
# 随机矩阵
print(
    np.random.randint(0, 10, size=(3, 5))
)
# 随机种子
np.random.seed(666)

# 生成符合正态分布的随机数
print(
    np.random.normal(0, 1, size=(3, 5))
)

print("\n数组基本操作:")
arr = np.random.randint(0, 15, size=(3, 5))
print(arr)
# 访问数组的维数
print(arr.ndim)
print(arr.shape)
# 切片(前两行,前三列) 注意: ,
print(arr[:2, :3])
