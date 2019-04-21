# -*- coding: utf-8 -*-
"""
   Description : about numpy
   Author :        xxm
"""
import numpy as np

'''
使用list初始化
'''
np_arr = np.array([i for i in range(10)])
print(np_arr.dtype)

'''
生成一些特殊矩阵
'''
zeros = np.zeros(10)  # 零矩阵 (默认浮点数)
print(zeros)

print(
    np.zeros((3, 5), dtype=int)
)

print(
    np.ones((3, 5))  # 单位矩阵
)

print(
    np.full(shape=(3, 5), fill_value=666)  # 填充矩阵
)

print(
    np.arange(0, 1, 0.2)  # 步长可为浮点数
)

print(
    np.linspace(0, 20, 10)  # arg3:在[0,20]中截取10个数
)

'''
随机数
'''
print(
    np.random.randint(0, 10)
)
print(
    np.random.randint(0, 10, size=10)  # 随机向量
)
print(
    np.random.randint(0, 10, size=(3, 5))  # 随机矩阵
)

np.random.seed(666)  # 随机种子

print(
    np.random.normal(0, 1, size=(3, 5))  # 生成符合正态分布的随机数
)

'''
操作numpy的数组
'''
print("\n数组基本操作:")
arr = np.random.randint(0, 15, size=(3, 5))
print(arr)
print(arr.ndim)  # 访问数组的维数
print(arr.shape)  # 维度
print(arr.size)  # 元素个数
# 切片访问多维数组(前两行,前三列) 注意: ,
print(arr[:2, :3])
# 倒序从头至尾访问
print(arr[::-1, ::-1])

'''
矩阵复制
'''
print(
    arr[:2, :3].copy()
)

'''
矩阵变型(变化后的size与变化前需要一致)
'''
print(arr)
Arr = arr.reshape(1, 15)
print(Arr)

'''
矩阵合并
'''
x = np.array([1, 2, 3])
y = np.arange(4, 7)
print(np.concatenate([x, y]))

z = np.array([[1, 2, 3], [4, 5, 6]])
print(z)
print(np.concatenate([z, z], axis=0))  # 列合并
print(np.concatenate([z, z], axis=1))  # 行合并

print(
    np.vstack([x, z])  # 竖直方向合并
)
print(
    np.hstack([x, y])  # 水平方向合并
)

'''
矩阵分割
'''
A = np.arange(16).reshape((4, 4))
A1, A2 = np.split(A, [2])  # 基于行分割
print(A1)
print(A2)
