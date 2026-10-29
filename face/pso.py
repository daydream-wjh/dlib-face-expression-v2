import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 数据只读取一次（原代码已在目标函数外加载，语义不变）
df = pd.read_excel("../数据集/数据.xlsx", header=None)
# 说明：这里沿用原来的 0:5（5 个特征）；同目录 SVM.py 用的是 0:6，如需对齐属另一处改动
X = df[df.columns[0:5]]
y = df[df.columns[6]]

# 划分训练集和测试集（同样只做一次，random_state 保持不变）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义SVM模型的参数优化目标函数
def optimize_svm(params):
    """PSO 目标函数：返回每个粒子的分类错误率（越小越好）。

    PySwarms 会把整个粒子矩阵（shape=(n_particles, dimensions)）一次性传进来，
    因此必须逐粒子评估并返回长度等于粒子数的代价数组；这里同时兼容
    传入单个参数向量（1-D）的调用方式。
    params[0] = C，params[1] = gamma。
    """
    particles = np.atleast_2d(np.asarray(params, dtype=float))
    costs = np.empty(particles.shape[0], dtype=float)

    for i, particle in enumerate(particles):
        C = particle[0]      # 正则化参数
        gamma = particle[1]  # 核函数系数
        svm_model = SVC(C=C, gamma=gamma, kernel='rbf')  # 创建SVC模型
        svm_model.fit(X_train, y_train)  # 在训练集上拟合模型
        accuracy = svm_model.score(X_test, y_test)  # 在测试集上评估模型性能
        costs[i] = 1 - accuracy  # 优化目标为最小化分类错误率

    return costs


# 设置PSO算法的参数
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 定义参数空间
max_bound = np.array([100, 100])  # 参数上界
min_bound = np.array([0.1, 0.1])  # 参数下界
bounds = (min_bound, max_bound)

# 使用PSO算法进行参数优化
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
result = optimizer.optimize(optimize_svm, iters=100)

# optimize() 返回 (best_cost, best_pos)；兼容只返回 best_cost 的旧版本
if isinstance(result, tuple) and len(result) == 2:
    best_cost, best_pos = result
else:
    best_cost, best_pos = result, optimizer.swarm.best_pos

# 输出最佳参数组合：best_pos[0] = C，best_pos[1] = gamma
print(f"最佳参数: C={best_pos[0]:.4f}, gamma={best_pos[1]:.4f}, 最优得分={best_cost:.4f}")
