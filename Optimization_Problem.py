import numpy as np
from typing import Callable, Tuple, List

# ==============
# 优化问题定义
# ==============

class OptimizationProblem:
    """优化问题基类"""
    def __init__(self, dim: int, lb: float, ub: float, name: str):
        self.dim = dim  # 问题维度
        self.lb = lb    # 下界
        self.ub = ub    # 上界
        self.name = name
        
    def evaluate(self, x: np.ndarray) -> float:
        """评估解的适应度值"""
        raise NotImplementedError
        
    def get_bounds(self) -> List[Tuple[float, float]]:
        """获取问题的边界"""
        return [(self.lb, self.ub) for _ in range(self.dim)]

# 连续优化问题 - Sphere函数
class SphereProblem(OptimizationProblem):
    def __init__(self, dim: int = 10):
        super().__init__(dim, -100.0, 100.0, "Sphere")
        
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)

# 连续优化问题 - Rastrigin函数
class RastriginProblem(OptimizationProblem):
    def __init__(self, dim: int = 10):
        super().__init__(dim, -5.12, 5.12, "Rastrigin")
        
    def evaluate(self, x: np.ndarray) -> float:
        A = 10
        return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# 离散优化问题 - 0-1背包问题
class KnapsackProblem(OptimizationProblem):
    def __init__(self, n_items: int = 20, max_weight: int = 500):
        super().__init__(n_items, 0, 1, "Knapsack")
        self.max_weight = max_weight
        # 随机生成物品价值和重量
        self.values = np.random.randint(1, 100, n_items)
        self.weights = np.random.randint(1, 100, n_items)
        
    def evaluate(self, x: np.ndarray) -> float:
        # 将连续值转换为二进制决策
        x_binary = (x > 0.5).astype(int)
        total_weight = np.sum(x_binary * self.weights)
        # 如果超过容量，惩罚
        if total_weight > self.max_weight:
            return -total_weight  # 惩罚函数
        return np.sum(x_binary * self.values)  # 价值总和
