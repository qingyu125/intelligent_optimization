import numpy as np
from Optimization_Problem import *

class PSO:
    """粒子群优化算法"""
    def __init__(self, pop_size: int = 30, max_iter: int = 100, 
                 w: float = 0.7, c1: float = 1.4, c2: float = 1.4):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w      # 惯性权重
        self.c1 = c1    # 个体学习因子
        self.c2 = c2    # 社会学习因子
        
    def optimize(self, problem: OptimizationProblem) -> Tuple[np.ndarray, float]:
        # 初始化粒子位置和速度
        particles = np.random.uniform(
            problem.lb, problem.ub, (self.pop_size, problem.dim)
        )
        velocities = np.random.uniform(
            -abs(problem.ub - problem.lb), 
            abs(problem.ub - problem.lb), 
            (self.pop_size, problem.dim)
        )
        
        # 初始化个体最优和全局最优
        pbest_positions = particles.copy()
        pbest_fitness = np.array([problem.evaluate(p) for p in particles])
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        
        # 迭代优化
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # 更新速度和位置
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (pbest_positions[i] - particles[i]) + 
                                self.c2 * r2 * (gbest_position - particles[i]))
                particles[i] += velocities[i]
                
                # 边界处理
                particles[i] = np.clip(particles[i], problem.lb, problem.ub)
                
                # 评估新位置
                fitness = problem.evaluate(particles[i])
                
                # 更新个体最优
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = particles[i]
                    
                    # 更新全局最优
                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = particles[i]
                        
        return gbest_position, gbest_fitness
    
class GA:
    """遗传算法"""
    def __init__(self, pop_size: int = 30, max_iter: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
    def optimize(self, problem: OptimizationProblem) -> Tuple[np.ndarray, float]:
        # 初始化种群（二进制编码）
        population = np.random.rand(self.pop_size, problem.dim)
        
        # 迭代优化
        for _ in range(self.max_iter):
            # 评估适应度
            fitness_values = np.array([problem.evaluate(ind) for ind in population])
            
            # 找到最优解
            best_idx = np.argmax(fitness_values)  # 最大化问题
            best_individual = population[best_idx]
            best_fitness = fitness_values[best_idx]
            
            # 选择操作 (锦标赛选择)
            new_population = []
            for _ in range(self.pop_size):
                # 随机选择三个个体进行锦标赛
                tournament_indices = np.random.choice(self.pop_size, 3, replace=False)
                tournament_fitness = fitness_values[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            new_population = np.array(new_population)
            
            # 交叉操作
            for i in range(0, self.pop_size, 2):
                if np.random.rand() < self.crossover_rate:
                    # 单点交叉
                    crossover_point = np.random.randint(1, problem.dim)
                    temp = new_population[i, crossover_point:].copy()
                    new_population[i, crossover_point:] = new_population[i+1, crossover_point:]
                    new_population[i+1, crossover_point:] = temp
            
            # 变异操作
            for i in range(self.pop_size):
                for j in range(problem.dim):
                    if np.random.rand() < self.mutation_rate:
                        new_population[i, j] = np.random.rand()
            
            population = new_population
            
        return best_individual, best_fitness