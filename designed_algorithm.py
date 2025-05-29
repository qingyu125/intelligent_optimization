import numpy as np
from Optimization_Problem import *
import random
# =====================
# 改进的连续优化算法 - 自适应多策略粒子群优化算法(AMPSO)
# =====================

class AMPSO:
    """自适应多策略粒子群优化算法"""
    def __init__(self, pop_size: int = 30, max_iter: int = 100, 
                 w_max: float = 0.9, w_min: float = 0.4, 
                 c1: float = 2.0, c2: float = 2.0):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重
        self.c1 = c1        # 个体学习因子
        self.c2 = c2        # 社会学习因子
        
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
        
        # 记录每代的最优适应度
        convergence_curve = np.zeros(self.max_iter)
        
        # 记录历史最优位置，用于计算收敛速度
        historical_best = np.zeros((self.max_iter, problem.dim))
        
        # 迭代优化
        for t in range(self.max_iter):
            # 非线性自适应惯性权重 - 采用凹函数，前期快速下降，后期缓慢下降
            w = self.w_min + (self.w_max - self.w_min) * ((self.max_iter - t) / self.max_iter) ** 2
            
            # 自适应学习因子 - 随迭代进行，个体学习因子减小，社会学习因子增大
            alpha = t / self.max_iter
            dynamic_c1 = self.c1 * (1 - alpha) + 0.5 * alpha  # 从c1逐渐减小到0.5
            dynamic_c2 = self.c2 * alpha + 0.5 * (1 - alpha)  # 从0.5逐渐增大到c2
            
            # 计算种群多样性
            population_center = np.mean(particles, axis=0)
            diversity = np.mean(np.linalg.norm(particles - population_center, axis=1))
            
            # 计算收敛速度
            if t > 0:
                convergence_speed = np.linalg.norm(historical_best[t-1] - gbest_position)
            else:
                convergence_speed = 0.0
            historical_best[t] = gbest_position.copy()
            
            # 更新每个粒子
            for i in range(self.pop_size):
                # 计算当前粒子与全局最优的距离
                distance_to_gbest = np.linalg.norm(particles[i] - gbest_position)
                
                # 根据距离和迭代进度动态调整策略选择概率
                if distance_to_gbest > diversity * 1.5:  # 远离全局最优，加强探索
                    p_basic = 0.5  # 基本PSO更新概率降低
                    p_gaussian = 0.3  # 高斯变异概率增加
                    p_local = 0.2    # 局部搜索概率增加
                else:  # 接近全局最优，加强开发
                    p_basic = 0.8  # 基本PSO更新概率增加
                    p_gaussian = 0.1  # 高斯变异概率降低
                    p_local = 0.1    # 局部搜索概率降低
                
                # 动态调整策略选择
                rand = random.random()
                
                # 1. 基本PSO更新
                if rand < p_basic:
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (w * velocities[i] + 
                                    dynamic_c1 * r1 * (pbest_positions[i] - particles[i]) + 
                                    dynamic_c2 * r2 * (gbest_position - particles[i]))
                
                # 2. 增强的高斯变异策略
                elif rand < p_basic + p_gaussian:
                    # 基于迭代进度调整高斯变异强度
                    mutation_strength = 0.1 * (1 - t / self.max_iter) + 0.01
                    velocities[i] *= np.random.normal(1, mutation_strength, problem.dim)
                
                # 3. 改进的局部搜索策略
                else:
                    # 基于迭代进度调整局部搜索范围
                    search_range = 0.1 * (1 - t / self.max_iter) + 0.01
                    local_pos = particles[i] + np.random.uniform(-search_range, search_range, problem.dim)
                    local_pos = np.clip(local_pos, problem.lb, problem.ub)
                    local_fitness = problem.evaluate(local_pos)
                    
                    if local_fitness < pbest_fitness[i]:
                        particles[i] = local_pos
                        pbest_fitness[i] = local_fitness
                
                # 更新位置
                particles[i] += velocities[i]
                
                # 改进的边界处理 - 采用反射机制
                for j in range(problem.dim):
                    if particles[i, j] < problem.lb:
                        particles[i, j] = problem.lb + (problem.lb - particles[i, j]) * 0.5
                        velocities[i, j] = -velocities[i, j] * 0.8  # 反弹并减速
                    elif particles[i, j] > problem.ub:
                        particles[i, j] = problem.ub - (particles[i, j] - problem.ub) * 0.5
                        velocities[i, j] = -velocities[i, j] * 0.8  # 反弹并减速
                
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
            
            # 记录当前代的最优适应度
            convergence_curve[t] = gbest_fitness
        
        return gbest_position, gbest_fitness

# =====================
# 改进的离散优化算法 - 混合遗传模拟退火算法(HGSA)
# =====================

class HGSA:
    """混合遗传模拟退火算法"""
    def __init__(self, pop_size: int = 30, max_iter: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 initial_temp: float = 100, cooling_rate: float = 0.95):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.initial_temp = initial_temp  # 初始温度
        self.cooling_rate = cooling_rate  # 冷却速率
        self.min_temp = 1e-10  # 最小温度，防止除零错误
        
    def optimize(self, problem: OptimizationProblem) -> Tuple[np.ndarray, float]:
        # 初始化种群（二进制编码）
        population = np.random.rand(self.pop_size, problem.dim)
        
        # 评估初始种群
        fitness_values = np.array([problem.evaluate(ind) for ind in population])
        
        # 找到初始最优解
        best_idx = np.argmax(fitness_values)  # 最大化问题
        best_individual = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        # 当前温度，确保不低于最小值
        current_temp = max(self.initial_temp, self.min_temp)
        
        # 迭代优化
        for t in range(self.max_iter):
            # 遗传操作
            new_population = []
            
            # 精英保留
            elite_idx = np.argmax(fitness_values)
            new_population.append(population[elite_idx].copy())
            
            # 选择操作 (锦标赛选择)
            for _ in range(self.pop_size - 1):
                # 随机选择三个个体进行锦标赛
                tournament_indices = np.random.choice(self.pop_size, 3, replace=False)
                tournament_fitness = fitness_values[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            new_population = np.array(new_population)
            
            # 交叉操作
            crossover_pairs = min(self.pop_size - 1, (self.pop_size - 1) // 2 * 2)  # 确保偶数对
            for i in range(1, crossover_pairs, 2):  # 从1开始，保留精英，步长为2
                if np.random.rand() < self.crossover_rate:
                    # 两点交叉
                    crossover_points = sorted(np.random.randint(1, problem.dim, 2))
                    temp = new_population[i, crossover_points[0]:crossover_points[1]].copy()
                    new_population[i, crossover_points[0]:crossover_points[1]] = \
                        new_population[i+1, crossover_points[0]:crossover_points[1]]
                    new_population[i+1, crossover_points[0]:crossover_points[1]] = temp
            
            # 变异操作
            for i in range(1, self.pop_size):  # 从1开始，保留精英
                for j in range(problem.dim):
                    if np.random.rand() < self.mutation_rate:
                        new_population[i, j] = np.random.rand()
            
            # 模拟退火接受准则 - 添加温度保护
            for i in range(1, self.pop_size):  # 从1开始，保留精英
                old_fitness = fitness_values[i]
                new_fitness = problem.evaluate(new_population[i])
                
                # 如果新解更好，接受
                if new_fitness > old_fitness:
                    fitness_values[i] = new_fitness
                # 否则，以一定概率接受
                else:
                    delta = old_fitness - new_fitness
                    # 使用受保护的温度值进行计算
                    safe_temp = max(current_temp, self.min_temp)
                    accept_prob = np.exp(-delta / safe_temp)
                    
                    if np.random.rand() < accept_prob:
                        fitness_values[i] = new_fitness
                    else:
                        new_population[i] = population[i].copy()  # 拒绝改变
            
            # 更新种群
            population = new_population
            
            # 更新全局最优
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_individual = population[current_best_idx]
            
            # 降温，确保不低于最小值
            current_temp = max(current_temp * self.cooling_rate, self.min_temp)
        
        return best_individual, best_fitness
