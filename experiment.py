import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import time
import pandas as pd
from typing import Callable, Tuple, List

from Optimization_Problem import *
from baseline_algorithm import *
from designed_algorithm import *
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
# 设置随机种子确保结果可复现
np.random.seed(42)
random.seed(42)

# =====================
# 实验分析部分
# =====================

class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self, n_runs: int = 30):
        self.n_runs = n_runs
        
    def competitiveness_test(self, algorithms: List, problems: List) -> pd.DataFrame:
        """竞争力测试：比较不同算法在不同问题上的表现"""
        results = []
        
        for problem in problems:
            for algorithm in algorithms:
                run_times = []
                best_fitness_values = []
                
                for _ in range(self.n_runs):
                    start_time = time.time()
                    _, best_fitness = algorithm.optimize(problem)
                    run_times.append(time.time() - start_time)
                    best_fitness_values.append(best_fitness)
                
                # 计算统计指标
                mean_fitness = np.mean(best_fitness_values)
                std_fitness = np.std(best_fitness_values)
                mean_time = np.mean(run_times)
                
                results.append({
                    'Problem': problem.name,
                    'Algorithm': type(algorithm).__name__,
                    'Mean Fitness': mean_fitness,
                    'Std Fitness': std_fitness,
                    'Mean Time (s)': mean_time
                })
        
        return pd.DataFrame(results)
    
    def ablation_study(self, algorithm_class, problem, param_names: List, 
                       param_values: List) -> pd.DataFrame:
        """消融实验：研究算法组件的影响"""
        results = []
        
        # 基础配置
        base_params = {
            'pop_size': 30,
            'max_iter': 100
        }
        
        # 完整算法
        full_algorithm = algorithm_class(**base_params)
        full_fitness_values = []
        
        for _ in range(self.n_runs):
            _, best_fitness = full_algorithm.optimize(problem)
            full_fitness_values.append(best_fitness)
        
        results.append({
            'Configuration': 'Full Algorithm',
            'Mean Fitness': np.mean(full_fitness_values),
            'Std Fitness': np.std(full_fitness_values)
        })
        
        # 逐个移除组件
        for param_name, param_value in zip(param_names, param_values):
            ablation_params = base_params.copy()
            ablation_params[param_name] = param_value
            
            ablation_algorithm = algorithm_class(**ablation_params)
            ablation_fitness_values = []
            
            for _ in range(self.n_runs):
                _, best_fitness = ablation_algorithm.optimize(problem)
                ablation_fitness_values.append(best_fitness)
            
            results.append({
                'Configuration': f'Without {param_name}',
                'Mean Fitness': np.mean(ablation_fitness_values),
                'Std Fitness': np.std(ablation_fitness_values)
            })
        
        return pd.DataFrame(results)
    
    def parameter_sensitivity(self, algorithm_class, problem, 
                              param_name: str, param_range: List) -> pd.DataFrame:
        """参数敏感性分析"""
        results = []
        
        base_params = {
            'pop_size': 30,
            'max_iter': 100
        }
        
        for param_value in param_range:
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
            algorithm = algorithm_class(**test_params)
            fitness_values = []
            
            for _ in range(self.n_runs):
                _, best_fitness = algorithm.optimize(problem)
                fitness_values.append(best_fitness)
            
            results.append({
                f'{param_name}': param_value,
                'Mean Fitness': np.mean(fitness_values),
                'Std Fitness': np.std(fitness_values)
            })
        
        return pd.DataFrame(results)
    
    def plot_parameter_sensitivity(self, results_df: pd.DataFrame, param_name: str, save_path: str = None):
        """绘制参数敏感性分析结果并保存图片"""
        # 将英文参数名映射为中文
        param_name_map = {
            'w_min': '惯性权重下限',
            'initial_temp': '初始温度',
            'crossover_rate': '交叉率',
            'mutation_rate': '变异率'
        }
        chinese_param_name = param_name_map.get(param_name, param_name)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            results_df[param_name], 
            results_df['Mean Fitness'], 
            yerr=results_df['Std Fitness'],
            fmt='-o', capsize=5,
            label='平均适应度 (±标准差)'
        )
        plt.xlabel(chinese_param_name, fontsize=12)
        plt.ylabel('平均适应度值', fontsize=12)
        plt.title(f'{chinese_param_name}参数敏感性分析', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_competitiveness(self, results_df: pd.DataFrame, save_path: str = None):
        """绘制竞争力测试结果对比图"""
        # 将算法名称映射为中文
        algorithm_name_map = {
            'AMPSO': '自适应多策略PSO',
            'PSO': '标准PSO',
            'HGSA': '混合遗传模拟退火',
            'GA': '标准遗传算法'
        }
        
        # 将问题名称映射为中文
        problem_name_map = {
            'Sphere': 'Sphere函数',
            'Rastrigin': 'Rastrigin函数',
            'Knapsack': '背包问题'
        }
        
        plt.figure(figsize=(12, 6))
        for algorithm in results_df['Algorithm'].unique():
            subset = results_df[results_df['Algorithm'] == algorithm]
            chinese_algorithm = algorithm_name_map.get(algorithm, algorithm)
            
            # 组合问题名称和算法名称作为x轴标签
            x_labels = [f"{problem_name_map.get(p, p)}\n({chinese_algorithm})" 
                        for p in subset['Problem']]
            
            plt.bar(x_labels, subset['Mean Fitness'], yerr=subset['Std Fitness'], 
                    capsize=5, label=chinese_algorithm, alpha=0.8)
        
        plt.xlabel('优化问题', fontsize=12)
        plt.ylabel('平均适应度值', fontsize=12)
        plt.title('算法竞争力测试结果对比', fontsize=14, pad=20)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ablation_study(self, results_df: pd.DataFrame, algorithm_name: str, save_path: str = None):
        """绘制消融实验结果对比图"""
        # 将算法名称映射为中文
        algorithm_name_map = {
            'AMPSO': '自适应多策略PSO',
            'HGSA': '混合遗传模拟退火'
        }
        chinese_algorithm = algorithm_name_map.get(algorithm_name, algorithm_name)
        
        # 配置名称映射
        config_name_map = {
            'Full Algorithm': '完整算法',
            'Without w_max': '无最大惯性权重',
            'Without w_min': '无最小惯性权重',
            'Without c1': '无个体学习因子',
            'Without c2': '无社会学习因子',
            'Without crossover_rate': '无交叉操作',
            'Without mutation_rate': '无变异操作',
            'Without initial_temp': '无模拟退火'
        }
        
        plt.figure(figsize=(12, 6))
        # 按适应度排序
        results_df = results_df.sort_values('Mean Fitness', ascending=False)
        
        configs = [config_name_map.get(c, c) for c in results_df['Configuration']]
        means = results_df['Mean Fitness']
        stds = results_df['Std Fitness']
        
        plt.bar(configs, means, yerr=stds, capsize=5, alpha=0.8)
        plt.xlabel('算法配置', fontsize=12)
        plt.ylabel('平均适应度值', fontsize=12)
        plt.title(f'{chinese_algorithm}消融实验结果', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# =====================
# 主函数：运行实验
# =====================

def run_experiments():
    # 创建实验分析器
    analyzer = ExperimentAnalyzer(n_runs=50)
    
    # 1. 竞争力测试
    print("===== 竞争力测试 =====")
    # 准备问题
    continuous_problems = [
        SphereProblem(dim=10),
        RastriginProblem(dim=10)
    ]
    
    discrete_problems = [
        KnapsackProblem(n_items=20, max_weight=50)
    ]
    
    # 准备算法
    continuous_algorithms = [
        AMPSO(pop_size=30, max_iter=100),
        PSO(pop_size=30, max_iter=100),  # 作为基准比较
    ]
    
    discrete_algorithms = [
        HGSA(pop_size=30, max_iter=100),
        GA(pop_size=30, max_iter=100),   # 作为基准比较
    ]
    
    # 运行连续优化问题的竞争力测试
    continuous_results = analyzer.competitiveness_test(
        continuous_algorithms, continuous_problems
    )
    print("\n连续优化问题竞争力测试结果:")
    print(continuous_results)
    
    # 运行离散优化问题的竞争力测试
    discrete_results = analyzer.competitiveness_test(
        discrete_algorithms, discrete_problems
    )
    print("\n离散优化问题竞争力测试结果:")
    print(discrete_results)
    
    # 合并所有竞争力测试结果并可视化
    all_competitiveness = pd.concat([continuous_results, discrete_results])
    analyzer.plot_competitiveness(all_competitiveness, save_path='竞争力测试结果.png')
    
    # 2. 消融实验
    print("\n===== 消融实验 =====")
    # 连续优化算法(AMPSO)的消融实验
    print("\nAMPSO消融实验:")
    ampsso_param_names = ['w_max', 'w_min', 'c1', 'c2']
    ampsso_param_values = [0.7, 0.7, 0, 0]  # 禁用参数的影响
    ampsso_ablation_results = analyzer.ablation_study(
        AMPSO, SphereProblem(dim=10), ampsso_param_names, ampsso_param_values
    )
    print(ampsso_ablation_results)
    analyzer.plot_ablation_study(ampsso_ablation_results, 'AMPSO', save_path='AMPSO消融实验.png')
    
    # 离散优化算法(HGSA)的消融实验
    print("\nHGSA消融实验:")
    hgsa_param_names = ['crossover_rate', 'mutation_rate', 'initial_temp']
    hgsa_param_values = [0, 0, 0]  # 禁用参数的影响
    hgsa_ablation_results = analyzer.ablation_study(
        HGSA, KnapsackProblem(n_items=20, max_weight=50), hgsa_param_names, hgsa_param_values
    )
    print(hgsa_ablation_results)
    analyzer.plot_ablation_study(hgsa_ablation_results, 'HGSA', save_path='HGSA消融实验.png')
    
    # 3. 参数敏感性分析
    print("\n===== 参数敏感性分析 =====")
    # AMPSO惯性权重敏感性分析
    print("\nAMPSO惯性权重衰减率敏感性分析:")
    w_range = [0.2, 0.4, 0.6, 0.8, 1.0]  # 不同的w_min值
    w_sensitivity_results = analyzer.parameter_sensitivity(
        AMPSO, RastriginProblem(dim=10), 'w_min', w_range
    )
    print(w_sensitivity_results)
    analyzer.plot_parameter_sensitivity(w_sensitivity_results, 'w_min', save_path='AMPSO惯性权重敏感性.png')
    
    # HGSA初始温度敏感性分析
    print("\nHGSA初始温度敏感性分析:")
    temp_range = [10, 50, 100, 200, 500]
    temp_sensitivity_results = analyzer.parameter_sensitivity(
        HGSA, KnapsackProblem(n_items=20, max_weight=50), 'initial_temp', temp_range
    )
    print(temp_sensitivity_results)
    analyzer.plot_parameter_sensitivity(temp_sensitivity_results, 'initial_temp', save_path='HGSA初始温度敏感性.png')

if __name__ == "__main__":
    run_experiments()    