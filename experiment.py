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
    
    def orthogonal_experiment(self, algorithm_class, problem, param_ranges: dict, 
                             orthogonal_array: np.ndarray) -> pd.DataFrame:
        """正交实验设计：多参数敏感性分析"""
        results = []
        
        base_params = {
            'pop_size': 30,
            'max_iter': 100
        }
        
        param_names = list(param_ranges.keys())
        levels = [len(param_ranges[name]) for name in param_names]
        
        # 确保正交表的列数与参数数量匹配
        assert orthogonal_array.shape[1] == len(param_names), "正交表列数与参数数量不匹配"
        
        for run_idx, row in enumerate(orthogonal_array):
            test_params = base_params.copy()
            
            # 根据正交表设置参数值
            for i, param_name in enumerate(param_names):
                level = row[i]
                test_params[param_name] = param_ranges[param_name][level]
            
            algorithm = algorithm_class(**test_params)
            fitness_values = []
            
            for _ in range(self.n_runs):
                _, best_fitness = algorithm.optimize(problem)
                fitness_values.append(best_fitness)
            
            # 记录实验结果
            result_entry = {f'{param_name}': test_params[param_name] for param_name in param_names}
            result_entry.update({
                'Run': run_idx + 1,
                'Mean Fitness': np.mean(fitness_values),
                'Std Fitness': np.std(fitness_values)
            })
            results.append(result_entry)
        
        return pd.DataFrame(results)
    
    def analyze_orthogonal_results(self, results_df: pd.DataFrame, param_ranges: dict):
        """分析正交实验结果，计算各参数的极差和贡献率"""
        param_names = list(param_ranges.keys())
        n_levels = len(next(iter(param_ranges.values())))  # 假设所有参数水平数相同
        
        analysis_results = []
        
        for param_name in param_names:
            # 计算每个水平下的平均适应度
            level_means = []
            for level in range(n_levels):
                level_value = param_ranges[param_name][level]
                level_data = results_df[results_df[param_name] == level_value]
                level_mean = level_data['Mean Fitness'].mean()
                level_means.append(level_mean)
            
            # 计算极差（最大值-最小值）
            range_value = max(level_means) - min(level_means)
            
            # 计算该参数的贡献率（简化计算，实际应通过方差分析）
            total_mean = results_df['Mean Fitness'].mean()
            sst = sum((results_df['Mean Fitness'] - total_mean) ** 2)
            
            ss_param = 0
            for level in range(n_levels):
                level_value = param_ranges[param_name][level]
                level_data = results_df[results_df[param_name] == level_value]
                level_mean = level_data['Mean Fitness'].mean()
                ss_param += len(level_data) * (level_mean - total_mean) ** 2
            
            contribution_rate = ss_param / sst * 100 if sst != 0 else 0
            
            analysis_results.append({
                '参数': param_name,
                '最优水平': param_ranges[param_name][np.argmin(level_means)],
                '极差': range_value,
                '贡献率(%)': contribution_rate,
                '水平均值': level_means
            })
        
        # 按贡献率排序
        analysis_results = sorted(analysis_results, key=lambda x: x['贡献率(%)'], reverse=True)
        
        return pd.DataFrame(analysis_results)
    
    def plot_orthogonal_results(self, analysis_df: pd.DataFrame, param_ranges: dict, save_path: str = None):
        """绘制正交实验结果分析图"""
        plt.figure(figsize=(12, 6))
        
        n_params = len(analysis_df)
        for i, row in analysis_df.iterrows():
            param_name = row['参数']
            level_means = row['水平均值']
            x_pos = np.arange(len(level_means))
            
            # 将英文参数名映射为中文
            param_name_map = {
                'w_min': '惯性权重下限',
                'w_max': '惯性权重上限',
                'c1': '个体学习因子',
                'c2': '社会学习因子',
                'initial_temp': '初始温度',
                'cooling_rate': '冷却率',
                'crossover_rate': '交叉率',
                'mutation_rate': '变异率'
            }
            chinese_param_name = param_name_map.get(param_name, param_name)
            
            plt.subplot(1, n_params, i+1)
            plt.bar(x_pos, level_means, align='center', alpha=0.7)
            plt.xticks(x_pos, param_ranges[param_name])
            plt.xlabel(f'{chinese_param_name}水平')
            plt.ylabel('平均适应度')
            plt.title(f'{chinese_param_name}对性能的影响')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
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
        KnapsackProblem(n_items=20, max_weight=500)
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
    
    # 3. 参数敏感性分析（修改为正交实验设计）
    print("\n===== 参数敏感性分析（正交实验） =====")
    
    # 3.1 AMPSO多参数正交实验
    print("\nAMPSO参数敏感性正交实验:")
    # 定义AMPSO参数及其水平
    ampsso_param_ranges = {
        'w_min': [0.2, 0.4, 0.6],  # 惯性权重下限
        'w_max': [0.7, 0.8, 0.9],  # 惯性权重上限
        'c1': [1.5, 2.0, 2.5],     # 个体学习因子
        'c2': [1.5, 2.0, 2.5]      # 社会学习因子
    }
    
    # 使用L9(3^4)正交表（3水平4因素）
    ampsso_orthogonal_array = np.array([
        [0, 0, 0, 0],  # 第1行：w_min=0.2, w_max=0.7, c1=1.5, c2=1.5
        [0, 1, 1, 1],  # 第2行：w_min=0.2, w_max=0.8, c1=2.0, c2=2.0
        [0, 2, 2, 2],  # 第3行：w_min=0.2, w_max=0.9, c1=2.5, c2=2.5
        [1, 0, 1, 2],  # 第4行：w_min=0.4, w_max=0.7, c1=2.0, c2=2.5
        [1, 1, 2, 0],  # 第5行：w_min=0.4, w_max=0.8, c1=2.5, c2=1.5
        [1, 2, 0, 1],  # 第6行：w_min=0.4, w_max=0.9, c1=1.5, c2=2.0
        [2, 0, 2, 1],  # 第7行：w_min=0.6, w_max=0.7, c1=2.5, c2=2.0
        [2, 1, 0, 2],  # 第8行：w_min=0.6, w_max=0.8, c1=1.5, c2=2.5
        [2, 2, 1, 0]   # 第9行：w_min=0.6, w_max=0.9, c1=2.0, c2=1.5
    ])
    
    # 运行正交实验
    ampsso_orthogonal_results = analyzer.orthogonal_experiment(
        AMPSO, RastriginProblem(dim=10), ampsso_param_ranges, ampsso_orthogonal_array
    )
    print("\nAMPSO正交实验结果:")
    print(ampsso_orthogonal_results)
    
    # 分析正交实验结果
    ampsso_analysis = analyzer.analyze_orthogonal_results(ampsso_orthogonal_results, ampsso_param_ranges)
    print("\nAMPSO正交实验分析结果:")
    print(ampsso_analysis)
    
    # 可视化正交实验结果
    analyzer.plot_orthogonal_results(ampsso_analysis, ampsso_param_ranges, save_path='AMPSO正交实验.png')
    
    # 3.2 HGSA多参数正交实验
    print("\nHGSA参数敏感性正交实验:")
    # 定义HGSA参数及其水平
    hgsa_param_ranges = {
        'crossover_rate': [0.6, 0.7, 0.8],  # 交叉率
        'mutation_rate': [0.05, 0.1, 0.15],  # 变异率
        'initial_temp': [50, 100, 200],     # 初始温度
        'cooling_rate': [0.9, 0.95, 0.99]   # 冷却率
    }
    
    # 使用L9(3^4)正交表
    hgsa_orthogonal_array = np.array([
        [0, 0, 0, 0],  # 第1行
        [0, 1, 1, 1],  # 第2行
        [0, 2, 2, 2],  # 第3行
        [1, 0, 1, 2],  # 第4行
        [1, 1, 2, 0],  # 第5行
        [1, 2, 0, 1],  # 第6行
        [2, 0, 2, 1],  # 第7行
        [2, 1, 0, 2],  # 第8行
        [2, 2, 1, 0]   # 第9行
    ])
    
    # 运行正交实验
    hgsa_orthogonal_results = analyzer.orthogonal_experiment(
        HGSA, KnapsackProblem(n_items=20, max_weight=50), hgsa_param_ranges, hgsa_orthogonal_array
    )
    print("\nHGSA正交实验结果:")
    print(hgsa_orthogonal_results)
    
    # 分析正交实验结果
    hgsa_analysis = analyzer.analyze_orthogonal_results(hgsa_orthogonal_results, hgsa_param_ranges)
    print("\nHGSA正交实验分析结果:")
    print(hgsa_analysis)
    
    # 可视化正交实验结果
    analyzer.plot_orthogonal_results(hgsa_analysis, hgsa_param_ranges, save_path='HGSA正交实验.png')

if __name__ == "__main__":
    run_experiments()    
