# -*- coding: utf-8 -*-
"""
无监督学习系统诊断和改进方案
分析当前系统问题并提供具体的改进建议
"""

import os
import pandas as pd
import numpy as np
import glob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SystemDiagnostic:
    """
    系统诊断类
    """
    def __init__(self):
        self.issues = []
        self.recommendations = []
        
    def check_data_availability(self):
        """
        检查数据可用性
        """
        print("=== 数据可用性检查 ===")
        
        # 检查各个目录
        directories = {
            'data': './data/',
            'label': './label/', 
            'patterns': './patterns/',
            'model': './model/',
            'predictions': './predictions/'
        }
        
        for name, path in directories.items():
            if os.path.exists(path):
                files = os.listdir(path)
                print(f"✓ {name} 目录存在，包含 {len(files)} 个文件/文件夹")
                if len(files) == 0:
                    self.issues.append(f"{name} 目录为空")
                    print(f"  ⚠️  警告: {name} 目录为空")
            else:
                self.issues.append(f"{name} 目录不存在")
                print(f"✗ {name} 目录不存在")
                
    def analyze_pattern_quality(self):
        """
        分析模式质量
        """
        print("\n=== 模式质量分析 ===")
        
        patterns_dir = './patterns/'
        if not os.path.exists(patterns_dir):
            print("✗ patterns 目录不存在")
            return
            
        # 统计聚类数量
        cluster_dirs = [d for d in os.listdir(patterns_dir) if d.startswith('cluster_')]
        print(f"发现 {len(cluster_dirs)} 个聚类")
        
        if len(cluster_dirs) > 1000:
            self.issues.append("聚类数量过多，可能导致过拟合")
            self.recommendations.append("减少聚类数量，建议使用50-200个聚类")
            
        # 检查聚类内容
        empty_clusters = 0
        for cluster_dir in cluster_dirs[:10]:  # 检查前10个聚类
            cluster_path = os.path.join(patterns_dir, cluster_dir)
            if os.path.isdir(cluster_path):
                files = os.listdir(cluster_path)
                if len(files) == 0:
                    empty_clusters += 1
                    
        if empty_clusters > 0:
            self.issues.append(f"发现 {empty_clusters} 个空聚类")
            self.recommendations.append("清理空聚类，重新进行聚类分析")
            
    def check_model_performance(self):
        """
        检查模型性能
        """
        print("\n=== 模型性能检查 ===")
        
        model_dir = './model/'
        if os.path.exists(model_dir):
            model_files = glob.glob(os.path.join(model_dir, '**', '*.pkl'), recursive=True)
            model_files.extend(glob.glob(os.path.join(model_dir, '**', '*.json'), recursive=True))
            model_files.extend(glob.glob(os.path.join(model_dir, '**', '*.pth'), recursive=True))
            
            print(f"发现 {len(model_files)} 个模型文件")
            for model_file in model_files:
                print(f"  - {os.path.relpath(model_file)}")
                
            if len(model_files) == 0:
                self.issues.append("没有找到训练好的模型")
                self.recommendations.append("需要重新训练模型")
        else:
            self.issues.append("模型目录不存在")
            
    def analyze_signal_distribution(self):
        """
        分析信号分布
        """
        print("\n=== 信号分布分析 ===")
        
        label_dir = './label/'
        if not os.path.exists(label_dir):
            print("✗ label 目录不存在")
            self.issues.append("标签数据缺失")
            return
            
        label_files = glob.glob(os.path.join(label_dir, '*.csv'))
        if len(label_files) == 0:
            print("✗ 没有找到标签文件")
            self.issues.append("标签文件缺失")
            return
            
        print(f"发现 {len(label_files)} 个标签文件")
        
        # 分析信号分布
        all_signals = []
        for file_path in label_files[:5]:  # 分析前5个文件
            try:
                df = pd.read_csv(file_path)
                if 'label' in df.columns:
                    signals = df['label'].values
                    all_signals.extend(signals)
            except Exception as e:
                print(f"读取文件 {file_path} 失败: {e}")
                
        if all_signals:
            signal_counts = Counter(all_signals)
            print("信号分布:")
            for signal, count in sorted(signal_counts.items()):
                percentage = count / len(all_signals) * 100
                print(f"  信号 {signal}: {count} ({percentage:.1f}%)")
                
            # 检查信号不平衡
            if signal_counts.get(0, 0) / len(all_signals) > 0.95:
                self.issues.append("信号严重不平衡，0信号占比过高")
                self.recommendations.append("使用SMOTE或其他方法处理类别不平衡")
                
            # 检查交易信号稀疏性
            trading_signals = sum(count for signal, count in signal_counts.items() if signal != 0)
            if trading_signals / len(all_signals) < 0.05:
                self.issues.append("交易信号过于稀疏")
                self.recommendations.append("调整标签生成策略，增加有效交易信号")
                
    def generate_improvement_plan(self):
        """
        生成改进计划
        """
        print("\n=== 系统改进建议 ===")
        
        if not self.issues:
            print("✓ 未发现明显问题")
            return
            
        print("发现的问题:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
            
        print("\n改进建议:")
        
        # 基础建议
        basic_recommendations = [
            "确保有足够的原始交易数据（建议至少100个交易日）",
            "重新生成标签数据，确保标签逻辑正确",
            "优化聚类参数，减少聚类数量到合理范围（50-200个）",
            "使用更robust的特征工程方法",
            "实施交叉验证和更严格的模型评估",
            "添加数据增强技术处理信号稀疏性"
        ]
        
        all_recommendations = self.recommendations + basic_recommendations
        for i, rec in enumerate(set(all_recommendations), 1):
            print(f"  {i}. {rec}")
            
    def create_sample_data(self):
        """
        创建示例数据用于测试
        """
        print("\n=== 创建示例数据 ===")
        
        # 创建目录
        os.makedirs('./data/', exist_ok=True)
        os.makedirs('./label/', exist_ok=True)
        
        # 生成示例交易数据
        np.random.seed(42)
        n_days = 5
        points_per_day = 1000
        
        for day in range(n_days):
            # 生成基础时间序列
            t = np.linspace(0, 24*3600, points_per_day)  # 一天的秒数
            
            # 生成特征
            trend = 0.001 * t + np.random.normal(0, 0.1, points_per_day)
            x = t
            a = np.sin(2*np.pi*t/3600) + np.random.normal(0, 0.1, points_per_day)  # 小时周期
            b = np.cos(2*np.pi*t/1800) + np.random.normal(0, 0.1, points_per_day)  # 30分钟周期
            c = np.random.normal(0, 0.2, points_per_day)
            d = np.random.normal(0, 0.2, points_per_day)
            
            # 生成指数价格（随机游走 + 趋势）
            price_changes = np.random.normal(0, 0.01, points_per_day)
            price_changes += 0.001 * (a + b)  # 受a,b影响
            index_value = 3000 + np.cumsum(price_changes)  # 从3000开始
            
            # 生成交易信号（稀疏）
            labels = np.zeros(points_per_day)
            
            # 随机生成一些交易信号
            signal_positions = np.random.choice(points_per_day, size=20, replace=False)
            signal_types = np.random.choice([1, 2, 3, 4], size=20)
            labels[signal_positions] = signal_types
            
            # 创建DataFrame
            df = pd.DataFrame({
                'x': x,
                'a': a,
                'b': b, 
                'c': c,
                'd': d,
                'index_value': index_value,
                'label': labels.astype(int)
            })
            
            # 保存文件
            filename = f'sample_day_{day+1:02d}.csv'
            df.to_csv(f'./data/{filename}', index=False)
            df.to_csv(f'./label/{filename}', index=False)
            
        print(f"✓ 已生成 {n_days} 天的示例数据")
        print("  - 数据保存在 ./data/ 目录")
        print("  - 标签保存在 ./label/ 目录")
        
    def run_full_diagnosis(self):
        """
        运行完整诊断
        """
        print("开始系统诊断...\n")
        
        self.check_data_availability()
        self.analyze_pattern_quality()
        self.check_model_performance()
        self.analyze_signal_distribution()
        self.generate_improvement_plan()
        
        # 如果数据缺失，提供创建示例数据的选项
        if any('目录为空' in issue or '缺失' in issue for issue in self.issues):
            print("\n=== 数据修复选项 ===")
            print("检测到数据缺失问题。")
            print("建议:")
            print("1. 如果有真实数据，请将CSV文件放入 ./data/ 目录")
            print("2. 运行 create_sample_data() 方法生成示例数据进行测试")
            
def main():
    """
    主函数
    """
    diagnostic = SystemDiagnostic()
    
    print("无监督学习交易信号预测系统 - 诊断工具")
    print("=" * 50)
    
    # 运行诊断
    diagnostic.run_full_diagnosis()
    
    print("\n=== 下一步行动建议 ===")
    print("1. 根据诊断结果修复发现的问题")
    print("2. 如果需要示例数据，运行: diagnostic.create_sample_data()")
    print("3. 重新训练模型")
    print("4. 验证预测性能")
    
    return diagnostic

if __name__ == "__main__":
    diagnostic = main()
    
    # 如果需要创建示例数据，取消下面的注释
    # diagnostic.create_sample_data()