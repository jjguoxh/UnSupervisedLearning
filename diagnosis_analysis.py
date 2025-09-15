# -*- coding: utf-8 -*-
"""
深度诊断分析脚本
分析预测效果不好的根本原因
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def analyze_data_distribution():
    """
    分析数据分布
    """
    print("=== 数据分布分析 ===")
    
    # 分析标签文件
    label_files = glob.glob("./label/*.csv")
    print(f"找到 {len(label_files)} 个标签文件")
    
    all_labels = []
    file_stats = []
    
    for file_path in label_files[:10]:  # 分析前10个文件
        try:
            df = pd.read_csv(file_path)
            labels = df['label'].values
            
            label_counts = Counter(labels)
            file_stats.append({
                'file': os.path.basename(file_path),
                'total_samples': len(df),
                'label_0': label_counts.get(0, 0),
                'label_1': label_counts.get(1, 0),
                'label_2': label_counts.get(2, 0),
                'label_3': label_counts.get(3, 0),
                'label_4': label_counts.get(4, 0),
                'trading_signals': sum(label_counts.get(i, 0) for i in [1, 2, 3, 4]),
                'signal_rate': sum(label_counts.get(i, 0) for i in [1, 2, 3, 4]) / len(df)
            })
            
            all_labels.extend(labels)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 整体标签分布
    overall_counts = Counter(all_labels)
    print(f"\n整体标签分布:")
    for label in sorted(overall_counts.keys()):
        count = overall_counts[label]
        percentage = count / len(all_labels) * 100
        label_name = {0: '无信号', 1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}.get(label, f'标签{label}')
        print(f"  {label_name} ({label}): {count} ({percentage:.2f}%)")
    
    # 文件级别统计
    stats_df = pd.DataFrame(file_stats)
    print(f"\n文件级别统计:")
    print(f"  平均信号率: {stats_df['signal_rate'].mean():.4f}")
    print(f"  信号率标准差: {stats_df['signal_rate'].std():.4f}")
    print(f"  最高信号率: {stats_df['signal_rate'].max():.4f}")
    print(f"  最低信号率: {stats_df['signal_rate'].min():.4f}")
    
    return stats_df, overall_counts

def analyze_pattern_quality():
    """
    分析模式质量
    """
    print("\n=== 模式质量分析 ===")
    
    # 分析改进的模式
    patterns_dir = "./patterns_improved/"
    if os.path.exists(patterns_dir):
        analysis_file = os.path.join(patterns_dir, "cluster_analysis.csv")
        if os.path.exists(analysis_file):
            cluster_df = pd.read_csv(analysis_file)
            print(f"\n聚类质量统计:")
            print(f"  聚类数量: {len(cluster_df)}")
            print(f"  平均质量得分: {cluster_df['quality_score'].mean():.4f}")
            print(f"  平均信号密度: {cluster_df['signal_density'].mean():.4f}")
            print(f"  平均聚类大小: {cluster_df['cluster_size'].mean():.2f}")
            
            print(f"\n各聚类详情:")
            for _, row in cluster_df.iterrows():
                signal_counts = eval(row['signal_counts'])
                print(f"  聚类 {row['cluster_id']}: 质量={row['quality_score']:.3f}, 大小={row['cluster_size']}, 信号={signal_counts}")
            
            return cluster_df
    
    print("未找到改进的模式分析文件")
    return None

def analyze_prediction_bias():
    """
    分析预测偏差
    """
    print("\n=== 预测偏差分析 ===")
    
    results_file = "./models_improved/backtest_results.csv"
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        
        print(f"\n预测结果统计:")
        print(f"  总预测数: {len(results_df)}")
        
        # 预测分布
        pred_counts = Counter(results_df['predicted'])
        print(f"\n预测信号分布:")
        for signal in sorted(pred_counts.keys()):
            count = pred_counts[signal]
            percentage = count / len(results_df) * 100
            signal_name = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}[signal]
            print(f"  {signal_name} ({signal}): {count} ({percentage:.2f}%)")
        
        # 实际分布
        actual_counts = Counter(results_df['actual'])
        print(f"\n实际信号分布:")
        for signal in sorted(actual_counts.keys()):
            count = actual_counts[signal]
            percentage = count / len(results_df) * 100
            signal_name = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}[signal]
            print(f"  {signal_name} ({signal}): {count} ({percentage:.2f}%)")
        
        # 置信度分析
        print(f"\n置信度统计:")
        print(f"  平均置信度: {results_df['confidence'].mean():.4f}")
        print(f"  置信度标准差: {results_df['confidence'].std():.4f}")
        print(f"  最高置信度: {results_df['confidence'].max():.4f}")
        print(f"  最低置信度: {results_df['confidence'].min():.4f}")
        
        # 聚类使用情况
        cluster_usage = Counter(results_df['cluster_id'])
        print(f"\n聚类使用情况:")
        for cluster_id in sorted(cluster_usage.keys()):
            count = cluster_usage[cluster_id]
            percentage = count / len(results_df) * 100
            print(f"  聚类 {cluster_id}: {count} 次 ({percentage:.2f}%)")
        
        return results_df
    
    print("未找到预测结果文件")
    return None

def analyze_feature_correlation():
    """
    分析特征相关性
    """
    print("\n=== 特征相关性分析 ===")
    
    # 分析一个样本文件的特征
    sample_files = glob.glob("./label/*.csv")[:3]
    
    for file_path in sample_files:
        try:
            df = pd.read_csv(file_path)
            print(f"\n分析文件: {os.path.basename(file_path)}")
            
            # 基础统计
            print(f"  样本数: {len(df)}")
            print(f"  特征列: {list(df.columns)}")
            
            # 价格变化分析
            if 'index_value' in df.columns:
                prices = df['index_value'].values
                price_changes = np.diff(prices) / prices[:-1]
                print(f"  价格变化统计:")
                print(f"    均值: {np.mean(price_changes):.6f}")
                print(f"    标准差: {np.std(price_changes):.6f}")
                print(f"    最大涨幅: {np.max(price_changes):.6f}")
                print(f"    最大跌幅: {np.min(price_changes):.6f}")
            
            # 影响因子分析
            factors = ['a', 'b', 'c', 'd']
            for factor in factors:
                if factor in df.columns:
                    factor_values = df[factor].values
                    print(f"  因子 {factor} 统计:")
                    print(f"    均值: {np.mean(factor_values):.4f}")
                    print(f"    标准差: {np.std(factor_values):.4f}")
                    print(f"    范围: [{np.min(factor_values):.4f}, {np.max(factor_values):.4f}]")
            
            # 标签与特征的关系
            if 'label' in df.columns:
                trading_signals = df[df['label'].isin([1, 2, 3, 4])]
                if len(trading_signals) > 0:
                    print(f"  交易信号时的特征:")
                    if 'index_value' in df.columns:
                        signal_prices = trading_signals['index_value'].values
                        print(f"    信号时价格均值: {np.mean(signal_prices):.4f}")
                    
                    for factor in factors:
                        if factor in trading_signals.columns:
                            signal_factors = trading_signals[factor].values
                            print(f"    信号时因子{factor}均值: {np.mean(signal_factors):.4f}")
            
        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {e}")

def identify_key_issues():
    """
    识别关键问题
    """
    print("\n=== 关键问题识别 ===")
    
    issues = []
    
    # 检查数据质量
    label_files = glob.glob("./label/*.csv")
    if len(label_files) == 0:
        issues.append("❌ 没有找到标签数据文件")
    else:
        # 检查信号分布
        sample_df = pd.read_csv(label_files[0])
        label_counts = Counter(sample_df['label'])
        signal_rate = sum(label_counts.get(i, 0) for i in [1, 2, 3, 4]) / len(sample_df)
        
        if signal_rate < 0.01:
            issues.append(f"❌ 交易信号过少 (信号率: {signal_rate:.4f})")
        elif signal_rate < 0.05:
            issues.append(f"⚠️  交易信号较少 (信号率: {signal_rate:.4f})")
    
    # 检查模式质量
    patterns_dir = "./patterns_improved/"
    if not os.path.exists(patterns_dir):
        issues.append("❌ 没有找到改进的模式目录")
    else:
        analysis_file = os.path.join(patterns_dir, "cluster_analysis.csv")
        if os.path.exists(analysis_file):
            cluster_df = pd.read_csv(analysis_file)
            avg_quality = cluster_df['quality_score'].mean()
            
            if avg_quality < 0.3:
                issues.append(f"❌ 聚类质量过低 (平均质量: {avg_quality:.4f})")
            elif avg_quality < 0.5:
                issues.append(f"⚠️  聚类质量一般 (平均质量: {avg_quality:.4f})")
    
    # 检查预测结果
    results_file = "./models_improved/backtest_results.csv"
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        
        # 检查预测偏差
        pred_counts = Counter(results_df['predicted'])
        if len(pred_counts) == 1:
            dominant_signal = list(pred_counts.keys())[0]
            issues.append(f"❌ 预测器只预测单一信号 (信号 {dominant_signal})")
        
        # 检查置信度
        avg_confidence = results_df['confidence'].mean()
        if avg_confidence < 0.5:
            issues.append(f"❌ 预测置信度过低 (平均置信度: {avg_confidence:.4f})")
    
    # 输出问题
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 未发现明显问题")
    
    return issues

def suggest_improvements():
    """
    建议改进方案
    """
    print("\n=== 改进建议 ===")
    
    suggestions = [
        "1. 数据质量改进:",
        "   - 增加更多样化的交易信号数据",
        "   - 平衡各类信号的分布",
        "   - 检查数据标注的准确性",
        "",
        "2. 特征工程改进:",
        "   - 增加更多技术指标 (MACD, KDJ, 布林带等)",
        "   - 考虑多时间尺度特征",
        "   - 添加市场情绪指标",
        "",
        "3. 模式识别改进:",
        "   - 使用更复杂的相似性度量",
        "   - 考虑时序依赖关系",
        "   - 引入深度学习方法",
        "",
        "4. 预测模型改进:",
        "   - 使用集成学习方法",
        "   - 添加模型校准",
        "   - 实现在线学习机制",
        "",
        "5. 评估方法改进:",
        "   - 使用更合适的评估指标",
        "   - 考虑交易成本和滑点",
        "   - 实现风险调整后的收益评估"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """
    主函数
    """
    print("开始深度诊断分析...")
    
    # 数据分布分析
    stats_df, label_counts = analyze_data_distribution()
    
    # 模式质量分析
    cluster_df = analyze_pattern_quality()
    
    # 预测偏差分析
    results_df = analyze_prediction_bias()
    
    # 特征相关性分析
    analyze_feature_correlation()
    
    # 识别关键问题
    issues = identify_key_issues()
    
    # 建议改进方案
    suggest_improvements()
    
    print("\n=== 诊断分析完成 ===")

if __name__ == "__main__":
    main()