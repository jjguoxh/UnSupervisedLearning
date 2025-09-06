# -*- coding: utf-8 -*-
"""
彻底的聚类平衡工具
通过重新采样实现信号分布的真正平衡
"""

import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# ========= 配置参数 =========
# 使用相对于脚本位置的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERNS_DIR = os.path.join(CURRENT_DIR, "..", "patterns/")  # 模式数据目录

def analyze_cluster_signals():
    """
    分析聚类中的信号分布
    """
    print("=== 聚类信号分析 ===")
    
    # 读取聚类分析结果
    cluster_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
    if not os.path.exists(cluster_file):
        print(f"未找到聚类分析文件: {cluster_file}")
        return None
    
    try:
        df = pd.read_csv(cluster_file)
        print(f"找到 {len(df)} 个聚类")
        
        # 按信号类型统计聚类
        signal_clusters = defaultdict(list)
        
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            # 检查每种信号类型
            if row['long_open'] > 0:
                signal_clusters['long_open'].append({
                    'cluster_id': cluster_id,
                    'signal_count': row['long_open'],
                    'row': row
                })
            if row['long_close'] > 0:
                signal_clusters['long_close'].append({
                    'cluster_id': cluster_id,
                    'signal_count': row['long_close'],
                    'row': row
                })
            if row['short_open'] > 0:
                signal_clusters['short_open'].append({
                    'cluster_id': cluster_id,
                    'signal_count': row['short_open'],
                    'row': row
                })
            if row['short_close'] > 0:
                signal_clusters['short_close'].append({
                    'cluster_id': cluster_id,
                    'signal_count': row['short_close'],
                    'row': row
                })
        
        print(f"\n各信号类型聚类数量:")
        for signal_type, clusters in signal_clusters.items():
            total_signals = sum(c['signal_count'] for c in clusters)
            print(f"  {signal_type}: {len(clusters)} 个聚类, {total_signals} 个信号")
        
        return df, signal_clusters
        
    except Exception as e:
        print(f"分析聚类信号时出错: {e}")
        return None, None

def balance_signal_clusters(signal_clusters, target_count=500):
    """
    平衡各信号类型的聚类数量
    """
    print(f"\n=== 平衡信号聚类 (目标数量: {target_count}) ===")
    
    balanced_clusters = {}
    
    for signal_type, clusters in signal_clusters.items():
        print(f"\n处理 {signal_type}:")
        print(f"  原始聚类数: {len(clusters)}")
        
        if len(clusters) > target_count:
            # 下采样
            # 按信号数量进行加权采样
            signal_counts = [c['signal_count'] for c in clusters]
            total_signals = sum(signal_counts)
            
            # 归一化权重
            weights = [count/total_signals for count in signal_counts]
            
            # 随机采样
            selected_indices = np.random.choice(
                len(clusters), 
                size=target_count, 
                replace=False,
                p=weights if sum(weights) > 0 else None
            )
            
            balanced_clusters[signal_type] = [clusters[i] for i in selected_indices]
            print(f"  采样后聚类数: {len(balanced_clusters[signal_type])}")
        else:
            # 保持原样
            balanced_clusters[signal_type] = clusters
            print(f"  保持原聚类数: {len(balanced_clusters[signal_type])}")
    
    return balanced_clusters

def create_final_balanced_dataset(original_df, balanced_clusters):
    """
    创建最终的平衡数据集
    """
    print("\n=== 创建最终平衡数据集 ===")
    
    # 收集所有选中的聚类ID
    selected_cluster_ids = set()
    for clusters in balanced_clusters.values():
        for cluster in clusters:
            selected_cluster_ids.add(cluster['cluster_id'])
    
    print(f"选中的聚类数量: {len(selected_cluster_ids)}")
    
    # 从原始数据中筛选出选中的聚类
    final_df = original_df[original_df['cluster_id'].isin(selected_cluster_ids)].copy()
    
    # 重新计算信号分布
    total_long_open = final_df['long_open'].sum()
    total_long_close = final_df['long_close'].sum()
    total_short_open = final_df['short_open'].sum()
    total_short_close = final_df['short_close'].sum()
    
    total_signals = total_long_open + total_long_close + total_short_open + total_short_close
    
    print(f"\n平衡后的信号分布:")
    print(f"  做多开仓: {total_long_open} ({total_long_open/total_signals*100:.2f}%)")
    print(f"  做多平仓: {total_long_close} ({total_long_close/total_signals*100:.2f}%)")
    print(f"  做空开仓: {total_short_open} ({total_short_open/total_signals*100:.2f}%)")
    print(f"  做空平仓: {total_short_close} ({total_short_close/total_signals*100:.2f}%)")
    
    return final_df

def save_balanced_dataset(balanced_df):
    """
    保存平衡后的数据集
    """
    print("\n=== 保存平衡数据集 ===")
    
    try:
        # 创建平衡数据目录
        balanced_dir = os.path.join(PATTERNS_DIR, "balanced")
        os.makedirs(balanced_dir, exist_ok=True)
        
        # 保存平衡后的聚类分析文件
        output_file = os.path.join(balanced_dir, "cluster_analysis_balanced.csv")
        balanced_df.to_csv(output_file, index=False)
        print(f"平衡后的聚类分析结果已保存到: {output_file}")
        
        # 创建一个报告文件
        report_file = os.path.join(balanced_dir, "balance_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("聚类平衡报告\n")
            f.write("=" * 30 + "\n")
            f.write(f"平衡后的聚类数量: {len(balanced_df)}\n")
            f.write(f"做多开仓信号: {balanced_df['long_open'].sum()}\n")
            f.write(f"做多平仓信号: {balanced_df['long_close'].sum()}\n")
            f.write(f"做空开仓信号: {balanced_df['short_open'].sum()}\n")
            f.write(f"做空平仓信号: {balanced_df['short_close'].sum()}\n")
        
        print(f"平衡报告已保存到: {report_file}")
        
        return output_file
        
    except Exception as e:
        print(f"保存平衡数据集时出错: {e}")
        return None

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 彻底的聚类平衡工具")
    print("=" * 60)
    
    # 1. 分析聚类信号
    original_df, signal_clusters = analyze_cluster_signals()
    
    if original_df is None or signal_clusters is None:
        print("无法分析聚类信号，退出程序")
        return
    
    # 2. 平衡信号聚类
    balanced_clusters = balance_signal_clusters(signal_clusters, target_count=500)
    
    # 3. 创建最终平衡数据集
    balanced_df = create_final_balanced_dataset(original_df, balanced_clusters)
    
    # 4. 保存平衡数据集
    save_balanced_dataset(balanced_df)
    
    print("\n" + "=" * 60)
    print("彻底的聚类平衡处理完成!")
    print("\n建议下一步操作:")
    print("1. 使用平衡后的数据重新训练模型")
    print("2. 重新运行预测程序验证效果")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    random.seed(42)
    
    main()