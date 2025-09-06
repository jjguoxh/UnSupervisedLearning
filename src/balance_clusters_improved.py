# -*- coding: utf-8 -*-
"""
改进的聚类平衡工具
解决聚类中信号分布不均的问题
"""

import pandas as pd
import numpy as np
import os
import random
from collections import Counter

# ========= 配置参数 =========
# 使用相对于脚本位置的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERNS_DIR = os.path.join(CURRENT_DIR, "..", "patterns/")  # 模式数据目录

def balance_clusters_improved():
    """
    改进的聚类平衡处理
    """
    print("=== 改进的聚类平衡处理 ===")
    
    # 读取聚类分析结果
    cluster_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
    if not os.path.exists(cluster_file):
        print(f"未找到聚类分析文件: {cluster_file}")
        return
    
    try:
        df = pd.read_csv(cluster_file)
        print(f"找到 {len(df)} 个聚类")
        
        # 分析原始信号分布
        total_long_open = df['long_open'].sum()
        total_long_close = df['long_close'].sum()
        total_short_open = df['short_open'].sum()
        total_short_close = df['short_close'].sum()
        
        total_signals = total_long_open + total_long_close + total_short_open + total_short_close
        
        print(f"\n原始信号分布:")
        print(f"  做多开仓: {total_long_open} ({total_long_open/total_signals*100:.2f}%)")
        print(f"  做多平仓: {total_long_close} ({total_long_close/total_signals*100:.2f}%)")
        print(f"  做空开仓: {total_short_open} ({total_short_open/total_signals*100:.2f}%)")
        print(f"  做空平仓: {total_short_close} ({total_short_close/total_signals*100:.2f}%)")
        
        # 确定最大信号类型的数量作为基准
        max_signal_count = max(total_long_open, total_long_close, total_short_open, total_short_close)
        target_count = int(max_signal_count * 0.8)  # 设置为目标数量为最大信号类型的80%
        
        print(f"\n目标信号数量: {target_count}")
        
        # 对做空开仓聚类进行下采样（因为数量过多）
        short_open_clusters = df[df['short_open'] > 0]
        if len(short_open_clusters) > 0:
            # 计算需要采样的聚类数量
            total_short_open_signals = short_open_clusters['short_open'].sum()
            sample_ratio = min(1.0, target_count / total_short_open_signals)
            
            # 按信号数量进行加权采样
            sampled_short_open = short_open_clusters.sample(
                frac=sample_ratio, 
                weights=short_open_clusters['short_open'],
                random_state=42
            )
            
            print(f"\n对做空开仓聚类进行加权采样:")
            print(f"  原始信号数: {total_short_open_signals}")
            print(f"  采样后信号数: {sampled_short_open['short_open'].sum()}")
        else:
            sampled_short_open = short_open_clusters
            
        # 对做多开仓聚类进行适当处理
        long_open_clusters = df[df['long_open'] > 0]
        if len(long_open_clusters) > 0:
            total_long_open_signals = long_open_clusters['long_open'].sum()
            # 如果做多开仓信号太少，可以考虑上采样（但这里我们只做下采样）
            sampled_long_open = long_open_clusters
            print(f"\n做多开仓聚类:")
            print(f"  原始信号数: {total_long_open_signals}")
        else:
            sampled_long_open = long_open_clusters
            
        # 其他信号类型保持不变
        long_close_clusters = df[df['long_close'] > 0]
        short_close_clusters = df[df['short_close'] > 0]
        
        print(f"\n其他信号类型:")
        print(f"  做多平仓聚类数: {len(long_close_clusters)}, 信号数: {long_close_clusters['long_close'].sum()}")
        print(f"  做空平仓聚类数: {len(short_close_clusters)}, 信号数: {short_close_clusters['short_close'].sum()}")
        
        # 合并平衡后的聚类
        balanced_dfs = [sampled_long_open, long_close_clusters, sampled_short_open, short_close_clusters]
        balanced_dfs = [df for df in balanced_dfs if len(df) > 0]  # 过滤空的DataFrame
        
        if balanced_dfs:
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            # 去重（一个聚类可能包含多种信号）
            balanced_df = balanced_df.drop_duplicates(subset=['cluster_id'])
        else:
            balanced_df = pd.DataFrame()
            
        # 保存平衡后的聚类分析结果
        output_file = os.path.join(PATTERNS_DIR, "cluster_analysis_balanced.csv")
        if len(balanced_df) > 0:
            balanced_df.to_csv(output_file, index=False)
            print(f"\n平衡后的聚类分析结果已保存到: {output_file}")
        else:
            print("\n没有生成平衡后的聚类数据")
            return None
        
        # 验证平衡后的信号分布
        balanced_long_open = balanced_df['long_open'].sum()
        balanced_long_close = balanced_df['long_close'].sum()
        balanced_short_open = balanced_df['short_open'].sum()
        balanced_short_close = balanced_df['short_close'].sum()
        
        balanced_total = balanced_long_open + balanced_long_close + balanced_short_open + balanced_short_close
        
        print(f"\n平衡后的信号分布:")
        print(f"  做多开仓: {balanced_long_open} ({balanced_long_open/balanced_total*100:.2f}%)")
        print(f"  做多平仓: {balanced_long_close} ({balanced_long_close/balanced_total*100:.2f}%)")
        print(f"  做空开仓: {balanced_short_open} ({balanced_short_open/balanced_total*100:.2f}%)")
        print(f"  做空平仓: {balanced_short_close} ({balanced_short_close/balanced_total*100:.2f}%)")
        
        return balanced_df
        
    except Exception as e:
        print(f"处理聚类平衡时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_balanced_model_data(balanced_df):
    """
    基于平衡后的聚类创建模型训练数据
    """
    print("\n=== 创建平衡的模型训练数据 ===")
    
    if balanced_df is None or len(balanced_df) == 0:
        print("没有平衡后的聚类数据")
        return
    
    try:
        # 创建一个目录来存储平衡后的模型数据
        balanced_model_dir = os.path.join(PATTERNS_DIR, "balanced_model")
        os.makedirs(balanced_model_dir, exist_ok=True)
        
        # 保存平衡后的聚类数据
        balanced_clusters_file = os.path.join(balanced_model_dir, "balanced_clusters.csv")
        balanced_df.to_csv(balanced_clusters_file, index=False)
        
        print(f"平衡后的模型数据已保存到: {balanced_clusters_file}")
        print("现在可以使用这些平衡后的数据重新训练模型")
        
    except Exception as e:
        print(f"创建平衡模型数据时出错: {e}")

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 改进的聚类平衡工具")
    print("=" * 60)
    
    # 执行聚类平衡
    balanced_df = balance_clusters_improved()
    
    # 创建平衡的模型数据
    create_balanced_model_data(balanced_df)
    
    print("\n" + "=" * 60)
    print("改进的聚类平衡处理完成!")
    print("\n建议下一步操作:")
    print("1. 使用平衡后的数据重新训练模型")
    print("2. 重新运行预测程序验证效果")

if __name__ == "__main__":
    main()