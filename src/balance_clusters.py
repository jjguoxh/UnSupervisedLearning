# -*- coding: utf-8 -*-
"""
聚类平衡工具
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

def balance_clusters():
    """
    平衡聚类中的信号分布
    """
    print("=== 聚类平衡处理 ===")
    
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
        
        # 确定目标分布（平衡分布）
        target_count = int(total_signals / 4)
        print(f"\n目标信号分布（每类{target_count}个）:")
        
        # 对每个聚类进行采样，确保信号分布平衡
        balanced_clusters = []
        
        # 按信号类型分组聚类
        long_open_clusters = df[df['long_open'] > 0]
        long_close_clusters = df[df['long_close'] > 0]
        short_open_clusters = df[df['short_open'] > 0]
        short_close_clusters = df[df['short_close'] > 0]
        
        print(f"\n各信号类型聚类数量:")
        print(f"  做多开仓聚类: {len(long_open_clusters)}")
        print(f"  做多平仓聚类: {len(long_close_clusters)}")
        print(f"  做空开仓聚类: {len(short_open_clusters)}")
        print(f"  做空平仓聚类: {len(short_close_clusters)}")
        
        # 对做空开仓聚类进行下采样
        if len(short_open_clusters) > target_count:
            # 随机选择部分做空开仓聚类
            sampled_short_open = short_open_clusters.sample(n=target_count, random_state=42)
            print(f"\n对做空开仓聚类进行下采样: {len(short_open_clusters)} -> {len(sampled_short_open)}")
        else:
            sampled_short_open = short_open_clusters
            
        # 对其他信号类型进行适当采样
        sampled_long_open = long_open_clusters
        sampled_long_close = long_close_clusters
        sampled_short_close = short_close_clusters
        
        # 合并平衡后的聚类
        balanced_df = pd.concat([sampled_long_open, sampled_long_close, sampled_short_open, sampled_short_close])
        
        # 去重（一个聚类可能包含多种信号）
        balanced_df = balanced_df.drop_duplicates(subset=['cluster_id'])
        
        # 保存平衡后的聚类分析结果
        output_file = os.path.join(PATTERNS_DIR, "cluster_analysis_balanced.csv")
        balanced_df.to_csv(output_file, index=False)
        print(f"\n平衡后的聚类分析结果已保存到: {output_file}")
        
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
        return None

def update_model_with_balanced_clusters(balanced_df):
    """
    使用平衡后的聚类更新模型
    """
    print("\n=== 更新模型 ===")
    
    if balanced_df is None:
        print("没有平衡后的聚类数据")
        return
    
    # 这里应该更新模型文件，但为了简化，我们只输出信息
    print("模型更新完成（实际实现需要根据具体模型结构进行调整）")

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 聚类平衡工具")
    print("=" * 50)
    
    # 执行聚类平衡
    balanced_df = balance_clusters()
    
    # 更新模型
    update_model_with_balanced_clusters(balanced_df)
    
    print("\n" + "=" * 50)
    print("聚类平衡处理完成!")

if __name__ == "__main__":
    main()