# -*- coding: utf-8 -*-
"""
最终的聚类平衡工具
通过严格的平衡策略解决信号分布不均问题
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

def strict_balance_clusters():
    """
    严格平衡聚类中的信号分布
    """
    print("=== 严格聚类平衡处理 ===")
    
    # 读取聚类分析结果
    cluster_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
    if not os.path.exists(cluster_file):
        print(f"未找到聚类分析文件: {cluster_file}")
        return None
    
    try:
        df = pd.read_csv(cluster_file)
        print(f"找到 {len(df)} 个聚类")
        
        # 确定每种信号类型的最小聚类数作为目标
        signal_counts = {
            'long_open': df['long_open'].sum(),
            'long_close': df['long_close'].sum(),
            'short_open': df['short_open'].sum(),
            'short_close': df['short_close'].sum()
        }
        
        # 找到最小的信号数量作为目标
        target_signal_count = min(signal_counts.values())
        # 但设置一个合理的最小值，避免过少的样本
        target_signal_count = max(target_signal_count, 50)
        
        print(f"\n信号分布:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count} 个信号")
        
        print(f"\n目标信号数量: {target_signal_count}")
        
        # 对每种信号类型进行严格采样
        balanced_data = {}
        
        for signal_type in ['long_open', 'long_close', 'short_open', 'short_close']:
            signal_df = df[df[signal_type] > 0].copy()
            
            if len(signal_df) == 0:
                print(f"  {signal_type}: 无数据")
                balanced_data[signal_type] = signal_df
                continue
                
            current_signal_count = signal_df[signal_type].sum()
            print(f"\n处理 {signal_type}:")
            print(f"  当前信号数: {current_signal_count}")
            
            if current_signal_count <= target_signal_count:
                # 信号数已经很少，保持原样
                balanced_data[signal_type] = signal_df
                print(f"  保持原样: {len(signal_df)} 个聚类")
            else:
                # 需要下采样
                # 按信号数量进行加权采样
                clusters_with_signals = []
                for _, row in signal_df.iterrows():
                    # 为每个聚类添加与其信号数量成比例的条目
                    for _ in range(row[signal_type]):
                        clusters_with_signals.append(row['cluster_id'])
                
                # 随机采样目标数量的信号
                sampled_cluster_ids = random.sample(clusters_with_signals, target_signal_count)
                
                # 统计每个聚类被采样的次数
                cluster_counts = defaultdict(int)
                for cluster_id in sampled_cluster_ids:
                    cluster_counts[cluster_id] += 1
                
                # 创建新的DataFrame，只包含被采样的聚类
                selected_clusters = []
                for cluster_id, count in cluster_counts.items():
                    row = signal_df[signal_df['cluster_id'] == cluster_id].iloc[0].copy()
                    # 调整信号数量为采样后的数量
                    row[signal_type] = count
                    selected_clusters.append(row)
                
                balanced_signal_df = pd.DataFrame(selected_clusters)
                balanced_data[signal_type] = balanced_signal_df
                
                print(f"  采样后: {len(balanced_signal_df)} 个聚类, {balanced_signal_df[signal_type].sum()} 个信号")
        
        # 合并所有平衡后的数据
        all_balanced_dfs = list(balanced_data.values())
        all_balanced_dfs = [df for df in all_balanced_dfs if len(df) > 0]
        
        if all_balanced_dfs:
            # 合并所有DataFrame
            final_balanced_df = pd.concat(all_balanced_dfs, ignore_index=True)
            # 去重（一个聚类可能包含多种信号）
            final_balanced_df = final_balanced_df.drop_duplicates(subset=['cluster_id'])
        else:
            final_balanced_df = pd.DataFrame()
            
        print(f"\n最终平衡数据:")
        print(f"  聚类数量: {len(final_balanced_df)}")
        if len(final_balanced_df) > 0:
            total_long_open = final_balanced_df['long_open'].sum()
            total_long_close = final_balanced_df['long_close'].sum()
            total_short_open = final_balanced_df['short_open'].sum()
            total_short_close = final_balanced_df['short_close'].sum()
            
            total_signals = total_long_open + total_long_close + total_short_open + total_short_close
            
            print(f"  总信号数: {total_signals}")
            print(f"  做多开仓: {total_long_open} ({total_long_open/total_signals*100:.2f}%)")
            print(f"  做多平仓: {total_long_close} ({total_long_close/total_signals*100:.2f}%)")
            print(f"  做空开仓: {total_short_open} ({total_short_open/total_signals*100:.2f}%)")
            print(f"  做空平仓: {total_short_close} ({total_short_close/total_signals*100:.2f}%)")
        
        return final_balanced_df
        
    except Exception as e:
        print(f"严格平衡聚类时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_strictly_balanced_dataset(balanced_df):
    """
    保存严格平衡后的数据集
    """
    print("\n=== 保存严格平衡数据集 ===")
    
    if balanced_df is None or len(balanced_df) == 0:
        print("没有平衡后的数据可保存")
        return None
    
    try:
        # 创建严格平衡数据目录
        strict_balanced_dir = os.path.join(PATTERNS_DIR, "strict_balanced")
        os.makedirs(strict_balanced_dir, exist_ok=True)
        
        # 保存平衡后的聚类分析文件
        output_file = os.path.join(strict_balanced_dir, "cluster_analysis_strict_balanced.csv")
        balanced_df.to_csv(output_file, index=False)
        print(f"严格平衡后的聚类分析结果已保存到: {output_file}")
        
        # 创建详细报告
        report_file = os.path.join(strict_balanced_dir, "strict_balance_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("严格聚类平衡报告\n")
            f.write("=" * 30 + "\n")
            f.write(f"聚类数量: {len(balanced_df)}\n")
            f.write(f"做多开仓信号: {balanced_df['long_open'].sum()}\n")
            f.write(f"做多平仓信号: {balanced_df['long_close'].sum()}\n")
            f.write(f"做空开仓信号: {balanced_df['short_open'].sum()}\n")
            f.write(f"做空平仓信号: {balanced_df['short_close'].sum()}\n")
            
            total_signals = (balanced_df['long_open'].sum() + 
                           balanced_df['long_close'].sum() + 
                           balanced_df['short_open'].sum() + 
                           balanced_df['short_close'].sum())
            
            if total_signals > 0:
                f.write(f"\n信号分布:\n")
                f.write(f"做多开仓: {balanced_df['long_open'].sum()/total_signals*100:.2f}%\n")
                f.write(f"做多平仓: {balanced_df['long_close'].sum()/total_signals*100:.2f}%\n")
                f.write(f"做空开仓: {balanced_df['short_open'].sum()/total_signals*100:.2f}%\n")
                f.write(f"做空平仓: {balanced_df['short_close'].sum()/total_signals*100:.2f}%\n")
        
        print(f"详细报告已保存到: {report_file}")
        
        return output_file
        
    except Exception as e:
        print(f"保存严格平衡数据集时出错: {e}")
        return None

def generate_recommendations(balanced_df):
    """
    生成使用建议
    """
    print("\n=== 使用建议 ===")
    
    if balanced_df is None or len(balanced_df) == 0:
        print("没有平衡后的数据，无法生成建议")
        return
    
    print("1. 使用严格平衡后的数据重新训练模型:")
    print("   - 将平衡后的聚类数据用于模型训练")
    print("   - 确保模型能够学习到平衡的信号分布")
    
    print("\n2. 验证模型效果:")
    print("   - 重新运行预测程序")
    print("   - 检查预测结果的分布是否更加均衡")
    print("   - 验证预测置信度是否有所提高")
    
    print("\n3. 进一步优化:")
    print("   - 如果效果仍不理想，可调整目标信号数量")
    print("   - 考虑增加更多训练数据")
    print("   - 调整模型超参数")

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 严格聚类平衡工具")
    print("=" * 60)
    
    # 执行严格聚类平衡
    balanced_df = strict_balance_clusters()
    
    # 保存平衡数据集
    save_strictly_balanced_dataset(balanced_df)
    
    # 生成使用建议
    generate_recommendations(balanced_df)
    
    print("\n" + "=" * 60)
    print("严格聚类平衡处理完成!")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    main()