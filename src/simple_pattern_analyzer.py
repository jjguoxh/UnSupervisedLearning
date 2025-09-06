# -*- coding: utf-8 -*-
"""
简单交易模式分析器
直接分析标签数据中的交易模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
OUTPUT_DIR = "../simple_patterns/"  # 简单模式输出目录
WINDOW_SIZE = 30  # 时间窗口大小

def load_data(file_path):
    """
    加载单个CSV文件数据
    """
    df = pd.read_csv(file_path)
    return df

def extract_signal_context(df, signal_idx, window_size=30):
    """
    提取信号前后的时间序列上下文
    """
    start_idx = max(0, signal_idx - window_size)
    end_idx = min(len(df), signal_idx + window_size)
    
    context = df.iloc[start_idx:end_idx][['x', 'a', 'b', 'c', 'd', 'index_value']].copy()
    context['distance_from_signal'] = range(start_idx - signal_idx, end_idx - signal_idx)
    context['is_signal'] = context.index == signal_idx
    
    return context

def analyze_signal_patterns(label_files, window_size=30):
    """
    分析所有文件中的信号模式
    """
    all_patterns = []
    
    for file_path in label_files[:10]:  # 只分析前10个文件
        print(f"Analyzing {os.path.basename(file_path)}...")
        df = load_data(file_path)
        
        # 找到所有信号点
        signals = df[df['label'] != 0]
        
        for idx, signal_row in signals.iterrows():
            # 提取信号上下文
            context = extract_signal_context(df, idx, window_size)
            
            pattern = {
                'file': os.path.basename(file_path),
                'signal_index': idx,
                'signal_label': signal_row['label'],
                'context': context,
                'index_value_at_signal': signal_row['index_value'],
                'a_at_signal': signal_row['a'],
                'b_at_signal': signal_row['b'],
                'c_at_signal': signal_row['c'],
                'd_at_signal': signal_row['d']
            }
            
            all_patterns.append(pattern)
    
    return all_patterns

def categorize_patterns(patterns):
    """
    根据信号类型对模式进行分类
    """
    categorized = {
        1: [],  # 做多开仓（包括开仓点和持仓状态）
        2: [],  # 做多平仓
        3: [],  # 做空开仓（包括开仓点和持仓状态）
        4: []   # 做空平仓
    }
    
    for pattern in patterns:
        label = pattern['signal_label']
        if label in categorized:
            categorized[label].append(pattern)
    
    return categorized

def calculate_pattern_statistics(categorized_patterns):
    """
    计算各类模式的统计信息
    """
    stats = {}
    
    for label, patterns in categorized_patterns.items():
        if len(patterns) == 0:
            continue
            
        # 计算信号发生前后的价格变化
        price_changes = []
        for pattern in patterns:
            context = pattern['context']
            signal_time = context[context['is_signal']].index[0]
            
            # 计算信号后10个时间点的价格变化
            future_indices = list(range(signal_time+1, min(signal_time+11, len(context))))
            if future_indices:
                current_price = context.loc[signal_time, 'index_value']
                future_prices = context.loc[future_indices, 'index_value'].values
                changes = (future_prices - current_price) / current_price
                price_changes.extend(changes)
        
        stats[label] = {
            'count': len(patterns),
            'avg_price_change': np.mean(price_changes) if price_changes else 0,
            'price_change_std': np.std(price_changes) if price_changes else 0,
            'win_rate': np.mean(np.array(price_changes) > 0) if price_changes else 0
        }
    
    return stats

def visualize_patterns(categorized_patterns, top_k=5):
    """
    可视化各类模式
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    label_names = {
        1: 'Long Open/Holding',  # 做多开仓（包括开仓点和持仓状态）
        2: 'Long Close',         # 做多平仓
        3: 'Short Open/Holding', # 做空开仓（包括开仓点和持仓状态）
        4: 'Short Close'         # 做空平仓
    }
    
    for label, patterns in categorized_patterns.items():
        if len(patterns) == 0:
            continue
            
        # 创建图表
        fig, axes = plt.subplots(min(top_k, len(patterns)), 1, figsize=(15, 4*min(top_k, len(patterns))))
        if min(top_k, len(patterns)) == 1:
            axes = [axes]
        
        # 绘制前几个模式
        for i, pattern in enumerate(patterns[:top_k]):
            ax = axes[i] if len(patterns) > 1 else axes[0]
            context = pattern['context']
            
            # 绘制价格序列
            ax.plot(context['distance_from_signal'], context['index_value'], 'b-', linewidth=2)
            
            # 标记信号点
            signal_point = context[context['is_signal']]
            ax.scatter(signal_point['distance_from_signal'], signal_point['index_value'], 
                      color='red', s=100, zorder=5, label=f'Signal (Label {label})')
            
            ax.set_title(f'{label_names.get(label, f"Label {label}")} - {pattern["file"]} (Index: {pattern["signal_index"]})')
            ax.set_xlabel('Time (relative to signal)')
            ax.set_ylabel('Index Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'patterns_label_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_pattern_analysis(stats):
    """
    保存模式分析结果
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 创建统计结果DataFrame
    results = []
    label_names = {
        1: 'Long Open/Holding',  # 做多开仓（包括开仓点和持仓状态）
        2: 'Long Close',         # 做多平仓
        3: 'Short Open/Holding', # 做空开仓（包括开仓点和持仓状态）
        4: 'Short Close'         # 做空平仓
    }
    
    for label, stat in stats.items():
        results.append({
            'Signal_Type': label_names.get(label, f'Label {label}'),
            'Count': stat['count'],
            'Avg_Price_Change': f"{stat['avg_price_change']:.4f}",
            'Price_Change_Std': f"{stat['price_change_std']:.4f}",
            'Win_Rate': f"{stat['win_rate']:.2%}"
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'pattern_analysis.csv'), index=False)
    
    # 打印结果
    print("\nPattern Analysis Results:")
    print("=" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['Signal_Type']}:")
        print(f"  Count: {row['Count']}")
        print(f"  Avg Price Change: {row['Avg_Price_Change']}")
        print(f"  Win Rate: {row['Win_Rate']}")
        print()

def main():
    """
    主函数
    """
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    print(f"Found {len(label_files)} label files")
    
    # 分析信号模式
    patterns = analyze_signal_patterns(label_files, WINDOW_SIZE)
    print(f"\nFound {len(patterns)} signal patterns")
    
    # 对模式进行分类
    categorized_patterns = categorize_patterns(patterns)
    
    # 计算统计信息
    stats = calculate_pattern_statistics(categorized_patterns)
    
    # 可视化模式
    visualize_patterns(categorized_patterns)
    
    # 保存分析结果
    save_pattern_analysis(stats)
    
    print(f"\nAnalysis completed! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()