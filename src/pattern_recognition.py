# -*- coding: utf-8 -*-
"""
交易模式识别模型
从历史标签数据中学习交易模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
OUTPUT_DIR = "../patterns/"  # 模式输出目录
WINDOW_SIZE = 30  # 时间窗口大小（秒）
PATTERN_LENGTH = 10  # 模式长度

def load_label_data(file_path):
    """
    加载标签数据
    """
    df = pd.read_csv(file_path)
    return df

def extract_pattern_features(df, window_size=30):
    """
    从数据中提取模式特征
    """
    features = []
    
    # 使用滑动窗口提取特征
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        
        # 提取窗口特征
        window_features = {
            'start_idx': i,
            'end_idx': i + window_size - 1,
            'x_start': window['x'].iloc[0],
            'x_end': window['x'].iloc[-1],
            'a_mean': window['a'].mean(),
            'a_std': window['a'].std(),
            'b_mean': window['b'].mean(),
            'b_std': window['b'].std(),
            'c_mean': window['c'].mean(),
            'c_std': window['c'].std(),
            'd_mean': window['d'].mean(),
            'd_std': window['d'].std(),
            'index_value_start': window['index_value'].iloc[0],
            'index_value_end': window['index_value'].iloc[-1],
            'index_value_min': window['index_value'].min(),
            'index_value_max': window['index_value'].max(),
            'index_value_mean': window['index_value'].mean(),
            'index_value_std': window['index_value'].std(),
        }
        
        features.append(window_features)
    
    return pd.DataFrame(features)

def identify_trading_signals(df):
    """
    识别交易信号
    标签定义：
    0: 无操作
    1: 做空开仓
    2: 做空平仓
    3: 做多开仓
    4: 做多平仓
    """
    signals = []
    
    for i in range(len(df)):
        label = df['label'].iloc[i]
        if label != 0:  # 有交易信号
            signal = {
                'index': i,
                'x': df['x'].iloc[i],
                'label': label,
                'a': df['a'].iloc[i],
                'b': df['b'].iloc[i],
                'c': df['c'].iloc[i],
                'd': df['d'].iloc[i],
                'index_value': df['index_value'].iloc[i]
            }
            signals.append(signal)
    
    return pd.DataFrame(signals)

def extract_signal_patterns(df, signals, pattern_length=10):
    """
    提取信号前后的模式
    """
    patterns = []
    
    for _, signal in signals.iterrows():
        signal_idx = signal['index']
        
        # 确保有足够的历史数据
        if signal_idx >= pattern_length:
            # 提取信号前pattern_length个数据点
            start_idx = signal_idx - pattern_length
            end_idx = signal_idx
            pattern_data = df.iloc[start_idx:end_idx]
            
            pattern = {
                'signal_index': signal_idx,
                'signal_x': signal['x'],
                'signal_label': signal['label'],
                'pattern_start_idx': start_idx,
                'pattern_end_idx': end_idx,
                'a_values': pattern_data['a'].tolist(),
                'b_values': pattern_data['b'].tolist(),
                'c_values': pattern_data['c'].tolist(),
                'd_values': pattern_data['d'].tolist(),
                'index_value_values': pattern_data['index_value'].tolist(),
                'x_values': pattern_data['x'].tolist()
            }
            patterns.append(pattern)
    
    return patterns

def calculate_pattern_similarity(pattern1, pattern2):
    """
    计算两个模式的相似性
    """
    # 计算index_value的相似性（使用相关系数）
    val1 = np.array(pattern1['index_value_values'])
    val2 = np.array(pattern2['index_value_values'])
    
    # 标准化
    scaler = StandardScaler()
    val1_scaled = scaler.fit_transform(val1.reshape(-1, 1)).flatten()
    val2_scaled = scaler.transform(val2.reshape(-1, 1)).flatten()
    
    # 计算相关系数
    correlation = np.corrcoef(val1_scaled, val2_scaled)[0, 1]
    return correlation if not np.isnan(correlation) else 0

def cluster_patterns(patterns, similarity_threshold=0.8):
    """
    基于相似性对模式进行聚类
    """
    clusters = []
    visited = set()
    
    for i, pattern1 in enumerate(patterns):
        if i in visited:
            continue
            
        # 创建新聚类
        current_cluster = [i]
        visited.add(i)
        
        # 查找相似的模式
        for j, pattern2 in enumerate(patterns):
            if j in visited or i == j:
                continue
                
            similarity = calculate_pattern_similarity(pattern1, pattern2)
            if similarity >= similarity_threshold:
                current_cluster.append(j)
                visited.add(j)
        
        clusters.append(current_cluster)
    
    return clusters

def analyze_cluster_profitability(patterns, clusters):
    """
    分析每个聚类的盈利能力
    """
    cluster_analysis = []
    
    for cluster_idx, cluster in enumerate(clusters):
        # 收集该聚类中的所有信号
        cluster_signals = []
        for pattern_idx in cluster:
            pattern = patterns[pattern_idx]
            cluster_signals.append(pattern['signal_label'])
        
        # 统计信号类型
        signal_counts = Counter(cluster_signals)
        
        # 计算盈利能力指标
        long_open = signal_counts[3]  # 做多开仓
        long_close = signal_counts[4]  # 做多平仓
        short_open = signal_counts[1]  # 做空开仓
        short_close = signal_counts[2]  # 做空平仓
        
        # 计算配对信号数
        long_pairs = min(long_open, long_close)
        short_pairs = min(short_open, short_close)
        total_pairs = long_pairs + short_pairs
        
        # 计算信号密度
        total_signals = len(cluster_signals)
        signal_density = total_pairs / len(cluster) if len(cluster) > 0 else 0
        
        analysis = {
            'cluster_id': cluster_idx,
            'cluster_size': len(cluster),
            'signal_counts': signal_counts,
            'long_open': long_open,
            'long_close': long_close,
            'short_open': short_open,
            'short_close': short_close,
            'long_pairs': long_pairs,
            'short_pairs': short_pairs,
            'total_pairs': total_pairs,
            'signal_density': signal_density
        }
        
        cluster_analysis.append(analysis)
    
    return cluster_analysis

def visualize_patterns(patterns, clusters, cluster_analysis, top_k=5):
    """
    可视化最具盈利潜力的模式
    """
    # 按信号密度排序
    sorted_clusters = sorted(cluster_analysis, 
                           key=lambda x: x['signal_density'], 
                           reverse=True)
    
    top_clusters = sorted_clusters[:top_k]
    
    fig, axes = plt.subplots(top_k, 1, figsize=(15, 4*top_k))
    if top_k == 1:
        axes = [axes]
    
    for idx, cluster_info in enumerate(top_clusters):
        cluster_id = cluster_info['cluster_id']
        cluster = clusters[cluster_id]
        
        ax = axes[idx]
        
        # 绘制该聚类中的所有模式
        for pattern_idx in cluster[:10]:  # 最多绘制10个样本
            pattern = patterns[pattern_idx]
            x_vals = range(len(pattern['index_value_values']))
            y_vals = pattern['index_value_values']
            ax.plot(x_vals, y_vals, alpha=0.7, color='blue')
        
        ax.set_title(f'Cluster {cluster_id} (Signal Density: {cluster_info["signal_density"]:.3f}, '
                    f'Size: {cluster_info["cluster_size"]})')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Index Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_profitable_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_patterns(patterns, clusters, cluster_analysis):
    """
    保存模式到文件
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存聚类分析结果
    analysis_df = pd.DataFrame(cluster_analysis)
    analysis_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_analysis.csv'), index=False)
    
    # 保存每个聚类的详细信息
    for cluster_info in cluster_analysis:
        cluster_id = cluster_info['cluster_id']
        cluster = clusters[cluster_id]
        
        # 创建聚类目录
        cluster_dir = os.path.join(OUTPUT_DIR, f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 保存该聚类的代表性模式
        cluster_patterns = [patterns[i] for i in cluster[:5]]  # 保存前5个
        
        for i, pattern in enumerate(cluster_patterns):
            pattern_df = pd.DataFrame({
                'x': pattern['x_values'],
                'a': pattern['a_values'],
                'b': pattern['b_values'],
                'c': pattern['c_values'],
                'd': pattern['d_values'],
                'index_value': pattern['index_value_values']
            })
            pattern_df.to_csv(os.path.join(cluster_dir, f'pattern_{i}.csv'), index=False)
        
        # 保存聚类统计信息
        cluster_info_df = pd.DataFrame([cluster_info])
        cluster_info_df.to_csv(os.path.join(cluster_dir, 'cluster_info.csv'), index=False)

def process_single_file(file_path):
    """
    处理单个标签文件
    """
    print(f"Processing {os.path.basename(file_path)}...")
    
    # 加载数据
    df = load_label_data(file_path)
    
    # 提取特征
    features = extract_pattern_features(df, WINDOW_SIZE)
    
    # 识别交易信号
    signals = identify_trading_signals(df)
    print(f"  Found {len(signals)} trading signals")
    
    # 提取信号模式
    patterns = extract_signal_patterns(df, signals, PATTERN_LENGTH)
    print(f"  Extracted {len(patterns)} patterns")
    
    if len(patterns) == 0:
        return None, None, None
    
    # 对模式进行聚类
    clusters = cluster_patterns(patterns, similarity_threshold=0.7)
    print(f"  Created {len(clusters)} clusters")
    
    # 分析聚类盈利能力
    cluster_analysis = analyze_cluster_profitability(patterns, clusters)
    
    return patterns, clusters, cluster_analysis

def main():
    """
    主函数
    """
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    print(f"Found {len(label_files)} label files")
    
    # 存储所有模式
    all_patterns = []
    all_clusters = []
    all_cluster_analysis = []
    
    # 处理前10个文件作为示例
    for i, file_path in enumerate(label_files[:10]):
        try:
            patterns, clusters, cluster_analysis = process_single_file(file_path)
            if patterns is not None:
                all_patterns.extend(patterns)
                all_clusters.extend(clusters)
                all_cluster_analysis.extend(cluster_analysis)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_patterns:
        print("No patterns found!")
        return
    
    print(f"\nTotal patterns: {len(all_patterns)}")
    print(f"Total clusters: {len(all_clusters)}")
    
    # 合并所有聚类分析结果
    # 重新计算总体盈利能力
    total_analysis = []
    for cluster_idx in range(len(all_clusters)):
        # 收集该聚类中的所有信号
        cluster_signals = []
        cluster = all_clusters[cluster_idx]
        for pattern_idx in cluster:
            if pattern_idx < len(all_patterns):
                pattern = all_patterns[pattern_idx]
                cluster_signals.append(pattern['signal_label'])
        
        if not cluster_signals:
            continue
            
        # 统计信号类型
        signal_counts = Counter(cluster_signals)
        
        # 计算盈利能力指标
        long_open = signal_counts[3]  # 做多开仓
        long_close = signal_counts[4]  # 做多平仓
        short_open = signal_counts[1]  # 做空开仓
        short_close = signal_counts[2]  # 做空平仓
        
        # 计算配对信号数
        long_pairs = min(long_open, long_close)
        short_pairs = min(short_open, short_close)
        total_pairs = long_pairs + short_pairs
        
        # 计算信号密度
        total_signals = len(cluster_signals)
        signal_density = total_pairs / len(cluster) if len(cluster) > 0 else 0
        
        analysis = {
            'cluster_id': cluster_idx,
            'cluster_size': len(cluster),
            'signal_counts': dict(signal_counts),
            'long_open': long_open,
            'long_close': long_close,
            'short_open': short_open,
            'short_close': short_close,
            'long_pairs': long_pairs,
            'short_pairs': short_pairs,
            'total_pairs': total_pairs,
            'signal_density': signal_density
        }
        
        total_analysis.append(analysis)
    
    # 可视化结果
    visualize_patterns(all_patterns, all_clusters, total_analysis, top_k=5)
    
    # 保存结果
    save_patterns(all_patterns, all_clusters, total_analysis)
    
    # 打印最佳聚类
    sorted_analysis = sorted(total_analysis, 
                           key=lambda x: x['signal_density'], 
                           reverse=True)
    
    print("\nTop profitable clusters:")
    for i, analysis in enumerate(sorted_analysis[:5]):
        print(f"  {i+1}. Cluster {analysis['cluster_id']}: "
              f"Signal density = {analysis['signal_density']:.3f}, "
              f"Size = {analysis['cluster_size']}")
        print(f"     Long pairs: {analysis['long_pairs']}, "
              f"Short pairs: {analysis['short_pairs']}")
    
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()