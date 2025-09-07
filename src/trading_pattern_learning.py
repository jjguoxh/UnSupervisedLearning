# -*- coding: utf-8 -*-
"""
无监督学习交易模式模型
基于历史交易数据学习能够显著盈利的交易模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "./label/"  # 标签数据目录
OUTPUT_DIR = "./patterns/"  # 模式输出目录
N_CLUSTERS = 10  # 聚类数量
WINDOW_SIZE = 20  # 时间窗口大小
N_COMPONENTS = 5  # PCA降维后的主成分数量

def load_data(file_path):
    """
    加载单个CSV文件数据
    """
    df = pd.read_csv(file_path)
    return df

def extract_features(df):
    """
    从数据中提取特征
    x: 时间轴
    a, b, c, d: 影响因子
    index_value: 股指期货走势
    """
    features = df[['x', 'a', 'b', 'c', 'd', 'index_value']].copy()
    return features

def create_sliding_windows_around_signals(df, window_size=20):
    """
    围绕交易信号创建滑动窗口
    专门针对稀疏信号进行优化
    """
    windows = []
    signal_indices = []
    
    # 找到所有非零标签的索引
    non_zero_labels = df[df['label'] != 0].index.tolist()
    
    # 为每个信号创建窗口
    for idx in non_zero_labels:
        # 确保窗口不会越界
        start_idx = max(0, idx - window_size // 2)
        end_idx = min(len(df), idx + window_size // 2)
        
        # 如果窗口大小不够，跳过
        if end_idx - start_idx < window_size:
            continue
            
        # 提取窗口数据
        window_data = df.iloc[start_idx:end_idx][['x', 'a', 'b', 'c', 'd', 'index_value']].values
        windows.append(window_data)
        signal_indices.append(idx)
    
    return np.array(windows), signal_indices

def create_sliding_windows_features(df, window_size=20):
    """
    从窗口中提取特征
    """
    # 围绕信号创建窗口
    windows, signal_indices = create_sliding_windows_around_signals(df, window_size)
    
    if len(windows) == 0:
        return np.array([]), []
    
    # 提取特征
    features_list = []
    
    for window in windows:
        features = []
        # 对每个特征列计算统计量
        for j in range(window.shape[1]):
            feature_values = window[:, j]
            features.extend([
                np.mean(feature_values),
                np.std(feature_values),
                np.min(feature_values),
                np.max(feature_values),
                np.percentile(feature_values, 25),
                np.percentile(feature_values, 75),
                np.median(feature_values),
                np.max(feature_values) - np.min(feature_values)  # 极差
            ])
        features_list.append(features)
    
    return np.array(features_list), signal_indices

def normalize_features(features):
    """
    标准化特征
    """
    if len(features) == 0:
        return features
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    return features_normalized, scaler

def perform_clustering(features, n_clusters=10):
    """
    执行聚类分析
    """
    if len(features) == 0:
        return np.array([]), None
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels, kmeans

def perform_pca(features, n_components=5):
    """
    执行PCA降维
    """
    if len(features) == 0:
        return features, None
    
    # 确保组件数不超过特征数和样本数
    n_components = min(n_components, features.shape[1], features.shape[0])
    
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca, pca

def analyze_signal_patterns(df, cluster_labels, signal_indices):
    """
    分析信号模式
    """
    if len(cluster_labels) == 0 or len(signal_indices) == 0:
        return {}
    
    # 获取每个信号的标签
    signal_labels = []
    for idx in signal_indices:
        signal_labels.append(df['label'].iloc[idx])
    
    # 统计每个聚类中的标签分布
    cluster_label_dist = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_label_dist:
            cluster_label_dist[cluster_id] = []
        cluster_label_dist[cluster_id].append(signal_labels[i])
    
    return cluster_label_dist

def calculate_cluster_profitability(cluster_label_dist):
    """
    计算每个聚类的盈利能力
    标签定义：
    0: 无操作状态
    1: 做多开仓（包括开仓点和持仓状态）
    2: 做多平仓
    3: 做空开仓（包括开仓点和持仓状态）
    4: 做空平仓
    """
    cluster_profit = {}
    
    for cluster_id, labels in cluster_label_dist.items():
        # 统计各类标签的数量
        label_counts = Counter(labels)
        long_open_count = label_counts[1]  # 做多开仓
        long_close_count = label_counts[2]  # 做多平仓
        short_open_count = label_counts[3]  # 做空开仓
        short_close_count = label_counts[4]  # 做空平仓
        no_action_count = label_counts[0]   # 无操作
        
        # 简单的盈利能力指标：开仓和平仓的匹配程度
        long_signals = min(long_open_count, long_close_count)
        short_signals = min(short_open_count, short_close_count)
        total_signals = long_signals + short_signals
        
        # 计算信号密度
        total_labels = len(labels)
        signal_density = total_signals / total_labels if total_labels > 0 else 0
        
        cluster_profit[cluster_id] = {
            'long_open': long_open_count,
            'long_close': long_close_count,
            'short_open': short_open_count,
            'short_close': short_close_count,
            'no_action': no_action_count,
            'total_signals': total_signals,
            'signal_density': signal_density,
            'total_labels': total_labels
        }
    
    return cluster_profit

def visualize_cluster_patterns(df, windows, cluster_labels, signal_indices, cluster_profit, n_clusters=10, top_k=5):
    """
    可视化聚类模式
    """
    if len(windows) == 0 or len(cluster_labels) == 0:
        print("No windows or clusters to visualize")
        return
    
    # 找出盈利能力最强的几个聚类
    sorted_clusters = sorted(cluster_profit.items(), 
                           key=lambda x: x[1]['signal_density'], 
                           reverse=True)
    top_clusters = sorted_clusters[:top_k]
    
    fig, axes = plt.subplots(top_k, 1, figsize=(15, 4*top_k))
    if top_k == 1:
        axes = [axes]
    
    for idx, (cluster_id, profit_info) in enumerate(top_clusters):
        # 找到属于该聚类的所有窗口
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # 随机选择几个窗口进行可视化
        n_samples = min(5, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        
        ax = axes[idx]
        for i in sample_indices:
            window = windows[i]
            # 绘制index_value特征（第5个特征）
            ax.plot(range(len(window)), window[:, 5], alpha=0.7, color='blue')
        
        ax.set_title(f'Cluster {cluster_id} (Signal Density: {profit_info["signal_density"]:.3f})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Index Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_profitable_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_cluster_patterns(windows, cluster_labels, cluster_profit, signal_indices, n_clusters=10):
    """
    保存聚类模式到文件
    """
    if len(windows) == 0 or len(cluster_labels) == 0:
        print("No windows or clusters to save")
        return
    
    patterns_dir = os.path.join(OUTPUT_DIR, 'cluster_patterns')
    os.makedirs(patterns_dir, exist_ok=True)
    
    # 获取唯一的聚类ID
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # 找到属于该聚类的所有窗口
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
            
        # 保存该聚类的统计信息
        profit_info = cluster_profit.get(cluster_id, {})
        pattern_info = {
            'cluster_id': cluster_id,
            'n_samples': len(cluster_indices),
            'profit_info': profit_info
        }
        
        # 保存前几个样本作为代表
        n_samples = min(5, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        sample_windows = windows[sample_indices]
        
        # 保存到文件
        pattern_file = os.path.join(patterns_dir, f'cluster_{cluster_id}_pattern.npz')
        np.savez(pattern_file, 
                pattern_info=pattern_info,
                sample_windows=sample_windows,
                cluster_indices=cluster_indices)
        
        if 'signal_density' in profit_info:
            print(f"Cluster {cluster_id}: {len(cluster_indices)} samples, "
                  f"Signal density: {profit_info['signal_density']:.3f}")

def process_single_file(label_file):
    """
    处理单个标签文件
    """
    print(f"Processing {os.path.basename(label_file)}...")
    
    # 加载数据
    df = load_data(label_file)
    
    # 围绕信号创建窗口并提取特征
    window_features, signal_indices = create_sliding_windows_features(df, WINDOW_SIZE)
    
    if len(window_features) == 0:
        print("  No signals found in this file")
        return np.array([]), np.array([]), {}, np.array([]), []
    
    print(f"  Found {len(window_features)} signal windows")
    
    # 标准化特征
    features_normalized, scaler = normalize_features(window_features)
    
    # PCA降维
    features_pca, pca = perform_pca(features_normalized, N_COMPONENTS)
    print(f"  PCA reduced to {features_pca.shape[1] if len(features_pca) > 0 else 0} components")
    
    # 聚类分析
    cluster_labels, kmeans = perform_clustering(features_pca, N_CLUSTERS)
    print(f"  Performed clustering with {len(np.unique(cluster_labels)) if len(cluster_labels) > 0 else 0} clusters")
    
    # 分析聚类结果
    cluster_label_dist = analyze_signal_patterns(df, cluster_labels, signal_indices)
    cluster_profit = calculate_cluster_profitability(cluster_label_dist)
    
    # 显示聚类结果
    print("  Cluster profitability:")
    for cluster_id, profit_info in cluster_profit.items():
        print(f"    Cluster {cluster_id}: Signal density = {profit_info['signal_density']:.3f}, "
              f"Total signals = {profit_info['total_signals']}")
    
    # 重新创建窗口用于可视化
    windows, _ = create_sliding_windows_around_signals(df, WINDOW_SIZE)
    
    return windows, cluster_labels, cluster_profit, features_normalized, signal_indices

def main():
    """
    主函数
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patterns_dir = os.path.join(OUTPUT_DIR, 'cluster_patterns')
    os.makedirs(patterns_dir, exist_ok=True)
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    
    print(f"Found {len(label_files)} label files")
    
    # 处理所有文件并收集结果
    all_windows = []
    all_cluster_labels = []
    all_cluster_profits = []
    all_features = []
    all_signal_indices = []
    
    # 处理前10个文件作为示例
    processed_files = 0
    for i, label_file in enumerate(label_files[:10]):
        try:
            windows, cluster_labels, cluster_profit, features, signal_indices = process_single_file(label_file)
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_cluster_labels.append(cluster_labels)
                all_cluster_profits.append(cluster_profit)
                all_features.append(features)
                all_signal_indices.extend(signal_indices)
                processed_files += 1
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    if processed_files == 0:
        print("No files with signals found!")
        return
    
    print(f"\nProcessed {processed_files} files with signals")
    
    # 合并所有窗口数据
    if all_windows:
        combined_windows = np.concatenate(all_windows, axis=0)
        combined_cluster_labels = np.concatenate(all_cluster_labels, axis=0)
        combined_features = np.concatenate(all_features, axis=0)
        
        # 计算总体盈利能力
        total_profit = {}
        total_labels = 0
        total_signals = 0
        
        for cluster_profit_dict in all_cluster_profits:
            for cluster_id, profit_info in cluster_profit_dict.items():
                if cluster_id not in total_profit:
                    total_profit[cluster_id] = {
                        'long_open': 0,
                        'long_close': 0,
                        'short_open': 0,
                        'short_close': 0,
                        'no_action': 0,
                        'total_signals': 0,
                        'total_labels': 0
                    }
                
                total_profit[cluster_id]['long_open'] += profit_info.get('long_open', 0)
                total_profit[cluster_id]['long_close'] += profit_info.get('long_close', 0)
                total_profit[cluster_id]['short_open'] += profit_info.get('short_open', 0)
                total_profit[cluster_id]['short_close'] += profit_info.get('short_close', 0)
                total_profit[cluster_id]['total_signals'] += profit_info.get('total_signals', 0)
                total_profit[cluster_id]['total_labels'] += profit_info.get('total_labels', 0)
                total_labels += profit_info.get('total_labels', 0)
                total_signals += profit_info.get('total_signals', 0)
        
        # 计算平均信号密度
        for cluster_id, profit_info in total_profit.items():
            profit_info['signal_density'] = (
                profit_info['total_signals'] / profit_info['total_labels'] 
                if profit_info['total_labels'] > 0 else 0
            )
        
        # 可视化最佳模式
        visualize_cluster_patterns(load_data(label_files[0]), combined_windows, combined_cluster_labels, all_signal_indices, total_profit, len(np.unique(combined_cluster_labels)) if len(combined_cluster_labels) > 0 else 0)
        
        # 保存模式
        save_cluster_patterns(combined_windows, combined_cluster_labels, total_profit, all_signal_indices, len(np.unique(combined_cluster_labels)) if len(combined_cluster_labels) > 0 else 0)
        
        print("\nProcessing completed!")
        print("Top profitable clusters:")
        sorted_clusters = sorted(total_profit.items(), 
                               key=lambda x: x[1]['signal_density'], 
                               reverse=True)
        for cluster_id, profit_info in sorted_clusters[:5]:
            print(f"  Cluster {cluster_id}: Signal density = {profit_info['signal_density']:.3f}")

if __name__ == "__main__":
    main()