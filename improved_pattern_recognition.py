# -*- coding: utf-8 -*-
"""
改进的交易模式识别模型
解决原有系统的关键问题：
1. 增强特征工程
2. 优化聚类算法
3. 改进相似性计算
4. 修复模式保存机制
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import talib
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "./label/"  # 标签数据目录
OUTPUT_DIR = "./patterns_improved/"  # 改进的模式输出目录
WINDOW_SIZE = 20  # 减小时间窗口
PATTERN_LENGTH = 15  # 增加模式长度
MIN_CLUSTER_SIZE = 3  # 最小聚类大小
MAX_CLUSTERS = 50  # 最大聚类数量

class ImprovedPatternRecognition:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def extract_enhanced_features(self, df, window_size=WINDOW_SIZE):
        """
        提取增强的技术指标特征
        """
        features = []
        
        # 确保数据长度足够计算技术指标
        if len(df) < max(window_size, 30):
            return pd.DataFrame()
        
        # 计算技术指标
        prices = df['index_value'].values.astype(float)
        
        # 移动平均线
        sma_5 = talib.SMA(prices, timeperiod=5)
        sma_10 = talib.SMA(prices, timeperiod=10)
        sma_20 = talib.SMA(prices, timeperiod=20)
        
        # 指数移动平均
        ema_12 = talib.EMA(prices, timeperiod=12)
        ema_26 = talib.EMA(prices, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(prices)
        
        # RSI
        rsi = talib.RSI(prices, timeperiod=14)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
        
        # 成交量相关（使用价格变化作为代理）
        volume_proxy = np.abs(np.diff(prices, prepend=prices[0]))
        
        # 波动率
        volatility = talib.STDDEV(prices, timeperiod=10)
        
        # 使用滑动窗口提取特征
        start_idx = max(30, window_size)  # 确保有足够的历史数据计算指标
        
        for i in range(start_idx, len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]
            
            # 基础价格特征
            price_features = {
                'price_start': prices[i],
                'price_end': prices[i+window_size-1],
                'price_min': np.min(prices[i:i+window_size]),
                'price_max': np.max(prices[i:i+window_size]),
                'price_mean': np.mean(prices[i:i+window_size]),
                'price_std': np.std(prices[i:i+window_size]),
                'price_change': (prices[i+window_size-1] - prices[i]) / prices[i],
            }
            
            # 技术指标特征
            tech_features = {
                'sma_5': sma_5[i+window_size-1] if not np.isnan(sma_5[i+window_size-1]) else 0,
                'sma_10': sma_10[i+window_size-1] if not np.isnan(sma_10[i+window_size-1]) else 0,
                'sma_20': sma_20[i+window_size-1] if not np.isnan(sma_20[i+window_size-1]) else 0,
                'ema_12': ema_12[i+window_size-1] if not np.isnan(ema_12[i+window_size-1]) else 0,
                'ema_26': ema_26[i+window_size-1] if not np.isnan(ema_26[i+window_size-1]) else 0,
                'macd': macd[i+window_size-1] if not np.isnan(macd[i+window_size-1]) else 0,
                'macd_signal': macd_signal[i+window_size-1] if not np.isnan(macd_signal[i+window_size-1]) else 0,
                'macd_hist': macd_hist[i+window_size-1] if not np.isnan(macd_hist[i+window_size-1]) else 0,
                'rsi': rsi[i+window_size-1] if not np.isnan(rsi[i+window_size-1]) else 50,
                'bb_position': (prices[i+window_size-1] - bb_lower[i+window_size-1]) / (bb_upper[i+window_size-1] - bb_lower[i+window_size-1]) if not np.isnan(bb_upper[i+window_size-1]) else 0.5,
                'volatility': volatility[i+window_size-1] if not np.isnan(volatility[i+window_size-1]) else 0,
            }
            
            # 影响因子特征
            factor_features = {}
            for factor in ['a', 'b', 'c', 'd']:
                if factor in window.columns:
                    factor_values = window[factor].values
                    factor_features.update({
                        f'{factor}_mean': np.mean(factor_values),
                        f'{factor}_std': np.std(factor_values),
                        f'{factor}_min': np.min(factor_values),
                        f'{factor}_max': np.max(factor_values),
                        f'{factor}_trend': np.polyfit(range(len(factor_values)), factor_values, 1)[0] if len(factor_values) > 1 else 0,
                    })
            
            # 合并所有特征
            window_features = {
                'start_idx': i,
                'end_idx': i + window_size - 1,
                **price_features,
                **tech_features,
                **factor_features
            }
            
            features.append(window_features)
        
        return pd.DataFrame(features)
    
    def identify_trading_signals(self, df):
        """
        识别交易信号，排除标签0，并应用交易逻辑约束
        约束规则：
        1. 在一个方向开仓后，必须等该方向平仓才能出现反方向开仓信号
        2. 同一方向的重复开仓信号只保留第一个
        """
        # 首先提取所有原始交易信号
        raw_signals = []
        for i in range(len(df)):
            label = df['label'].iloc[i]
            if label in [1, 2, 3, 4]:  # 只保留交易信号
                raw_signals.append({
                    'index': i,
                    'x': df['x'].iloc[i],
                    'label': label,
                    'index_value': df['index_value'].iloc[i]
                })
        
        # 应用交易逻辑约束过滤信号
        filtered_signals = self.filter_trading_signals(raw_signals)
        
        # 添加影响因子
        for signal in filtered_signals:
            for factor in ['a', 'b', 'c', 'd']:
                if factor in df.columns:
                    signal[factor] = df[factor].iloc[signal['index']]
        
        return pd.DataFrame(filtered_signals)
    
    def filter_trading_signals(self, raw_signals):
        """
        过滤交易信号，应用交易逻辑约束
        """
        if not raw_signals:
            return []
        
        filtered_signals = []
        position_state = 0  # 0: 无仓位, 1: 多头仓位, -1: 空头仓位
        
        for signal in raw_signals:
            label = signal['label']
            should_keep = False
            
            if label == 1:  # 做多开仓
                if position_state == 0:  # 无仓位时可以开多仓
                    should_keep = True
                    position_state = 1
                # 如果已有多头仓位或空头仓位，忽略重复开仓信号
                
            elif label == 2:  # 做多平仓
                if position_state == 1:  # 有多头仓位时可以平仓
                    should_keep = True
                    position_state = 0
                # 如果无多头仓位，忽略平仓信号
                
            elif label == 3:  # 做空开仓
                if position_state == 0:  # 无仓位时可以开空仓
                    should_keep = True
                    position_state = -1
                # 如果已有空头仓位或多头仓位，忽略重复开仓信号
                
            elif label == 4:  # 做空平仓
                if position_state == -1:  # 有空头仓位时可以平仓
                    should_keep = True
                    position_state = 0
                # 如果无空头仓位，忽略平仓信号
            
            if should_keep:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def extract_signal_patterns(self, df, signals, pattern_length=PATTERN_LENGTH):
        """
        提取信号前后的模式，包含更多上下文信息
        """
        patterns = []
        
        for _, signal in signals.iterrows():
            signal_idx = signal['index']
            
            # 确保有足够的历史数据
            if signal_idx >= pattern_length:
                start_idx = signal_idx - pattern_length
                end_idx = signal_idx
                pattern_data = df.iloc[start_idx:end_idx]
                
                # 提取模式特征向量
                feature_vector = []
                
                # 价格序列特征
                prices = pattern_data['index_value'].values
                normalized_prices = (prices - prices[0]) / prices[0]  # 归一化
                feature_vector.extend(normalized_prices)
                
                # 价格变化率
                price_changes = np.diff(prices) / prices[:-1]
                feature_vector.extend(price_changes)
                
                # 影响因子特征
                for factor in ['a', 'b', 'c', 'd']:
                    if factor in pattern_data.columns:
                        factor_values = pattern_data[factor].values
                        feature_vector.extend([
                            np.mean(factor_values),
                            np.std(factor_values),
                            np.min(factor_values),
                            np.max(factor_values)
                        ])
                
                pattern = {
                    'signal_index': signal_idx,
                    'signal_label': signal['label'],
                    'pattern_start_idx': start_idx,
                    'pattern_end_idx': end_idx,
                    'feature_vector': feature_vector,
                    'context_data': pattern_data.copy()
                }
                patterns.append(pattern)
        
        return patterns
    
    def calculate_multi_dimensional_similarity(self, pattern1, pattern2):
        """
        计算多维特征相似性
        """
        vec1 = np.array(pattern1['feature_vector'])
        vec2 = np.array(pattern2['feature_vector'])
        
        if len(vec1) != len(vec2):
            return 0
        
        # 欧几里得距离相似性
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # 余弦相似性
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0
        else:
            cosine_sim = dot_product / (norm1 * norm2)
            cosine_sim = (cosine_sim + 1) / 2  # 归一化到[0,1]
        
        # 皮尔逊相关系数
        try:
            correlation = np.corrcoef(vec1, vec2)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            correlation = (correlation + 1) / 2  # 归一化到[0,1]
        except:
            correlation = 0
        
        # 综合相似性得分
        similarity = (euclidean_sim * 0.3 + cosine_sim * 0.4 + correlation * 0.3)
        return similarity
    
    def optimal_clustering(self, patterns, max_clusters=MAX_CLUSTERS):
        """
        使用优化的聚类算法
        """
        if len(patterns) < MIN_CLUSTER_SIZE:
            return [list(range(len(patterns)))]
        
        # 构建特征矩阵
        feature_matrix = []
        for pattern in patterns:
            feature_matrix.append(pattern['feature_vector'])
        
        feature_matrix = np.array(feature_matrix)
        
        # 标准化特征
        feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)
        
        # 使用肘部法则确定最优聚类数
        max_k = min(max_clusters, len(patterns) // MIN_CLUSTER_SIZE)
        if max_k < 2:
            return [list(range(len(patterns)))]
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
            
            inertias.append(kmeans.inertia_)
            
            if len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(feature_matrix_scaled, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # 选择最优K值（基于轮廓系数）
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 2
        
        # 执行最终聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
        
        # 过滤小聚类
        filtered_clusters = [cluster for cluster in clusters.values() if len(cluster) >= MIN_CLUSTER_SIZE]
        
        print(f"Optimal clustering: {len(filtered_clusters)} clusters from {len(patterns)} patterns")
        return filtered_clusters
    
    def analyze_cluster_profitability(self, patterns, clusters):
        """
        分析聚类盈利能力（改进版）
        """
        cluster_analysis = []
        
        for cluster_idx, cluster in enumerate(clusters):
            # 收集聚类中的信号
            cluster_signals = []
            for pattern_idx in cluster:
                pattern = patterns[pattern_idx]
                cluster_signals.append(pattern['signal_label'])
            
            # 统计信号类型
            signal_counts = Counter(cluster_signals)
            
            # 计算盈利指标
            long_open = signal_counts.get(1, 0)
            long_close = signal_counts.get(2, 0)
            short_open = signal_counts.get(3, 0)
            short_close = signal_counts.get(4, 0)
            
            # 计算配对交易
            long_pairs = min(long_open, long_close)
            short_pairs = min(short_open, short_close)
            total_pairs = long_pairs + short_pairs
            
            # 计算信号密度和质量指标
            cluster_size = len(cluster)
            signal_density = total_pairs / cluster_size if cluster_size > 0 else 0
            
            # 计算信号平衡度
            total_signals = sum(signal_counts.values())
            signal_balance = 1 - abs(long_open + long_close - short_open - short_close) / total_signals if total_signals > 0 else 0
            
            # 计算聚类质量得分
            quality_score = signal_density * 0.6 + signal_balance * 0.4
            
            analysis = {
                'cluster_id': cluster_idx,
                'cluster_size': cluster_size,
                'signal_counts': dict(signal_counts),
                'long_open': long_open,
                'long_close': long_close,
                'short_open': short_open,
                'short_close': short_close,
                'long_pairs': long_pairs,
                'short_pairs': short_pairs,
                'total_pairs': total_pairs,
                'signal_density': signal_density,
                'signal_balance': signal_balance,
                'quality_score': quality_score
            }
            
            cluster_analysis.append(analysis)
        
        # 按质量得分排序
        cluster_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
        return cluster_analysis
    
    def save_patterns_properly(self, patterns, clusters, cluster_analysis):
        """
        正确保存模式数据
        """
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存聚类分析结果
        analysis_df = pd.DataFrame(cluster_analysis)
        analysis_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_analysis.csv"), index=False)
        
        # 保存每个聚类的详细信息
        for analysis in cluster_analysis:
            cluster_id = analysis['cluster_id']
            cluster_dir = os.path.join(OUTPUT_DIR, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # 获取该聚类的模式
            cluster_patterns = [patterns[i] for i in clusters[cluster_id]]
            
            # 保存模式数据
            pattern_data = []
            for i, pattern in enumerate(cluster_patterns):
                pattern_info = {
                    'pattern_id': i,
                    'signal_index': pattern['signal_index'],
                    'signal_label': pattern['signal_label'],
                    'pattern_start_idx': pattern['pattern_start_idx'],
                    'pattern_end_idx': pattern['pattern_end_idx']
                }
                pattern_data.append(pattern_info)
                
                # 保存上下文数据
                context_df = pattern['context_data']
                context_df.to_csv(os.path.join(cluster_dir, f'pattern_{i}_context.csv'), index=False)
            
            # 保存模式汇总
            patterns_df = pd.DataFrame(pattern_data)
            patterns_df.to_csv(os.path.join(cluster_dir, 'patterns.csv'), index=False)
            
            # 保存聚类统计信息
            with open(os.path.join(cluster_dir, 'cluster_info.txt'), 'w') as f:
                f.write(f"Cluster {cluster_id} Information\n")
                f.write(f"========================\n")
                f.write(f"Cluster Size: {analysis['cluster_size']}\n")
                f.write(f"Signal Density: {analysis['signal_density']:.4f}\n")
                f.write(f"Quality Score: {analysis['quality_score']:.4f}\n")
                f.write(f"Signal Counts: {analysis['signal_counts']}\n")
                f.write(f"Long Pairs: {analysis['long_pairs']}\n")
                f.write(f"Short Pairs: {analysis['short_pairs']}\n")
        
        print(f"Patterns saved to {OUTPUT_DIR}")
        print(f"Total clusters: {len(cluster_analysis)}")
        print(f"High-quality clusters (quality_score > 0.3): {sum(1 for c in cluster_analysis if c['quality_score'] > 0.3)}")
    
    def process_file(self, file_path):
        """
        处理单个文件
        """
        print(f"Processing {os.path.basename(file_path)}...")
        
        # 加载数据
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} samples")
        
        # 识别交易信号
        signals = self.identify_trading_signals(df)
        print(f"  Found {len(signals)} trading signals")
        
        if len(signals) == 0:
            print(f"  No trading signals found, skipping...")
            return [], [], []
        
        # 提取信号模式
        patterns = self.extract_signal_patterns(df, signals, PATTERN_LENGTH)
        print(f"  Extracted {len(patterns)} patterns")
        
        if len(patterns) < MIN_CLUSTER_SIZE:
            print(f"  Too few patterns for clustering, skipping...")
            return patterns, [], []
        
        # 执行聚类
        clusters = self.optimal_clustering(patterns)
        print(f"  Created {len(clusters)} clusters")
        
        # 分析聚类盈利能力
        cluster_analysis = self.analyze_cluster_profitability(patterns, clusters)
        
        return patterns, clusters, cluster_analysis
    
    def run(self, n_files=None):
        """
        运行改进的模式识别流程
        """
        # 获取所有标签文件
        label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
        print(f"Found {len(label_files)} label files")
        
        if n_files:
            label_files = label_files[:n_files]
        
        # 处理所有文件
        all_patterns = []
        all_clusters = []
        all_cluster_analysis = []
        
        for file_path in label_files:
            try:
                patterns, clusters, cluster_analysis = self.process_file(file_path)
                
                if patterns and clusters and cluster_analysis:
                    # 调整聚类索引以避免冲突
                    offset = len(all_patterns)
                    adjusted_clusters = [[idx + offset for idx in cluster] for cluster in clusters]
                    
                    for analysis in cluster_analysis:
                        analysis['cluster_id'] += len(all_cluster_analysis)
                    
                    all_patterns.extend(patterns)
                    all_clusters.extend(adjusted_clusters)
                    all_cluster_analysis.extend(cluster_analysis)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_patterns:
            print("No patterns found across all files!")
            return
        
        print(f"\nTotal patterns: {len(all_patterns)}")
        print(f"Total clusters: {len(all_clusters)}")
        
        # 保存结果
        self.save_patterns_properly(all_patterns, all_clusters, all_cluster_analysis)
        
        # 显示质量统计
        high_quality = [c for c in all_cluster_analysis if c['quality_score'] > 0.3]
        medium_quality = [c for c in all_cluster_analysis if 0.1 < c['quality_score'] <= 0.3]
        
        print(f"\nQuality Analysis:")
        print(f"  High quality clusters (>0.3): {len(high_quality)}")
        print(f"  Medium quality clusters (0.1-0.3): {len(medium_quality)}")
        print(f"  Low quality clusters (<0.1): {len(all_cluster_analysis) - len(high_quality) - len(medium_quality)}")
        
        return all_patterns, all_clusters, all_cluster_analysis

def main():
    """
    主函数
    """
    print("Starting Improved Pattern Recognition...")
    
    # 创建改进的模式识别器
    recognizer = ImprovedPatternRecognition()
    
    # 运行识别流程
    patterns, clusters, analysis = recognizer.run(n_files=5)  # 处理前5个文件
    
    if patterns:
        print("\n=== Pattern Recognition Completed Successfully ===")
        print(f"Results saved to: {OUTPUT_DIR}")
    else:
        print("\n=== No Patterns Found ===")

if __name__ == "__main__":
    main()