# -*- coding: utf-8 -*-
"""
盈利模式检测器
识别能够显著盈利的交易模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
OUTPUT_DIR = "../profitable_patterns/"  # 盈利模式输出目录
CONTEXT_WINDOW = 30  # 信号前后上下文窗口大小
MIN_PROFIT_THRESHOLD = 0.001  # 最小盈利阈值
N_CLUSTERS = 8  # 聚类数量

class ProfitablePatternDetector:
    def __init__(self):
        self.patterns = []
        self.profitable_patterns = []
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        加载单个CSV文件数据
        """
        df = pd.read_csv(file_path)
        return df
    
    def extract_signal_context(self, df, signal_idx, window_size=CONTEXT_WINDOW):
        """
        提取信号前后的时间序列上下文
        """
        start_idx = max(0, signal_idx - window_size)
        end_idx = min(len(df), signal_idx + window_size)
        
        context = df.iloc[start_idx:end_idx][['x', 'a', 'b', 'c', 'd', 'index_value']].copy()
        context['distance_from_signal'] = range(start_idx - signal_idx, end_idx - signal_idx)
        
        return context
    
    def extract_features(self, context_data):
        """
        从上下文数据中提取特征
        """
        if len(context_data) == 0:
            return np.array([])
        
        features = []
        
        # 基本统计特征
        features.extend([
            context_data['index_value'].mean(),
            context_data['index_value'].std(),
            context_data['index_value'].min(),
            context_data['index_value'].max(),
            np.percentile(context_data['index_value'], 25),
            np.percentile(context_data['index_value'], 75)
        ])
        
        # 影响因子特征
        for factor in ['a', 'b', 'c', 'd']:
            features.extend([
                context_data[factor].mean(),
                context_data[factor].std(),
                context_data[factor].min(),
                context_data[factor].max()
            ])
        
        # 趋势特征
        index_values = context_data['index_value'].values
        if len(index_values) > 1:
            # 线性趋势
            slope = np.polyfit(range(len(index_values)), index_values, 1)[0]
            features.append(slope)
            
            # 二次趋势
            try:
                curve = np.polyfit(range(len(index_values)), index_values, 2)[0]
                features.append(curve)
            except:
                features.append(0)
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def calculate_future_performance(self, df, signal_idx, signal_label, hold_period=20):
        """
        计算信号发出后的未来表现
        """
        if signal_idx + hold_period >= len(df):
            return 0, 0
        
        current_price = df.iloc[signal_idx]['index_value']
        future_price = df.iloc[signal_idx + hold_period]['index_value']
        
        # 计算收益率
        price_change = (future_price - current_price) / current_price
        
        # 根据信号类型调整收益率符号
        # 做多开仓(3)和做空平仓(2)希望价格上涨
        # 做空开仓(1)和做多平仓(4)希望价格下跌
        if signal_label in [3, 2]:
            return price_change, hold_period
        else:
            return -price_change, hold_period
    
    def identify_profitable_patterns(self, df):
        """
        识别盈利模式
        """
        # 找到所有交易信号
        signals = df[df['label'] != 0]
        
        profitable_signals = []
        
        for idx, signal_row in signals.iterrows():
            signal_idx = idx
            signal_label = signal_row['label']
            
            # 提取信号上下文
            context = self.extract_signal_context(df, signal_idx)
            
            # 计算未来表现
            future_return, hold_period = self.calculate_future_performance(df, signal_idx, signal_label)
            
            # 判断是否为盈利信号
            is_profitable = future_return > MIN_PROFIT_THRESHOLD
            
            pattern = {
                'signal_index': signal_idx,
                'signal_label': signal_label,
                'context': context,
                'future_return': future_return,
                'hold_period': hold_period,
                'is_profitable': is_profitable,
                'features': self.extract_features(context[context['distance_from_signal'] < 0])  # 信号前的特征
            }
            
            profitable_signals.append(pattern)
        
        return profitable_signals
    
    def cluster_patterns(self, patterns):
        """
        对模式进行聚类
        """
        # 提取特征矩阵
        feature_matrix = np.array([p['features'] for p in patterns if len(p['features']) > 0])
        
        if len(feature_matrix) == 0:
            return []
        
        # 标准化特征
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # 执行聚类
        kmeans = KMeans(n_clusters=min(N_CLUSTERS, len(feature_matrix)), random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # 将聚类标签添加到模式中
        valid_patterns = [p for p in patterns if len(p['features']) > 0]
        for i, pattern in enumerate(valid_patterns):
            pattern['cluster_id'] = cluster_labels[i]
        
        return valid_patterns
    
    def evaluate_cluster_profitability(self, patterns):
        """
        评估聚类的盈利能力
        """
        cluster_stats = {}
        
        # 按聚类分组
        for pattern in patterns:
            cluster_id = pattern.get('cluster_id', -1)
            if cluster_id == -1:
                continue
                
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {
                    'patterns': [],
                    'profitable_count': 0,
                    'total_count': 0,
                    'total_return': 0
                }
            
            cluster_stats[cluster_id]['patterns'].append(pattern)
            cluster_stats[cluster_id]['total_count'] += 1
            cluster_stats[cluster_id]['total_return'] += pattern['future_return']
            if pattern['is_profitable']:
                cluster_stats[cluster_id]['profitable_count'] += 1
        
        # 计算聚类统计信息
        for cluster_id, stats in cluster_stats.items():
            stats['profit_rate'] = stats['profitable_count'] / stats['total_count'] if stats['total_count'] > 0 else 0
            stats['avg_return'] = stats['total_return'] / stats['total_count'] if stats['total_count'] > 0 else 0
            stats['profitability_score'] = stats['profit_rate'] * 0.7 + (stats['avg_return'] > 0) * 0.3
        
        return cluster_stats
    
    def visualize_profitable_clusters(self, cluster_stats, top_k=5):
        """
        可视化盈利聚类
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 按盈利能力排序
        sorted_clusters = sorted(cluster_stats.items(), 
                               key=lambda x: x[1]['profitability_score'], 
                               reverse=True)
        top_clusters = sorted_clusters[:min(top_k, len(sorted_clusters))]
        
        fig, axes = plt.subplots(len(top_clusters), 1, figsize=(15, 4*len(top_clusters)))
        if len(top_clusters) == 1:
            axes = [axes]
        
        for i, (cluster_id, stats) in enumerate(top_clusters):
            ax = axes[i] if len(top_clusters) > 1 else axes[0]
            
            # 绘制该聚类中的前几个模式
            patterns = stats['patterns'][:5]  # 最多绘制5个
            for pattern in patterns:
                context = pattern['context']
                ax.plot(context['distance_from_signal'], context['index_value'], 
                       alpha=0.7, linewidth=1)
            
            ax.set_title(f'Cluster {cluster_id} (Profit Rate: {stats["profit_rate"]:.2%}, '
                        f'Avg Return: {stats["avg_return"]:.4f})')
            ax.set_xlabel('Time (relative to signal)')
            ax.set_ylabel('Index Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'profitable_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_cluster_analysis(self, cluster_stats):
        """
        保存聚类分析结果
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 创建聚类统计DataFrame
        cluster_data = []
        for cluster_id, stats in cluster_stats.items():
            cluster_data.append({
                'cluster_id': cluster_id,
                'pattern_count': stats['total_count'],
                'profitable_count': stats['profitable_count'],
                'profit_rate': stats['profit_rate'],
                'avg_return': stats['avg_return'],
                'profitability_score': stats['profitability_score']
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        cluster_df = cluster_df.sort_values('profitability_score', ascending=False)
        cluster_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_analysis.csv'), index=False)
        
        # 打印结果
        print("\nCluster Analysis Results:")
        print("=" * 50)
        for _, row in cluster_df.iterrows():
            print(f"Cluster {row['cluster_id']}:")
            print(f"  Pattern Count: {row['pattern_count']}")
            print(f"  Profit Rate: {row['profit_rate']:.2%}")
            print(f"  Avg Return: {row['avg_return']:.4f}")
            print(f"  Profitability Score: {row['profitability_score']:.4f}")
            print()
    
    def find_similar_patterns(self, target_pattern, all_patterns, top_k=5):
        """
        查找相似模式
        """
        if len(target_pattern['features']) == 0:
            return []
        
        similarities = []
        target_features = target_pattern['features'].reshape(1, -1)
        target_features_scaled = self.scaler.transform(target_features)
        
        for pattern in all_patterns:
            if len(pattern['features']) == 0 or pattern == target_pattern:
                continue
                
            pattern_features = pattern['features'].reshape(1, -1)
            pattern_features_scaled = self.scaler.transform(pattern_features)
            
            # 计算余弦相似度
            similarity = cosine_similarity(target_features_scaled, pattern_features_scaled)[0][0]
            similarities.append((pattern, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def process_file(self, file_path):
        """
        处理单个文件
        """
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        # 加载数据
        df = self.load_data(file_path)
        
        # 识别盈利模式
        profitable_patterns = self.identify_profitable_patterns(df)
        print(f"  Found {len(profitable_patterns)} patterns ({sum(1 for p in profitable_patterns if p['is_profitable'])} profitable)")
        
        return profitable_patterns
    
    def run(self, n_files=10):
        """
        运行完整的模式检测流程
        """
        # 获取所有标签文件
        label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
        print(f"Found {len(label_files)} label files")
        
        # 处理所有文件
        all_patterns = []
        for i, file_path in enumerate(label_files[:n_files]):
            try:
                patterns = self.process_file(file_path)
                all_patterns.extend(patterns)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_patterns:
            print("No patterns found!")
            return
        
        print(f"\nTotal patterns found: {len(all_patterns)}")
        print(f"Profitable patterns: {sum(1 for p in all_patterns if p['is_profitable'])}")
        
        # 对模式进行聚类
        clustered_patterns = self.cluster_patterns(all_patterns)
        print(f"Clustered patterns: {len(clustered_patterns)}")
        
        if not clustered_patterns:
            print("No patterns could be clustered!")
            return
        
        # 评估聚类盈利能力
        cluster_stats = self.evaluate_cluster_profitability(clustered_patterns)
        
        # 可视化结果
        self.visualize_profitable_clusters(cluster_stats)
        
        # 保存分析结果
        self.save_cluster_analysis(cluster_stats)
        
        # 保存最佳聚类的详细信息
        self.save_best_clusters(cluster_stats)
        
        print(f"\nAnalysis completed! Results saved to {OUTPUT_DIR}")
        
        return cluster_stats
    
    def save_best_clusters(self, cluster_stats, top_k=3):
        """
        保存最佳聚类的详细信息
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 按盈利能力排序
        sorted_clusters = sorted(cluster_stats.items(), 
                               key=lambda x: x[1]['profitability_score'], 
                               reverse=True)
        best_clusters = sorted_clusters[:min(top_k, len(sorted_clusters))]
        
        for i, (cluster_id, stats) in enumerate(best_clusters):
            cluster_dir = os.path.join(OUTPUT_DIR, f'cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            # 保存聚类统计信息
            stats_df = pd.DataFrame([{
                'cluster_id': cluster_id,
                'pattern_count': stats['total_count'],
                'profitable_count': stats['profitable_count'],
                'profit_rate': stats['profit_rate'],
                'avg_return': stats['avg_return'],
                'profitability_score': stats['profitability_score']
            }])
            stats_df.to_csv(os.path.join(cluster_dir, 'cluster_stats.csv'), index=False)
            
            # 保存该聚类中的模式示例
            patterns = stats['patterns'][:10]  # 保存前10个
            pattern_data = []
            for j, pattern in enumerate(patterns):
                pattern_data.append({
                    'pattern_id': j,
                    'signal_index': pattern['signal_index'],
                    'signal_label': pattern['signal_label'],
                    'future_return': pattern['future_return'],
                    'is_profitable': pattern['is_profitable']
                })
                
                # 保存上下文数据
                context_df = pattern['context']
                context_df.to_csv(os.path.join(cluster_dir, f'pattern_{j}_context.csv'), index=False)
            
            patterns_df = pd.DataFrame(pattern_data)
            patterns_df.to_csv(os.path.join(cluster_dir, 'patterns.csv'), index=False)

def main():
    """
    主函数
    """
    # 创建盈利模式检测器
    detector = ProfitablePatternDetector()
    
    # 运行检测流程
    cluster_stats = detector.run(n_files=10)

if __name__ == "__main__":
    main()