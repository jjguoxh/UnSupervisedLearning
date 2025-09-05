# -*- coding: utf-8 -*-
"""
高级交易模式学习模型
使用序列匹配和深度学习方法识别显著盈利的交易模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
OUTPUT_DIR = "../advanced_patterns/"  # 高级模式输出目录
SEQUENCE_LENGTH = 20  # 序列长度
MIN_SUPPORT = 5  # 最小支持度
SIMILARITY_THRESHOLD = 0.85  # 相似度阈值

class PatternLearner:
    def __init__(self, sequence_length=20, similarity_threshold=0.85, min_support=5):
        self.sequence_length = sequence_length
        self.similarity_threshold = similarity_threshold
        self.min_support = min_support
        self.patterns = []
        self.pattern_sequences = []
        self.pattern_labels = []
        self.pattern_returns = []
        
    def load_data(self, file_path):
        """
        加载标签数据
        """
        df = pd.read_csv(file_path)
        return df
    
    def normalize_sequence(self, sequence):
        """
        标准化序列数据
        """
        # 对价格序列进行归一化（0-1）
        price_seq = sequence[:, -1]  # 最后一列是index_value
        price_min = np.min(price_seq)
        price_max = np.max(price_seq)
        if price_max > price_min:
            normalized_price = (price_seq - price_min) / (price_max - price_min)
        else:
            normalized_price = np.zeros_like(price_seq)
            
        # 对其他特征进行标准化
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(sequence[:, :-1])
        
        # 合并标准化后的特征
        normalized_sequence = np.column_stack([normalized_features, normalized_price])
        return normalized_sequence
    
    def extract_sequences(self, df):
        """
        从数据中提取序列
        """
        sequences = []
        labels = []
        returns = []
        
        # 提取特征列
        feature_columns = ['a', 'b', 'c', 'd', 'index_value']
        features = df[feature_columns].values
        
        # 提取标签
        label_series = df['label'].values
        
        # 计算收益率序列
        index_values = df['index_value'].values
        returns_series = np.diff(index_values) / index_values[:-1]
        # 在开头添加0以保持长度一致
        returns_series = np.insert(returns_series, 0, 0)
        
        # 使用滑动窗口提取序列
        for i in range(len(features) - self.sequence_length + 1):
            sequence = features[i:i+self.sequence_length]
            label_seq = label_series[i:i+self.sequence_length]
            return_seq = returns_series[i:i+self.sequence_length]
            
            # 标准化序列
            normalized_seq = self.normalize_sequence(sequence)
            
            sequences.append(normalized_seq)
            labels.append(label_seq)
            returns.append(return_seq)
        
        return sequences, labels, returns
    
    def calculate_sequence_similarity(self, seq1, seq2):
        """
        计算两个序列的相似度
        """
        # 使用余弦相似度
        # 将三维序列展平为一维向量
        vec1 = seq1.flatten()
        vec2 = seq2.flatten()
        
        # 计算余弦相似度
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return similarity
    
    def find_frequent_patterns(self, sequences, labels, returns):
        """
        发现频繁模式
        """
        print("Finding frequent patterns...")
        frequent_patterns = []
        pattern_returns = []
        
        n_sequences = len(sequences)
        
        # 计算所有序列之间的相似度
        similarity_matrix = np.zeros((n_sequences, n_sequences))
        
        for i in range(n_sequences):
            for j in range(i, n_sequences):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self.calculate_sequence_similarity(sequences[i], sequences[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        # 基于相似度进行聚类
        visited = np.zeros(n_sequences, dtype=bool)
        clusters = []
        
        for i in range(n_sequences):
            if visited[i]:
                continue
                
            # 找到与当前序列相似的所有序列
            similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
            
            # 标记这些序列为已访问
            for idx in similar_indices:
                visited[idx] = True
            
            # 如果聚类大小满足最小支持度要求
            if len(similar_indices) >= self.min_support:
                cluster = {
                    'representative_index': i,
                    'member_indices': similar_indices.tolist(),
                    'size': len(similar_indices),
                    'avg_similarity': np.mean(similarity_matrix[i][similar_indices])
                }
                clusters.append(cluster)
        
        print(f"Found {len(clusters)} frequent pattern clusters")
        return clusters
    
    def calculate_pattern_return(self, cluster, sequences, labels, returns):
        """
        计算模式的预期收益率
        """
        member_indices = cluster['member_indices']
        
        # 收集所有成员序列的未来收益
        future_returns = []
        
        for idx in member_indices:
            # 获取序列最后一个点之后的收益
            # 简化处理：使用序列最后一个点的标签和收益
            if idx < len(labels):
                label_seq = labels[idx]
                return_seq = returns[idx]
                
                # 计算序列的平均收益
                avg_return = np.mean(return_seq)
                future_returns.append(avg_return)
        
        if future_returns:
            expected_return = np.mean(future_returns)
            risk = np.std(future_returns)
            sharpe_ratio = expected_return / risk if risk > 0 else 0
        else:
            expected_return = 0
            risk = 0
            sharpe_ratio = 0
        
        return {
            'expected_return': expected_return,
            'risk': risk,
            'sharpe_ratio': sharpe_ratio,
            'n_samples': len(future_returns)
        }
    
    def analyze_pattern_trading_signals(self, cluster, labels):
        """
        分析模式中的交易信号分布
        """
        member_indices = cluster['member_indices']
        
        # 收集所有交易信号
        all_signals = []
        for idx in member_indices:
            if idx < len(labels):
                label_seq = labels[idx]
                # 只考虑序列最后一个点的信号
                last_label = label_seq[-1]
                if last_label != 0:  # 非无操作信号
                    all_signals.append(last_label)
        
        signal_counts = Counter(all_signals)
        
        # 计算信号强度（信号数量/总样本数）
        total_samples = len(member_indices)
        signal_strength = len(all_signals) / total_samples if total_samples > 0 else 0
        
        return {
            'signal_counts': dict(signal_counts),
            'signal_strength': signal_strength,
            'total_signals': len(all_signals),
            'total_samples': total_samples
        }
    
    def evaluate_pattern_profitability(self, cluster, sequences, labels, returns):
        """
        评估模式的盈利能力
        """
        # 计算预期收益
        return_metrics = self.calculate_pattern_return(cluster, sequences, labels, returns)
        
        # 分析交易信号
        signal_metrics = self.analyze_pattern_trading_signals(cluster, labels)
        
        # 综合盈利能力指标
        profitability_score = (
            return_metrics['sharpe_ratio'] * 0.5 + 
            signal_metrics['signal_strength'] * 0.3 + 
            (return_metrics['expected_return'] > 0) * 0.2
        )
        
        return {
            'cluster_info': cluster,
            'return_metrics': return_metrics,
            'signal_metrics': signal_metrics,
            'profitability_score': profitability_score
        }
    
    def visualize_patterns(self, clusters, sequences, profitability_results, top_k=5):
        """
        可视化最具盈利能力的模式
        """
        # 按盈利能力排序
        sorted_results = sorted(profitability_results, 
                              key=lambda x: x['profitability_score'], 
                              reverse=True)
        
        top_results = sorted_results[:top_k]
        
        fig, axes = plt.subplots(top_k, 1, figsize=(15, 4*top_k))
        if top_k == 1:
            axes = [axes]
        
        for idx, result in enumerate(top_results):
            cluster = result['cluster_info']
            ax = axes[idx]
            
            # 绘制该聚类中的几个代表性序列
            member_indices = cluster['member_indices'][:5]  # 最多绘制5个
            
            for member_idx in member_indices:
                if member_idx < len(sequences):
                    seq = sequences[member_idx]
                    # 绘制价格序列（最后一列）
                    price_seq = seq[:, -1]
                    ax.plot(range(len(price_seq)), price_seq, alpha=0.7, color='blue')
            
            ax.set_title(f'Pattern Cluster {cluster["representative_index"]} '
                        f'(Profitability: {result["profitability_score"]:.3f}, '
                        f'Size: {cluster["size"]})')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Normalized Price')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'profitable_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_patterns(self, profitability_results):
        """
        保存模式到文件
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存所有模式的分析结果
        results_data = []
        for result in profitability_results:
            cluster = result['cluster_info']
            return_metrics = result['return_metrics']
            signal_metrics = result['signal_metrics']
            
            result_entry = {
                'cluster_id': cluster['representative_index'],
                'cluster_size': cluster['size'],
                'avg_similarity': cluster['avg_similarity'],
                'profitability_score': result['profitability_score'],
                'expected_return': return_metrics['expected_return'],
                'risk': return_metrics['risk'],
                'sharpe_ratio': return_metrics['sharpe_ratio'],
                'signal_strength': signal_metrics['signal_strength'],
                'total_signals': signal_metrics['total_signals']
            }
            results_data.append(result_entry)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'pattern_analysis.csv'), index=False)
        
        # 保存前几个最具盈利能力的模式详情
        sorted_results = sorted(profitability_results, 
                              key=lambda x: x['profitability_score'], 
                              reverse=True)
        
        for i, result in enumerate(sorted_results[:10]):  # 保存前10个
            cluster = result['cluster_info']
            cluster_dir = os.path.join(OUTPUT_DIR, f'pattern_{cluster["representative_index"]}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            # 保存聚类信息
            cluster_info = {
                'representative_index': cluster['representative_index'],
                'member_indices': cluster['member_indices'],
                'size': cluster['size'],
                'avg_similarity': cluster['avg_similarity']
            }
            pd.DataFrame([cluster_info]).to_csv(
                os.path.join(cluster_dir, 'cluster_info.csv'), index=False)
            
            # 保存性能指标
            performance_df = pd.DataFrame([
                result['return_metrics'], 
                result['signal_metrics']
            ])
            performance_df.to_csv(
                os.path.join(cluster_dir, 'performance_metrics.csv'), index=False)
    
    def process_file(self, file_path):
        """
        处理单个文件
        """
        print(f"Processing {os.path.basename(file_path)}...")
        
        # 加载数据
        df = self.load_data(file_path)
        
        # 提取序列
        sequences, labels, returns = self.extract_sequences(df)
        print(f"  Extracted {len(sequences)} sequences")
        
        if len(sequences) == 0:
            return []
        
        # 发现频繁模式
        clusters = self.find_frequent_patterns(sequences, labels, returns)
        
        if not clusters:
            print("  No frequent patterns found")
            return []
        
        # 评估模式盈利能力
        profitability_results = []
        for cluster in clusters:
            profitability = self.evaluate_pattern_profitability(cluster, sequences, labels, returns)
            profitability_results.append(profitability)
        
        return profitability_results
    
    def run(self, n_files=10):
        """
        运行完整的模式学习流程
        """
        # 获取所有标签文件
        label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
        print(f"Found {len(label_files)} label files")
        
        # 存储所有结果
        all_results = []
        
        # 处理前n个文件
        for i, file_path in enumerate(label_files[:n_files]):
            try:
                results = self.process_file(file_path)
                all_results.extend(results)
                print(f"  Completed {i+1}/{min(n_files, len(label_files))} files")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_results:
            print("No patterns found in any files!")
            return
        
        print(f"\nTotal pattern clusters found: {len(all_results)}")
        
        # 可视化结果
        # 重新提取一些序列用于可视化
        if label_files:
            sample_df = self.load_data(label_files[0])
            sample_sequences, sample_labels, sample_returns = self.extract_sequences(sample_df)
            self.visualize_patterns([], sample_sequences, all_results, top_k=5)
        
        # 保存结果
        self.save_patterns(all_results)
        
        # 打印最具盈利能力的模式
        sorted_results = sorted(all_results, 
                              key=lambda x: x['profitability_score'], 
                              reverse=True)
        
        print("\nTop profitable patterns:")
        for i, result in enumerate(sorted_results[:10]):
            cluster = result['cluster_info']
            metrics = result['return_metrics']
            signals = result['signal_metrics']
            
            print(f"  {i+1}. Pattern {cluster['representative_index']}:")
            print(f"     Profitability Score: {result['profitability_score']:.3f}")
            print(f"     Cluster Size: {cluster['size']}")
            print(f"     Expected Return: {metrics['expected_return']:.4f}")
            print(f"     Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"     Signal Strength: {signals['signal_strength']:.3f}")
        
        print(f"\nResults saved to {OUTPUT_DIR}")
        return all_results

def main():
    """
    主函数
    """
    # 创建模式学习器
    learner = PatternLearner(
        sequence_length=SEQUENCE_LENGTH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_support=MIN_SUPPORT
    )
    
    # 运行学习流程
    results = learner.run(n_files=5)  # 处理前5个文件作为示例

if __name__ == "__main__":
    main()