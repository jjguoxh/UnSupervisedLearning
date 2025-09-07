# -*- coding: utf-8 -*-
"""
模式预测程序（使用严格平衡后的数据）
基于历史模式识别结果进行交易信号预测
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import logging
from collections import Counter
import warnings
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 导入强化学习交易器
from simple_rl_trader import SimpleRLTrader
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "..", "label/")  # 标签数据目录
PATTERNS_DIR = os.path.join(CURRENT_DIR, "..", "patterns/")  # 模式数据目录
STRICT_BALANCED_DIR = os.path.join(CURRENT_DIR, "..", "patterns/strict_balanced/")  # 严格平衡后的数据目录
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "model/balanced_model/")  # 平衡模型保存目录
PATTERN_LENGTH = 10  # 模式长度

class BalancedPatternPredictor:
    def __init__(self):
        self.patterns = {}
        self.cluster_models = {}
        self.thresholds = {}
        self.rl_trader = None  # 强化学习交易器
        self.load_patterns()
        
    def load_patterns(self):
        """
        加载已学习的模式（使用严格平衡后的数据）
        """
        logger.info("Loading strictly balanced patterns...")
        logger.info(f"Patterns directory: {PATTERNS_DIR}")
        logger.info(f"Strictly balanced directory: {STRICT_BALANCED_DIR}")
        
        # 加载聚类分析结果（使用严格平衡后的文件）
        cluster_analysis_path = os.path.join(STRICT_BALANCED_DIR, "cluster_analysis_strict_balanced.csv")
        logger.info(f"Cluster analysis path: {cluster_analysis_path}")
        if not os.path.exists(cluster_analysis_path):
            logger.error("Error: Strictly balanced cluster analysis file not found!")
            return
            
        try:
            cluster_df = pd.read_csv(cluster_analysis_path)
        except Exception as e:
            logger.error(f"Error reading cluster analysis file: {e}")
            return
        
        # 加载每个聚类的模式数据
        loaded_clusters = 0
        for _, row in cluster_df.iterrows():
            cluster_id = row['cluster_id']
            signal_density = row['signal_density']
            
            # 从原始patterns目录加载聚类数据
            cluster_dir = os.path.join(PATTERNS_DIR, f"cluster_{cluster_id}")
            if not os.path.exists(cluster_dir):
                logger.warning(f"Cluster directory {cluster_dir} not found!")
                continue
                
            # 加载该聚类的模式文件
            pattern_files = glob.glob(os.path.join(cluster_dir, "pattern_*.csv"))
            patterns = []
            
            for pattern_file in pattern_files:
                try:
                    pattern_data = pd.read_csv(pattern_file)
                    patterns.append(pattern_data)
                except Exception as e:
                    logger.warning(f"Error loading pattern {pattern_file}: {e}")
                    continue
            
            if patterns:
                self.patterns[cluster_id] = {
                    'patterns': patterns,
                    'signal_density': signal_density,
                    'signal_counts': eval(str(row['signal_counts'])) if isinstance(row['signal_counts'], str) else row['signal_counts'],
                    'long_pairs': row['long_pairs'],
                    'short_pairs': row['short_pairs']
                }
                
                # 为所有聚类创建预测模型（由于数据已平衡，可以为所有聚类创建模型）
                # 降低信号密度阈值，使用更多的聚类
                if signal_density >= 0.1:  # 从0.3降低到0.1以增加模型数量
                    self.cluster_models[cluster_id] = self.create_cluster_model(patterns)
                    loaded_clusters += 1
        
        logger.info(f"Loaded {loaded_clusters} clusters, {len(self.cluster_models)} predictive models from balanced data")
    
    def create_cluster_model(self, patterns):
        """
        为特定聚类创建预测模型
        """
        if not patterns:
            return None
            
        # 计算模式的统计特征
        index_values_list = []
        diff_values_list = []
        diff2_values_list = []
        
        for pattern in patterns:
            if 'index_value' in pattern.columns:
                index_values_list.append(pattern['index_value'].values)
            # 如果模式是字典格式（来自extract_recent_pattern）
            elif isinstance(pattern, dict) and 'index_value' in pattern:
                index_values_list.append(pattern['index_value'])
                
        if not index_values_list:
            return None
            
        # 计算平均模式
        max_length = max(len(vals) for vals in index_values_list)
        padded_values = []
        for vals in index_values_list:
            if len(vals) < max_length:
                # 填充到相同长度
                padded = np.pad(vals, (0, max_length - len(vals)), mode='edge')
            else:
                padded = vals
            padded_values.append(padded)
        
        try:
            avg_pattern = np.mean(padded_values, axis=0)
            std_pattern = np.std(padded_values, axis=0)
        except Exception as e:
            logger.error(f"Error calculating pattern statistics: {e}")
            return None
        
        return {
            'avg_pattern': avg_pattern,
            'std_pattern': std_pattern,
            'pattern_length': len(avg_pattern)
        }

    def extract_recent_pattern(self, df, end_idx, pattern_length=PATTERN_LENGTH):
        """
        从数据中提取最近的模式
        """
        start_idx = max(0, end_idx - pattern_length)
        if end_idx <= start_idx:
            return None
            
        pattern_data = df.iloc[start_idx:end_idx]
        
        # 提取更多的特征
        index_values = pattern_data['index_value'].values
        
        # 计算一些额外的特征
        diff_values = np.diff(index_values, prepend=index_values[0])  # 一阶差分
        diff2_values = np.diff(diff_values, prepend=diff_values[0])   # 二阶差分
        
        return {
            'index_value': index_values,
            'diff': diff_values,
            'diff2': diff2_values,
            'a': pattern_data['a'].values,
            'b': pattern_data['b'].values,
            'c': pattern_data['c'].values,
            'd': pattern_data['d'].values,
            'x': pattern_data['x'].values
        }

    def calculate_pattern_similarity(self, pattern1, pattern2):
        """
        计算两个模式的相似性
        """
        # 处理不同格式的输入
        if isinstance(pattern1, dict):
            pattern1_values = pattern1['index_value']
        else:
            pattern1_values = pattern1
            
        if isinstance(pattern2, dict):
            pattern2_values = pattern2['index_value']
        else:
            pattern2_values = pattern2
        
        # 确保两个模式长度相同
        if len(pattern1_values) != len(pattern2_values):
            # 如果长度不同，使用较短的长度
            min_length = min(len(pattern1_values), len(pattern2_values))
            pattern1_values = pattern1_values[:min_length]
            pattern2_values = pattern2_values[:min_length]
            
        if len(pattern1_values) == 0:
            return 0
            
        # 使用多种相似性度量方法
        try:
            # 1. 皮尔逊相关系数
            correlation = np.corrcoef(pattern1_values, pattern2_values)[0, 1]
            corr_similarity = correlation if not np.isnan(correlation) else 0
            
            # 2. 欧几里得距离相似性
            euclidean_distance = np.linalg.norm(pattern1_values - pattern2_values)
            euclidean_similarity = 1 / (1 + euclidean_distance)
            
            # 3. 余弦相似性
            dot_product = np.dot(pattern1_values, pattern2_values)
            norms = np.linalg.norm(pattern1_values) * np.linalg.norm(pattern2_values)
            cosine_similarity = dot_product / norms if norms != 0 else 0
            
            # 4. 动态时间规整(DTW)相似性（简化版本）
            # 这里我们使用一个简化的DTW实现
            dtw_similarity = 1 / (1 + self.calculate_dtw_distance(pattern1_values, pattern2_values))
            
            # 组合多种相似性度量
            combined_similarity = (corr_similarity + euclidean_similarity + cosine_similarity + dtw_similarity) / 4
            
            return combined_similarity
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0
    
    def calculate_dtw_distance(self, x, y):
        """
        计算两个序列之间的DTW距离（简化版本）
        """
        try:
            # 简化的DTW实现
            n, m = len(x), len(y)
            dtw_matrix = np.zeros((n+1, m+1))
            
            # 初始化边界条件
            for i in range(1, n+1):
                dtw_matrix[i, 0] = np.inf
            for j in range(1, m+1):
                dtw_matrix[0, j] = np.inf
            dtw_matrix[0, 0] = 0
            
            # 填充DTW矩阵
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(x[i-1] - y[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],    # 插入
                        dtw_matrix[i, j-1],    # 删除
                        dtw_matrix[i-1, j-1]   # 匹配
                    )
            
            return dtw_matrix[n, m]
        except:
            # 如果DTW计算失败，返回欧几里得距离作为备选
            return np.linalg.norm(x - y)
    
    def predict_signal_ensemble(self, df, current_idx):
        """
        使用集成方法预测信号
        """
        # 提取最近的模式
        recent_pattern = self.extract_recent_pattern(df, current_idx)
        if recent_pattern is None:
            return 0, 0.0  # 无操作或持仓，置信度0
        
        # 存储所有聚类的预测结果
        predictions = []
        
        # 对每个聚类进行预测
        for cluster_id, model in self.cluster_models.items():
            if model is None:
                continue
            
            # 计算与该聚类平均模式的相似性
            similarity = self.calculate_pattern_similarity(
                recent_pattern, 
                model['avg_pattern']
            )
            
            # 获取聚类信息
            cluster_info = self.patterns[cluster_id]
            signal_counts = cluster_info['signal_counts']
            
            # 预测信号类型（选择最常见的信号）
            if signal_counts:
                predicted_signal = max(signal_counts, key=signal_counts.get)
                # 计算置信度
                confidence = similarity * (1 + 2 * cluster_info['signal_density'])
                
                predictions.append({
                    'cluster_id': cluster_id,
                    'signal': predicted_signal,
                    'confidence': confidence,
                    'similarity': similarity,
                    'signal_density': cluster_info['signal_density']
                })
        
        # 如果没有预测结果，返回无操作
        if not predictions:
            return 0, 0.0
        
        # 根据置信度加权投票
        signal_votes = {}
        total_confidence = 0
        
        for pred in predictions:
            signal = pred['signal']
            confidence = pred['confidence']
            
            if signal not in signal_votes:
                signal_votes[signal] = 0
            signal_votes[signal] += confidence
            total_confidence += confidence
        
        # 选择得票最高的信号
        best_signal = max(signal_votes, key=signal_votes.get)
        best_confidence = signal_votes[best_signal] / total_confidence if total_confidence > 0 else 0
        
        # 动态调整置信度阈值
        dynamic_threshold = 0.1
        # 如果投票一致性高，可以降低阈值
        if len(signal_votes) == 1:
            dynamic_threshold = 0.05
        # 如果投票分散，需要提高阈值
        elif len(signal_votes) > 3:
            dynamic_threshold = 0.15
        
        # 只有当置信度足够高时才返回预测信号，否则返回0（无操作）
        if best_confidence < dynamic_threshold:
            return 0, 0.0
        
        return best_signal, best_confidence

    def predict_signal(self, df, current_idx):
        """
        预测在当前索引处的交易信号
        """
        # 提取最近的模式
        recent_pattern = self.extract_recent_pattern(df, current_idx)
        if recent_pattern is None:
            return 0, 0.0  # 无操作或持仓，置信度0
        
        # 计算与各聚类模式的相似性
        best_cluster = None
        best_similarity = -1
        best_signal = 0
        best_confidence = 0
        
        for cluster_id, model in self.cluster_models.items():
            if model is None:
                continue
            
            # 计算与该聚类平均模式的相似性
            similarity = self.calculate_pattern_similarity(
                recent_pattern, 
                model['avg_pattern']
            )
            
            # 如果相似性更高，更新最佳匹配
            # 降低相似性阈值以增加匹配机会
            if similarity > best_similarity and similarity > 0.05:  # 从0.1降低到0.05
                best_similarity = similarity
                best_cluster = cluster_id
                
                # 根据聚类中最常见的信号类型进行预测
                cluster_info = self.patterns[cluster_id]
                signal_counts = cluster_info['signal_counts']
                
                # 预测信号类型（选择最常见的信号）
                if signal_counts:
                    predicted_signal = max(signal_counts, key=signal_counts.get)
                    best_signal = predicted_signal
                    # 调整置信度计算方式，增加信号密度的权重
                    best_confidence = similarity * (1 + 2 * cluster_info['signal_density'])
        
        # 添加置信度阈值过滤
        # 只有当置信度足够高时才返回预测信号，否则返回0（无操作）
        if best_confidence < 0.1:  # 添加置信度阈值
            return 0, 0.0
        
        return best_signal, best_confidence
    
    def visualize_predictions(self, df, predictions, output_path=None):
        """
        可视化预测结果，显示指数值曲线和交易信号
        
        Parameters:
        df: DataFrame - 包含测试数据的DataFrame
        predictions: list - 预测结果列表
        output_path: str - 输出图像文件路径，默认为None（显示图像而不保存）
        """
        logger.info("Generating visualization of predictions...")
        
        # 检查是否包含实际标签（用于区分回测和实时预测）
        has_actual_labels = 'actual_signal' in predictions[0] if predictions else False
        
        # 准备数据
        indices = [pred['index'] for pred in predictions]
        predicted_signals = [pred['predicted_signal'] for pred in predictions]
        index_values = [df.iloc[i]['index_value'] for i in indices]
        
        # 如果有实际标签，也准备实际标签数据
        actual_signals = [pred['actual_signal'] for pred in predictions] if has_actual_labels else None
        
        # 合并连续的同向开仓信号
        # 过滤预测信号，合并连续的同向开仓信号
        filtered_predictions = []
        last_long_position = False  # 是否处于做多持仓状态
        last_short_position = False  # 是否处于做空持仓状态
        
        for i, (idx, pred_signal) in enumerate(zip(indices, predicted_signals)):
            actual_signal = actual_signals[i] if actual_signals else None
            
            # 处理预测信号的合并逻辑
            if pred_signal == 1:  # 做多开仓
                if not last_long_position:  # 如果当前不是做多持仓状态
                    filtered_predictions.append({
                        'index': idx,
                        'predicted_signal': pred_signal,
                        'actual_signal': actual_signal,
                        'index_value': index_values[i]
                    })
                    last_long_position = True
                # 如果已经是做多持仓状态，则忽略这个做多开仓信号
            elif pred_signal == 3:  # 做空开仓
                if not last_short_position:  # 如果当前不是做空持仓状态
                    filtered_predictions.append({
                        'index': idx,
                        'predicted_signal': pred_signal,
                        'actual_signal': actual_signal,
                        'index_value': index_values[i]
                    })
                    last_short_position = True
                # 如果已经是做空持仓状态，则忽略这个做空开仓信号
            elif pred_signal == 2:  # 做多平仓
                filtered_predictions.append({
                    'index': idx,
                    'predicted_signal': pred_signal,
                    'actual_signal': actual_signal,
                    'index_value': index_values[i]
                })
                last_long_position = False  # 重置做多持仓状态
            elif pred_signal == 4:  # 做空平仓
                filtered_predictions.append({
                    'index': idx,
                    'predicted_signal': pred_signal,
                    'actual_signal': actual_signal,
                    'index_value': index_values[i]
                })
                last_short_position = False  # 重置做空持仓状态
            else:  # 无操作信号
                filtered_predictions.append({
                    'index': idx,
                    'predicted_signal': pred_signal,
                    'actual_signal': actual_signal,
                    'index_value': index_values[i]
                })
                # 保持当前持仓状态不变
        
        # 打印过滤后的信号分布，用于调试
        signal_counts = {}
        for pred in filtered_predictions:
            signal = pred['predicted_signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        logger.info(f"Filtered signal distribution: {signal_counts}")
        
        # 计算每日收益
        daily_profits = []
        long_positions = []   # 存储做多开仓信息
        short_positions = []  # 存储做空开仓信息
        
        for pred in filtered_predictions:
            idx = pred['index']
            pred_signal = pred['predicted_signal']
            index_val = pred['index_value']
            
            # 处理做多交易
            if pred_signal == 1:  # 做多开仓
                long_positions.append((idx, index_val))
            elif pred_signal == 2 and long_positions:  # 做多平仓
                # 计算收益：平仓值 - 开仓值（指数是反向的，所以是负收益）
                open_idx, open_value = long_positions.pop()
                profit = open_value - index_val  # 做多时，指数下降是盈利
                daily_profits.append((idx, profit, '做多', open_idx, open_value, index_val))
                
            # 处理做空交易
            elif pred_signal == 3:  # 做空开仓
                short_positions.append((idx, index_val))
            elif pred_signal == 4 and short_positions:  # 做空平仓
                # 计算收益：开仓值 - 平仓值（指数是反向的，所以是正收益）
                open_idx, open_value = short_positions.pop()
                profit = index_val - open_value  # 做空时，指数上升是盈利
                daily_profits.append((idx, profit, '做空', open_idx, open_value, index_val))
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # 绘制指数值曲线（上下翻转y值）
        y_values = np.array(index_values)
        y_flipped = -y_values  # 上下翻转y值
        ax1.plot(indices, y_flipped, 'b-', linewidth=1, label='指数值')
        ax1.set_xlabel('时间索引')
        ax1.set_ylabel('指数值 (翻转)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 标识过滤后的预测信号
        long_open_indices = []    # 做多开仓
        long_close_indices = []   # 做多平仓
        short_open_indices = []   # 做空开仓
        short_close_indices = []  # 做空平仓
        
        for pred in filtered_predictions:
            idx = pred['index']
            pred_signal = pred['predicted_signal']
            index_val = pred['index_value']
            
            if pred_signal == 1:  # 做多开仓
                long_open_indices.append((idx, -index_val))  # 上下翻转y值
            elif pred_signal == 2:  # 做多平仓
                long_close_indices.append((idx, -index_val))  # 上下翻转y值
            elif pred_signal == 3:  # 做空开仓
                short_open_indices.append((idx, -index_val))  # 上下翻转y值
            elif pred_signal == 4:  # 做空平仓
                short_close_indices.append((idx, -index_val))  # 上下翻转y值
        
        # 在图表上标识各种信号
        # 修正信号颜色：做多用绿色，做空用红色
        if long_open_indices:
            lo_idx, lo_val = zip(*long_open_indices)
            ax1.scatter(lo_idx, lo_val, color='green', marker='^', s=100, label='预测做多开仓', zorder=5)
            
        if long_close_indices:
            lc_idx, lc_val = zip(*long_close_indices)
            ax1.scatter(lc_idx, lc_val, color='green', marker='v', s=100, label='预测做多平仓', zorder=5)
            
        if short_open_indices:
            so_idx, so_val = zip(*short_open_indices)
            ax1.scatter(so_idx, so_val, color='red', marker='^', s=100, label='预测做空开仓', zorder=5)
            
        if short_close_indices:
            sc_idx, sc_val = zip(*short_close_indices)
            ax1.scatter(sc_idx, sc_val, color='red', marker='v', s=100, label='预测做空平仓', zorder=5)
        
        # 如果有实际标签，标识实际信号（使用不同的标记）
        if has_actual_labels and actual_signals:
            actual_long_open_indices = []
            actual_long_close_indices = []
            actual_short_open_indices = []
            actual_short_close_indices = []
            
            for i, (idx, actual_signal) in enumerate(zip(indices, actual_signals)):
                if actual_signal == 1:  # 做多开仓
                    actual_long_open_indices.append((idx, -index_values[i]))  # 上下翻转y值
                elif actual_signal == 2:  # 做多平仓
                    actual_long_close_indices.append((idx, -index_values[i]))  # 上下翻转y值
                elif actual_signal == 3:  # 做空开仓
                    actual_short_open_indices.append((idx, -index_values[i]))  # 上下翻转y值
                elif actual_signal == 4:  # 做空平仓
                    actual_short_close_indices.append((idx, -index_values[i]))  # 上下翻转y值
            
            # 在图表上标识实际信号（使用正确的颜色）
            if actual_long_open_indices:
                alo_idx, alo_val = zip(*actual_long_open_indices)
                ax1.scatter(alo_idx, alo_val, color='green', marker='^', s=50, alpha=0.5, label='实际做多开仓', zorder=4)
                
            if actual_long_close_indices:
                alc_idx, alc_val = zip(*actual_long_close_indices)
                ax1.scatter(alc_idx, alc_val, color='green', marker='v', s=50, alpha=0.5, label='实际做多平仓', zorder=4)
                
            if actual_short_open_indices:
                aso_idx, aso_val = zip(*actual_short_open_indices)
                ax1.scatter(aso_idx, aso_val, color='red', marker='^', s=50, alpha=0.5, label='实际做空开仓', zorder=4)
                
            if actual_short_close_indices:
                asc_idx, asc_val = zip(*actual_short_close_indices)
                ax1.scatter(asc_idx, asc_val, color='red', marker='v', s=50, alpha=0.5, label='实际做空平仓', zorder=4)
        
        # 添加每日收益信息到图表
        if daily_profits:
            # 计算累积收益
            cumulative_profit = 0
            profit_text = "每日收益:\n"
            for i, (idx, profit, trade_type, open_idx, open_value, close_value) in enumerate(daily_profits[:5]):  # 只显示前5个
                cumulative_profit += profit
                profit_text += f"{idx}: {profit:.2f} ({trade_type})\n"
            
            profit_text += f"总收益: {cumulative_profit:.2f}"
            
            # 在图表上添加文本框
            ax1.text(0.02, 0.98, profit_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加图例
        ax1.legend(loc='upper left')
        
        # 设置标题
        title = '模式预测结果可视化（合并连续开仓信号）' if has_actual_labels else '实时预测结果可视化（合并连续开仓信号）'
        plt.title(title, fontsize=16)
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        
        # 优化布局
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        logger.info("Visualization generation completed.")

    def time_series_cross_validate(self, df, n_splits=5):
        """
        使用时间序列交叉验证评估模型性能
        """
        logger.info(f"Running time series cross-validation with {n_splits} folds...")
        
        # 确定测试范围
        start_idx = max(PATTERN_LENGTH, 0)
        end_idx = len(df) - 1
        
        # 创建时间序列分割对象
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        accuracies = []
        signal_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(range(start_idx, end_idx))):
            logger.info(f"Processing fold {fold + 1}/{n_splits}...")
            
            # 确定当前折的测试范围
            fold_start = test_idx[0] if len(test_idx) > 0 else start_idx
            fold_end = test_idx[-1] if len(test_idx) > 0 else end_idx
            
            correct_predictions = 0
            total_predictions = 0
            signal_matches = 0
            signal_actual = 0
            
            for i in range(fold_start, min(fold_end, end_idx)):
                # 获取实际信号
                actual_signal = df.iloc[i]['label']
                
                # 进行预测
                predicted_signal, confidence = self.predict_signal(df, i)
                
                # 检查预测是否正确
                if predicted_signal == actual_signal:
                    correct_predictions += 1
                total_predictions += 1
                
                # 统计信号数量
                if actual_signal != 0:
                    signal_actual += 1
                    if predicted_signal == actual_signal:
                        signal_matches += 1
            
            # 计算准确率
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            signal_accuracy = signal_matches / signal_actual if signal_actual > 0 else 0
            
            accuracies.append(accuracy)
            signal_accuracies.append(signal_accuracy)
            
            logger.info(f"  Fold {fold + 1} Accuracy: {accuracy:.2%}")
            logger.info(f"  Fold {fold + 1} Signal Accuracy: {signal_accuracy:.2%}")
        
        # 计算平均准确率
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_signal_accuracy = np.mean(signal_accuracies)
        std_signal_accuracy = np.std(signal_accuracies)
        
        logger.info(f"Time series cross-validation Results:")
        logger.info(f"  Mean Overall Accuracy: {mean_accuracy:.2%} (+/- {std_accuracy:.2%})")
        logger.info(f"  Mean Signal Accuracy: {mean_signal_accuracy:.2%} (+/- {std_signal_accuracy:.2%})")
        
        return mean_accuracy, std_accuracy, mean_signal_accuracy, std_signal_accuracy
    
    def backtest_prediction(self, df, test_size=100):
        """
        对历史数据进行回测预测
        """
        logger.info("Running backtest prediction with balanced model...")
        
        # 确定测试范围
        start_idx = max(PATTERN_LENGTH, len(df) - test_size)
        end_idx = len(df) - 1
        
        predictions = []
        actual_signals = []
        correct_predictions = 0
        total_predictions = 0
        signal_predictions = 0  # 预测为信号的数量
        signal_actual = 0  # 实际为信号的数量
        
        for i in range(start_idx, end_idx):
            # 获取实际信号
            actual_signal = df.iloc[i]['label']
            actual_signals.append(actual_signal)
            
            # 进行预测
            predicted_signal, confidence = self.predict_signal(df, i)
            predictions.append({
                'index': i,
                'predicted_signal': predicted_signal,
                'actual_signal': actual_signal,
                'confidence': confidence
            })
            
            # 统计信号数量
            if predicted_signal != 0:
                signal_predictions += 1
            if actual_signal != 0:
                signal_actual += 1
            
            # 检查预测是否正确
            if predicted_signal == actual_signal:
                correct_predictions += 1
            total_predictions += 1
    
        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 计算信号预测的准确率
        signal_accuracy = 0
        if signal_actual > 0:
            signal_matches = 0
            for pred in predictions:
                if pred['actual_signal'] != 0 and pred['predicted_signal'] == pred['actual_signal']:
                    signal_matches += 1
            signal_accuracy = signal_matches / signal_actual if signal_actual > 0 else 0
    
        logger.info(f"Backtest Results:")
        logger.info(f"  Total predictions: {total_predictions}")
        logger.info(f"  Correct predictions: {correct_predictions}")
        logger.info(f"  Overall Accuracy: {accuracy:.2%}")
        logger.info(f"  Signal predictions: {signal_predictions}")
        logger.info(f"  Actual signals: {signal_actual}")
        logger.info(f"  Signal accuracy (when actual signal exists): {signal_accuracy:.2%}")
        
        return predictions, accuracy
    
    def predict_realtime_signal(self, df):
        """
        实时预测函数，不需要标签数据
        预测数据集最后一个点的信号
        """
        logger.info("Running real-time prediction...")
        
        # 预测最后一个点的信号
        current_idx = len(df) - 1
        predicted_signal, confidence = self.predict_signal(df, current_idx)
        
        logger.info(f"Real-time Prediction Results:")
        logger.info(f"  Predicted Signal: {predicted_signal}")
        logger.info(f"  Confidence: {confidence:.4f}")
        
        return predicted_signal, confidence

    def predict_realtime_sequence(self, df, sequence_length=100):
        """
        实时预测函数，不需要标签数据
        预测数据集最后sequence_length个点的信号序列
        """
        logger.info(f"Running real-time sequence prediction for last {sequence_length} points...")
        
        # 确定预测范围
        start_idx = max(PATTERN_LENGTH, len(df) - sequence_length)
        end_idx = len(df) - 1
        
        predictions = []
        for i in range(start_idx, end_idx + 1):
            # 进行预测
            predicted_signal, confidence = self.predict_signal(df, i)
            predictions.append({
                'index': i,
                'predicted_signal': predicted_signal,
                'confidence': confidence
            })
        
        logger.info(f"Real-time sequence prediction completed for {len(predictions)} points")
        return predictions

    def predict_future_signal(self, df, steps_ahead=1):
        """
        预测未来的交易信号
        """
        current_idx = len(df) - 1 + steps_ahead
        predicted_signal, confidence = self.predict_signal(df, current_idx - steps_ahead)
        return predicted_signal, confidence
    
    def predict_signal_with_rl(self, df, current_idx):
        """
        使用强化学习优化的预测信号
        """
        # 首先获取基础预测
        predicted_signal, confidence = self.predict_signal(df, current_idx)
        
        # 如果没有强化学习模型，直接返回基础预测
        if self.rl_trader is None:
            return predicted_signal, confidence
        
        # 使用强化学习模型决定是否执行该信号
        action = self.rl_trader.predict(predicted_signal)
        
        # 如果强化学习模型建议忽略信号，则返回无操作
        if action == 0:  # 忽略信号
            return 0, 0.0
        else:  # 执行信号
            return predicted_signal, confidence

    def train_rl_model(self, training_data_file):
        """
        训练强化学习模型
        """
        logger.info("Training reinforcement learning model...")
        
        # 加载带预测信号的数据
        from simple_rl_trader import load_data_with_predictions
        train_data = load_data_with_predictions(training_data_file)
        if train_data is None:
            logger.error("Failed to load training data for RL model")
            return False
        
        # 创建并训练强化学习交易器
        self.rl_trader = SimpleRLTrader(learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
        self.rl_trader.train(train_data, episodes=50)
        
        logger.info("Reinforcement learning model training completed")
        return True

    def save_rl_model(self, model_path=None):
        """
        保存强化学习模型
        """
        if self.rl_trader is None:
            logger.warning("No RL model to save")
            return False
            
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "rl_trader_model.json")
        
        try:
            # 将Q表转换为可序列化的格式
            q_table_serializable = {}
            for state, values in self.rl_trader.q_table.items():
                q_table_serializable[state] = values.tolist()
            
            model_data = {
                'q_table': q_table_serializable,
                'learning_rate': self.rl_trader.learning_rate,
                'discount_factor': self.rl_trader.discount_factor,
                'epsilon': self.rl_trader.epsilon
            }
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            logger.info(f"RL model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
            return False

    def load_rl_model(self, model_path=None):
        """
        加载强化学习模型
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "rl_trader_model.json")
        
        if not os.path.exists(model_path):
            logger.warning(f"RL model file not found: {model_path}")
            return False
            
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # 创建新的强化学习交易器
            self.rl_trader = SimpleRLTrader(
                learning_rate=model_data['learning_rate'],
                discount_factor=model_data['discount_factor'],
                epsilon=model_data['epsilon']
            )
            
            # 恢复Q表
            for state, values in model_data['q_table'].items():
                self.rl_trader.q_table[state] = np.array(values)
            
            logger.info(f"RL model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False

    def save_model(self):
        """
        保存模型参数
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_data = {
            'cluster_models': {},
            'patterns_info': {}
        }
        
        # 保存聚类模型信息
        for cluster_id, model in self.cluster_models.items():
            if model is not None:
                model_data['cluster_models'][cluster_id] = {
                    'avg_pattern': model['avg_pattern'].tolist(),
                    'std_pattern': model['std_pattern'].tolist(),
                    'pattern_length': model['pattern_length']
                }
        
        # 保存模式信息
        for cluster_id, info in self.patterns.items():
            model_data['patterns_info'][cluster_id] = {
                'signal_density': info['signal_density'],
                'signal_counts': info['signal_counts'],
                'long_pairs': info['long_pairs'],
                'short_pairs': info['short_pairs']
            }
        
        model_file_path = os.path.join(MODEL_DIR, "balanced_pattern_predictor_model.json")
        try:
            with open(model_file_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Balanced model saved to {model_file_path}")
        except Exception as e:
            logger.error(f"Error saving balanced model: {e}")
    
    def load_model(self):
        """
        加载模型参数
        """
        model_file_path = os.path.join(MODEL_DIR, "balanced_pattern_predictor_model.json")
        if not os.path.exists(model_file_path):
            logger.warning("No saved balanced model found.")
            return False
            
        try:
            with open(model_file_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # 加载聚类模型
            for cluster_id, model_info in model_data['cluster_models'].items():
                self.cluster_models[int(cluster_id)] = {
                    'avg_pattern': np.array(model_info['avg_pattern']),
                    'std_pattern': np.array(model_info['std_pattern']),
                    'pattern_length': model_info['pattern_length']
                }
            
            # 加载模式信息
            for cluster_id, info in model_data['patterns_info'].items():
                self.patterns[int(cluster_id)] = {
                    'signal_density': info['signal_density'],
                    'signal_counts': info['signal_counts'],
                    'long_pairs': info['long_pairs'],
                    'short_pairs': info['short_pairs'],
                    'patterns': []  # 模式数据需要重新加载
                }
            
            logger.info("Balanced model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading balanced model: {e}")
            return False

def load_test_data(file_path):
    """
    加载测试数据
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded test data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading test data from {file_path}: {e}")
        return None

def load_realtime_data(file_path):
    """
    加载实时数据（不需要标签列）
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded real-time data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading real-time data from {file_path}: {e}")
        return None

def main():
    """
    主函数
    """
    import sys
    
    # 检查命令行参数，看是否要运行实时预测
    if len(sys.argv) > 1 and sys.argv[1] == '--realtime':
        realtime_prediction_main()
        return
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 获取测试数据文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    logger.info(f"Found {len(label_files)} label files")
    if not label_files:
        logger.error("No label files found!")
        return
    
    # 训练强化学习模型（使用第一个文件）
    if len(label_files) > 0:
        logger.info("Training reinforcement learning model...")
        predictor.train_rl_model(label_files[0])
        # 保存强化学习模型
        predictor.save_rl_model()
    else:
        # 如果没有数据用于训练，尝试加载已保存的模型
        logger.info("Attempting to load existing RL model...")
        predictor.load_rl_model()
    
    # 为每个文件进行预测和可视化
    all_accuracies = []
    for i, test_file in enumerate(label_files):
        logger.info(f"\nProcessing file {i+1}/{len(label_files)}: {test_file}")
        
        # 加载测试数据
        df = load_test_data(test_file)
        if df is None:
            logger.error(f"Failed to load test data from {test_file}")
            continue
        
        # 进行回测预测（使用强化学习优化的预测）
        predictions = []
        actual_signals = []
        correct_predictions = 0
        total_predictions = 0
        signal_predictions = 0  # 预测为信号的数量
        signal_actual = 0  # 实际为信号的数量
        
        # 确定测试范围
        start_idx = max(PATTERN_LENGTH, len(df) - 100)
        end_idx = len(df) - 1
        
        for idx in range(start_idx, end_idx):
            # 获取实际信号
            actual_signal = df.iloc[idx]['label']
            actual_signals.append(actual_signal)
            
            # 使用强化学习优化的预测
            predicted_signal, confidence = predictor.predict_signal_with_rl(df, idx)
            predictions.append({
                'index': idx,
                'predicted_signal': predicted_signal,
                'actual_signal': actual_signal,
                'confidence': confidence
            })
            
            # 统计信号数量
            if predicted_signal != 0:
                signal_predictions += 1
            if actual_signal != 0:
                signal_actual += 1
            
            # 检查预测是否正确
            if predicted_signal == actual_signal:
                correct_predictions += 1
            total_predictions += 1

        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        all_accuracies.append(accuracy)
        
        # 计算信号预测的准确率
        signal_accuracy = 0
        if signal_actual > 0:
            signal_matches = 0
            for pred in predictions:
                if pred['actual_signal'] != 0 and pred['predicted_signal'] == pred['actual_signal']:
                    signal_matches += 1
            signal_accuracy = signal_matches / signal_actual if signal_actual > 0 else 0

        logger.info(f"Prediction Results for {os.path.basename(test_file)}:")
        logger.info(f"  Total predictions: {total_predictions}")
        logger.info(f"  Correct predictions: {correct_predictions}")
        logger.info(f"  Overall Accuracy: {accuracy:.2%}")
        logger.info(f"  Signal predictions: {signal_predictions}")
        logger.info(f"  Actual signals: {signal_actual}")
        logger.info(f"  Signal accuracy (when actual signal exists): {signal_accuracy:.2%}")
        
        # 对第一个文件进行时间序列交叉验证
        if i == 0:
            logger.info("Running time series cross-validation on first file...")
            mean_acc, std_acc, mean_signal_acc, std_signal_acc = predictor.time_series_cross_validate(df, n_splits=5)
            logger.info(f"Cross-validation results for {os.path.basename(test_file)}:")
            logger.info(f"  Mean Overall Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
            logger.info(f"  Mean Signal Accuracy: {mean_signal_acc:.2%} (+/- {std_signal_acc:.2%})")
        
        # 保存模型（对所有文件都执行）
        predictor.save_model()
        
        # 生成可视化结果
        output_dir = os.path.join(CURRENT_DIR, "..", "visualization/")
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用文件名作为输出文件名
        file_name = os.path.splitext(os.path.basename(test_file))[0]
        output_path = os.path.join(output_dir, f"prediction_visualization_{file_name}.png")
        predictor.visualize_predictions(df, predictions, output_path)
        
        logger.info(f"Visualization saved to {output_path}")
    
    # 显示总体统计信息
    if all_accuracies:
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        logger.info(f"\nCompleted processing {len(label_files)} files")
        logger.info(f"Overall Mean Accuracy: {mean_accuracy:.2%} (+/- {std_accuracy:.2%})")
    else:
        logger.info(f"\nCompleted processing {len(label_files)} files")

def realtime_prediction_main():
    """
    实时预测主函数
    """
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 获取实时数据文件
    data_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))  # 可以修改为其他目录
    logger.info(f"Found {len(data_files)} data files for real-time prediction")
    if not data_files:
        logger.error("No data files found!")
        return
    
    # 为每个文件进行实时预测
    for i, data_file in enumerate(data_files):
        logger.info(f"\nProcessing real-time prediction for file {i+1}/{len(data_files)}: {data_file}")
        
        # 加载数据
        df = load_realtime_data(data_file)
        if df is None:
            logger.error(f"Failed to load data from {data_file}")
            continue
        
        # 进行实时预测（最后一个点）
        predicted_signal, confidence = predictor.predict_realtime_signal(df)
        
        # 进行序列预测（最后100个点）
        sequence_predictions = predictor.predict_realtime_sequence(df, sequence_length=100)
        
        # 保存预测结果
        output_dir = os.path.join(CURRENT_DIR, "..", "predictions/")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存单点预测结果
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        single_prediction_path = os.path.join(output_dir, f"realtime_prediction_{file_name}.json")
        
        single_result = {
            'file': data_file,
            'predicted_signal': int(predicted_signal),
            'confidence': float(confidence),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with open(single_prediction_path, 'w', encoding='utf-8') as f:
                json.dump(single_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Single point prediction saved to {single_prediction_path}")
        except Exception as e:
            logger.error(f"Error saving single point prediction: {e}")
        
        # 保存序列预测结果
        sequence_prediction_path = os.path.join(output_dir, f"realtime_sequence_prediction_{file_name}.json")
        sequence_results = {
            'file': data_file,
            'predictions': sequence_predictions,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with open(sequence_prediction_path, 'w', encoding='utf-8') as f:
                json.dump(sequence_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Sequence prediction saved to {sequence_prediction_path}")
        except Exception as e:
            logger.error(f"Error saving sequence prediction: {e}")
        
        # 生成可视化结果
        output_dir = os.path.join(CURRENT_DIR, "..", "visualization/")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"realtime_prediction_{file_name}.png")
        predictor.visualize_predictions(df, sequence_predictions, output_path)
        
        logger.info(f"Visualization saved to {output_path}")
    
    logger.info(f"\nCompleted real-time prediction for {len(data_files)} files")

if __name__ == "__main__":
    main()