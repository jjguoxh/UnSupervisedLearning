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
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "..", "predict/")  # 标签数据目录
PATTERNS_DIR = os.path.join(CURRENT_DIR, "..", "patterns/")  # 模式数据目录
STRICT_BALANCED_DIR = os.path.join(CURRENT_DIR, "..", "patterns/strict_balanced/")  # 严格平衡后的数据目录
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "model/balanced_model/")  # 平衡模型保存目录
PATTERN_LENGTH = 10  # 模式长度

class BalancedPatternPredictor:
    def __init__(self):
        self.patterns = {}
        self.cluster_models = {}
        self.thresholds = {}
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
        for pattern in patterns:
            if 'index_value' in pattern.columns:
                index_values_list.append(pattern['index_value'].values)
        
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
        
        return {
            'index_value': pattern_data['index_value'].values,
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
        if len(pattern1) != len(pattern2):
            return 0
            
        # 使用皮尔逊相关系数计算相似性
        try:
            correlation = np.corrcoef(pattern1, pattern2)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0
    
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
                recent_pattern['index_value'], 
                model['avg_pattern']
            )
            
            # 如果相似性更高，更新最佳匹配
            if similarity > best_similarity and similarity > 0.1:  # 保持原来的相似性阈值
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
    
    # 为每个文件进行预测和可视化
    for i, test_file in enumerate(label_files):
        logger.info(f"\nProcessing file {i+1}/{len(label_files)}: {test_file}")
        
        # 加载测试数据
        df = load_test_data(test_file)
        if df is None:
            logger.error(f"Failed to load test data from {test_file}")
            continue
        
        # 进行回测预测
        predictions, accuracy = predictor.backtest_prediction(df, test_size=100)
        
        # 显示预测结果摘要
        logger.info(f"Prediction Results for {os.path.basename(test_file)}:")
        logger.info(f"  Overall Accuracy: {accuracy:.2%}")
        
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