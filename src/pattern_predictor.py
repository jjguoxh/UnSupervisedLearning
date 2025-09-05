# -*- coding: utf-8 -*-
"""
模式预测程序
基于历史模式识别结果进行交易信号预测
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
PATTERNS_DIR = "../patterns/"  # 模式数据目录
MODEL_DIR = "../model/"  # 模型保存目录
PATTERN_LENGTH = 10  # 模式长度

class PatternPredictor:
    def __init__(self):
        self.patterns = {}
        self.cluster_models = {}
        self.thresholds = {}
        self.load_patterns()
        
    def load_patterns(self):
        """
        加载已学习的模式
        """
        print("Loading learned patterns...")
        
        # 加载聚类分析结果
        cluster_analysis_path = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
        if not os.path.exists(cluster_analysis_path):
            print("Error: Cluster analysis file not found!")
            return
            
        cluster_df = pd.read_csv(cluster_analysis_path)
        
        # 加载每个聚类的模式数据
        for _, row in cluster_df.iterrows():
            cluster_id = row['cluster_id']
            signal_density = row['signal_density']
            
            # 加载聚类目录中的模式
            cluster_dir = os.path.join(PATTERNS_DIR, f"cluster_{cluster_id}")
            if not os.path.exists(cluster_dir):
                continue
                
            # 加载该聚类的模式文件
            pattern_files = glob.glob(os.path.join(cluster_dir, "pattern_*.csv"))
            patterns = []
            
            for pattern_file in pattern_files:
                try:
                    pattern_data = pd.read_csv(pattern_file)
                    patterns.append(pattern_data)
                except Exception as e:
                    print(f"Error loading pattern {pattern_file}: {e}")
                    continue
            
            if patterns:
                self.patterns[cluster_id] = {
                    'patterns': patterns,
                    'signal_density': signal_density,
                    'signal_counts': eval(str(row['signal_counts'])) if isinstance(row['signal_counts'], str) else row['signal_counts'],
                    'long_pairs': row['long_pairs'],
                    'short_pairs': row['short_pairs']
                }
                
                # 为高盈利聚类创建预测模型
                if signal_density > 0.3:  # 只为信号密度较高的聚类创建模型
                    self.cluster_models[cluster_id] = self.create_cluster_model(patterns)
        
        print(f"Loaded {len(self.patterns)} clusters, {len(self.cluster_models)} predictive models")
    
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
        
        avg_pattern = np.mean(padded_values, axis=0)
        std_pattern = np.std(padded_values, axis=0)
        
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
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def predict_signal(self, df, current_idx):
        """
        预测在当前索引处的交易信号
        """
        # 提取最近的模式
        recent_pattern = self.extract_recent_pattern(df, current_idx)
        if recent_pattern is None:
            return 0, 0.0  # 无操作，置信度0
        
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
            if similarity > best_similarity and similarity > 0.7:  # 相似性阈值
                best_similarity = similarity
                best_cluster = cluster_id
                
                # 根据聚类中最常见的信号类型进行预测
                cluster_info = self.patterns[cluster_id]
                signal_counts = cluster_info['signal_counts']
                
                # 预测信号类型（选择最常见的信号）
                if signal_counts:
                    predicted_signal = max(signal_counts, key=signal_counts.get)
                    best_signal = predicted_signal
                    best_confidence = similarity * cluster_info['signal_density']
        
        return best_signal, best_confidence
    
    def backtest_prediction(self, df, test_size=100):
        """
        对历史数据进行回测预测
        """
        print("Running backtest prediction...")
        
        # 确定测试范围
        start_idx = max(PATTERN_LENGTH, len(df) - test_size)
        end_idx = len(df) - 1
        
        predictions = []
        actual_signals = []
        correct_predictions = 0
        total_predictions = 0
        
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
            
            # 检查预测是否正确（只检查非0信号）
            if predicted_signal != 0 and actual_signal != 0:
                total_predictions += 1
                if predicted_signal == actual_signal:
                    correct_predictions += 1
            elif predicted_signal == 0 and actual_signal == 0:
                total_predictions += 1
                correct_predictions += 1
        
        # 计算准确率
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Backtest Results:")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Accuracy: {accuracy:.2%}")
        
        return predictions, accuracy
    
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
        
        with open(os.path.join(MODEL_DIR, "pattern_predictor_model.json"), 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {MODEL_DIR}")
    
    def load_model(self):
        """
        加载模型参数
        """
        model_path = os.path.join(MODEL_DIR, "pattern_predictor_model.json")
        if not os.path.exists(model_path):
            print("No saved model found.")
            return False
            
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
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
            
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def load_test_data(file_path):
    """
    加载测试数据
    """
    df = pd.read_csv(file_path)
    return df

def main():
    """
    主函数
    """
    # 创建预测器
    predictor = PatternPredictor()
    
    # 获取测试数据文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    if not label_files:
        print("No label files found!")
        return
    
    # 使用最后一个文件作为测试数据
    test_file = label_files[-1]
    print(f"Using {test_file} as test data")
    
    # 加载测试数据
    df = load_test_data(test_file)
    
    # 进行回测预测
    predictions, accuracy = predictor.backtest_prediction(df, test_size=50)
    
    # 显示最近的几个预测结果
    print("\nRecent Predictions:")
    print("=" * 50)
    for pred in predictions[-10:]:  # 显示最近10个预测
        signal_names = {
            0: "无操作",
            1: "做多开仓",
            2: "做多平仓",
            3: "做空开仓",
            4: "做空平仓"
        }
        
        print(f"Index {pred['index']}: Predicted={signal_names.get(pred['predicted_signal'], 'Unknown')} "
              f"({pred['predicted_signal']}), Actual={signal_names.get(pred['actual_signal'], 'Unknown')} "
              f"({pred['actual_signal']}), Confidence={pred['confidence']:.3f}")
    
    # 预测下一个信号
    next_signal, confidence = predictor.predict_future_signal(df, steps_ahead=1)
    signal_names = {
        0: "无操作",
        1: "做多开仓",
        2: "做多平仓",
        3: "做空开仓",
        4: "做空平仓"
    }
    
    print(f"\nNext Signal Prediction:")
    print(f"  Predicted Signal: {signal_names.get(next_signal, 'Unknown')} ({next_signal})")
    print(f"  Confidence: {confidence:.3f}")
    
    # 保存模型
    predictor.save_model()

if __name__ == "__main__":
    main()