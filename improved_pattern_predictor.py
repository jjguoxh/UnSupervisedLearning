# -*- coding: utf-8 -*-
"""
改进的交易模式预测器
基于改进的模式识别结果，提供更准确的预测
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import talib
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
PATTERNS_DIR = "./patterns_improved/"  # 改进的模式目录
MODELS_DIR = "./models_improved/"  # 模型保存目录
PATTERN_LENGTH = 15  # 模式长度
MIN_CONFIDENCE = 0.6  # 最小置信度阈值
TOP_CLUSTERS = 10  # 使用前N个高质量聚类

class ImprovedPatternPredictor:
    def __init__(self):
        self.cluster_models = {}  # 每个聚类的预测模型
        self.cluster_info = {}    # 聚类信息
        self.feature_scaler = StandardScaler()
        self.pattern_cache = {}   # 模式缓存
        
    def load_cluster_patterns(self):
        """
        加载聚类模式数据
        """
        # 读取聚类分析结果
        analysis_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
        if not os.path.exists(analysis_file):
            raise FileNotFoundError(f"Cluster analysis file not found: {analysis_file}")
        
        cluster_analysis = pd.read_csv(analysis_file)
        print(f"Loaded {len(cluster_analysis)} clusters")
        
        # 选择高质量聚类
        high_quality_clusters = cluster_analysis[
            cluster_analysis['quality_score'] > 0.3
        ].head(TOP_CLUSTERS)
        
        print(f"Selected {len(high_quality_clusters)} high-quality clusters")
        
        # 加载每个聚类的详细信息
        for _, cluster_row in high_quality_clusters.iterrows():
            cluster_id = cluster_row['cluster_id']
            cluster_dir = os.path.join(PATTERNS_DIR, f"cluster_{cluster_id}")
            
            if os.path.exists(cluster_dir):
                self.cluster_info[cluster_id] = {
                    'quality_score': cluster_row['quality_score'],
                    'signal_density': cluster_row['signal_density'],
                    'cluster_size': cluster_row['cluster_size'],
                    'signal_counts': eval(cluster_row['signal_counts']),  # 转换字符串为字典
                    'long_pairs': cluster_row['long_pairs'],
                    'short_pairs': cluster_row['short_pairs']
                }
                
                # 加载模式文件（如果存在）
                patterns_file = os.path.join(cluster_dir, 'patterns.csv')
                if os.path.exists(patterns_file):
                    patterns_df = pd.read_csv(patterns_file)
                    self.pattern_cache[cluster_id] = patterns_df
        
        print(f"Loaded patterns for {len(self.cluster_info)} clusters")
        return self.cluster_info
    
    def extract_prediction_features(self, df, current_idx, window_size=PATTERN_LENGTH):
        """
        提取用于预测的特征（与模式识别保持一致）
        """
        if current_idx < window_size:
            return None
        
        # 获取历史数据窗口
        start_idx = current_idx - window_size
        end_idx = current_idx
        window_data = df.iloc[start_idx:end_idx]
        
        # 基础价格特征
        prices = window_data['index_value'].values.astype(float)
        
        # 计算技术指标
        try:
            # 移动平均线
            sma_5 = talib.SMA(prices, timeperiod=min(5, len(prices)))
            sma_10 = talib.SMA(prices, timeperiod=min(10, len(prices)))
            
            # RSI
            rsi = talib.RSI(prices, timeperiod=min(14, len(prices)))
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(prices)
            
            # 价格特征
            price_features = {
                'price_current': prices[-1],
                'price_change': (prices[-1] - prices[0]) / prices[0],
                'price_volatility': np.std(prices),
                'price_trend': np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 1 else 0,
                'price_momentum': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            }
            
            # 技术指标特征
            tech_features = {
                'sma_5': sma_5[-1] if not np.isnan(sma_5[-1]) else prices[-1],
                'sma_10': sma_10[-1] if not np.isnan(sma_10[-1]) else prices[-1],
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
            }
            
            # 影响因子特征
            factor_features = {}
            for factor in ['a', 'b', 'c', 'd']:
                if factor in window_data.columns:
                    factor_values = window_data[factor].values
                    factor_features.update({
                        f'{factor}_current': factor_values[-1],
                        f'{factor}_mean': np.mean(factor_values),
                        f'{factor}_std': np.std(factor_values),
                        f'{factor}_trend': np.polyfit(range(len(factor_values)), factor_values, 1)[0] if len(factor_values) > 1 else 0,
                    })
            
            # 合并所有特征
            all_features = {**price_features, **tech_features, **factor_features}
            
            # 转换为特征向量
            feature_vector = list(all_features.values())
            
            return feature_vector, all_features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_pattern_similarity_enhanced(self, current_features, cluster_id):
        """
        计算与聚类模式的增强相似性
        """
        if cluster_id not in self.cluster_info:
            return 0
        
        # 基于聚类质量的基础置信度
        base_confidence = self.cluster_info[cluster_id]['quality_score']
        
        # 基于信号密度的调整
        signal_density = self.cluster_info[cluster_id]['signal_density']
        density_bonus = min(signal_density * 0.5, 0.3)  # 最多增加0.3
        
        # 基于聚类大小的调整（更大的聚类更可靠）
        cluster_size = self.cluster_info[cluster_id]['cluster_size']
        size_bonus = min(np.log(cluster_size) * 0.1, 0.2)  # 最多增加0.2
        
        # 综合置信度
        total_confidence = base_confidence + density_bonus + size_bonus
        total_confidence = min(total_confidence, 1.0)  # 限制在1.0以内
        
        return total_confidence
    
    def predict_signal_enhanced(self, df, current_idx):
        """
        增强的信号预测
        """
        # 提取当前特征
        feature_result = self.extract_prediction_features(df, current_idx)
        if feature_result is None:
            return None
        
        current_features, feature_dict = feature_result
        
        # 对每个高质量聚类计算预测
        predictions = []
        
        for cluster_id in self.cluster_info.keys():
            # 计算相似性置信度
            confidence = self.calculate_pattern_similarity_enhanced(current_features, cluster_id)
            
            if confidence >= MIN_CONFIDENCE:
                cluster_info = self.cluster_info[cluster_id]
                signal_counts = cluster_info['signal_counts']
                
                # 基于历史信号分布预测最可能的信号
                most_common_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
                signal_probability = signal_counts[most_common_signal] / sum(signal_counts.values())
                
                # 调整置信度
                adjusted_confidence = confidence * signal_probability
                
                prediction = {
                    'cluster_id': cluster_id,
                    'predicted_signal': most_common_signal,
                    'confidence': adjusted_confidence,
                    'signal_probability': signal_probability,
                    'cluster_quality': cluster_info['quality_score'],
                    'signal_distribution': signal_counts
                }
                
                predictions.append(prediction)
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def backtest_enhanced_prediction(self, test_files=None, n_files=5):
        """
        增强的回测功能
        """
        if test_files is None:
            # 获取测试文件
            label_files = sorted(glob.glob("./label/*.csv"))
            test_files = label_files[-n_files:]  # 使用最后几个文件作为测试
        
        print(f"Starting enhanced backtest with {len(test_files)} files...")
        
        all_predictions = []
        all_actuals = []
        prediction_details = []
        
        for file_path in test_files:
            print(f"Testing on {os.path.basename(file_path)}...")
            
            try:
                df = pd.read_csv(file_path)
                
                # 对每个时间点进行预测
                for i in range(PATTERN_LENGTH, len(df) - 1):
                    # 获取预测
                    predictions = self.predict_signal_enhanced(df, i)
                    
                    if predictions:
                        best_prediction = predictions[0]  # 取置信度最高的预测
                        
                        # 获取实际标签（下一个时间点）
                        actual_label = df['label'].iloc[i + 1]
                        
                        if actual_label in [1, 2, 3, 4]:  # 只考虑交易信号
                            all_predictions.append(best_prediction['predicted_signal'])
                            all_actuals.append(actual_label)
                            
                            prediction_details.append({
                                'file': os.path.basename(file_path),
                                'index': i,
                                'predicted': best_prediction['predicted_signal'],
                                'actual': actual_label,
                                'confidence': best_prediction['confidence'],
                                'cluster_id': best_prediction['cluster_id'],
                                'correct': best_prediction['predicted_signal'] == actual_label
                            })
                            
            except Exception as e:
                print(f"Error testing {file_path}: {e}")
                continue
        
        # 计算性能指标
        if all_predictions:
            accuracy = sum(1 for p, a in zip(all_predictions, all_actuals) if p == a) / len(all_predictions)
            
            print(f"\n=== Enhanced Backtest Results ===")
            print(f"Total predictions: {len(all_predictions)}")
            print(f"Overall accuracy: {accuracy:.4f}")
            
            # 按信号类型分析
            signal_performance = {}
            for signal in [1, 2, 3, 4]:
                signal_preds = [p for p, a in zip(all_predictions, all_actuals) if a == signal]
                signal_actuals = [a for p, a in zip(all_predictions, all_actuals) if a == signal]
                
                if signal_preds:
                    signal_acc = sum(1 for p, a in zip(signal_preds, signal_actuals) if p == a) / len(signal_preds)
                    signal_performance[signal] = {
                        'count': len(signal_preds),
                        'accuracy': signal_acc
                    }
            
            print("\nSignal-wise performance:")
            for signal, perf in signal_performance.items():
                signal_name = {1: 'Long Open', 2: 'Long Close', 3: 'Short Open', 4: 'Short Close'}[signal]
                print(f"  {signal_name}: {perf['accuracy']:.4f} ({perf['count']} samples)")
            
            # 按置信度分析
            high_conf_details = [d for d in prediction_details if d['confidence'] > 0.8]
            if high_conf_details:
                high_conf_acc = sum(1 for d in high_conf_details if d['correct']) / len(high_conf_details)
                print(f"\nHigh confidence predictions (>0.8): {high_conf_acc:.4f} ({len(high_conf_details)} samples)")
            
            # 保存详细结果
            results_df = pd.DataFrame(prediction_details)
            results_file = os.path.join(MODELS_DIR, "backtest_results.csv")
            os.makedirs(MODELS_DIR, exist_ok=True)
            results_df.to_csv(results_file, index=False)
            print(f"\nDetailed results saved to: {results_file}")
            
            return accuracy, signal_performance, prediction_details
        
        else:
            print("No valid predictions made!")
            return 0, {}, []
    
    def generate_trading_signals(self, df, start_idx=None, end_idx=None):
        """
        为给定数据生成交易信号
        """
        if start_idx is None:
            start_idx = PATTERN_LENGTH
        if end_idx is None:
            end_idx = len(df)
        
        signals = []
        
        for i in range(start_idx, end_idx):
            predictions = self.predict_signal_enhanced(df, i)
            
            if predictions:
                best_prediction = predictions[0]
                
                if best_prediction['confidence'] >= MIN_CONFIDENCE:
                    signal = {
                        'index': i,
                        'timestamp': df['x'].iloc[i] if 'x' in df.columns else i,
                        'price': df['index_value'].iloc[i],
                        'predicted_signal': best_prediction['predicted_signal'],
                        'confidence': best_prediction['confidence'],
                        'cluster_id': best_prediction['cluster_id']
                    }
                    signals.append(signal)
        
        return pd.DataFrame(signals)
    
    def run_prediction_analysis(self):
        """
        运行完整的预测分析
        """
        print("Starting Enhanced Pattern Prediction Analysis...")
        
        # 加载聚类模式
        self.load_cluster_patterns()
        
        if not self.cluster_info:
            print("No cluster patterns loaded!")
            return
        
        # 运行回测
        accuracy, signal_perf, details = self.backtest_enhanced_prediction()
        
        # 生成示例交易信号
        print("\n=== Generating Sample Trading Signals ===")
        
        # 使用最新的测试文件
        test_files = sorted(glob.glob("./label/*.csv"))[-2:]
        
        for file_path in test_files:
            print(f"\nGenerating signals for {os.path.basename(file_path)}...")
            
            try:
                df = pd.read_csv(file_path)
                signals_df = self.generate_trading_signals(df)
                
                if not signals_df.empty:
                    print(f"Generated {len(signals_df)} high-confidence signals")
                    print("Sample signals:")
                    print(signals_df.head(10).to_string(index=False))
                    
                    # 保存信号
                    output_file = os.path.join(MODELS_DIR, f"signals_{os.path.basename(file_path)}")
                    signals_df.to_csv(output_file, index=False)
                    print(f"Signals saved to: {output_file}")
                else:
                    print("No high-confidence signals generated")
                    
            except Exception as e:
                print(f"Error generating signals for {file_path}: {e}")
        
        return accuracy, signal_perf

def main():
    """
    主函数
    """
    # 创建改进的预测器
    predictor = ImprovedPatternPredictor()
    
    # 运行预测分析
    accuracy, signal_perf = predictor.run_prediction_analysis()
    
    print(f"\n=== Final Results ===")
    print(f"Overall prediction accuracy: {accuracy:.4f}")
    
    if accuracy > 0.6:
        print("✅ Good prediction performance!")
    elif accuracy > 0.4:
        print("⚠️  Moderate prediction performance")
    else:
        print("❌ Poor prediction performance - needs further improvement")

if __name__ == "__main__":
    main()