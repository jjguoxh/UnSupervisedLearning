# -*- coding: utf-8 -*-
"""
增强信号多样性的改进模型
解决模型只输出单一信号类型的问题
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import talib
import warnings
warnings.filterwarnings('ignore')

class EnhancedSignalDiversityPredictor:
    def __init__(self):
        self.patterns_dir = "./patterns_improved/"
        self.models_dir = "./models_enhanced/"
        self.cluster_info = {}
        self.signal_thresholds = {
            1: 0.3,  # 做多开仓阈值
            2: 0.35, # 做多平仓阈值
            3: 0.3,  # 做空开仓阈值
            4: 0.35  # 做空平仓阈值
        }
        
    def load_and_rebalance_clusters(self):
        """
        加载聚类并重新平衡信号分布
        """
        analysis_file = os.path.join(self.patterns_dir, "cluster_analysis.csv")
        if not os.path.exists(analysis_file):
            raise FileNotFoundError(f"聚类分析文件未找到: {analysis_file}")
        
        cluster_analysis = pd.read_csv(analysis_file)
        print(f"加载了 {len(cluster_analysis)} 个聚类")
        
        # 分析每个聚类的信号分布
        signal_distribution = {1: [], 2: [], 3: [], 4: []}
        
        for _, cluster_row in cluster_analysis.iterrows():
            cluster_id = cluster_row['cluster_id']
            signal_counts = eval(cluster_row['signal_counts'])
            
            # 计算每种信号的相对强度
            total_signals = sum(signal_counts.values())
            if total_signals > 0:
                for signal_type in [1, 2, 3, 4]:
                    signal_strength = signal_counts.get(signal_type, 0) / total_signals
                    if signal_strength > 0.1:  # 只考虑有一定强度的信号
                        signal_distribution[signal_type].append({
                            'cluster_id': cluster_id,
                            'strength': signal_strength,
                            'quality': cluster_row['quality_score'],
                            'size': cluster_row['cluster_size']
                        })
        
        # 为每种信号类型选择最佳聚类
        self.signal_clusters = {}
        for signal_type, clusters in signal_distribution.items():
            if clusters:
                # 按质量和强度排序
                clusters.sort(key=lambda x: x['quality'] * x['strength'], reverse=True)
                self.signal_clusters[signal_type] = clusters[:2]  # 每种信号最多选择2个聚类
                print(f"信号类型 {signal_type}: 选择了 {len(self.signal_clusters[signal_type])} 个聚类")
            else:
                self.signal_clusters[signal_type] = []
                print(f"信号类型 {signal_type}: 未找到合适的聚类")
        
        return self.signal_clusters
    
    def extract_enhanced_features(self, df, current_idx, window_size=30):
        """
        提取增强特征，提高信号区分度
        """
        if current_idx < window_size:
            return None
        
        start_idx = current_idx - window_size
        end_idx = current_idx
        window_data = df.iloc[start_idx:end_idx]
        
        prices = window_data['index_value'].values.astype(float)
        
        try:
            # 基础技术指标
            sma_5 = talib.SMA(prices, timeperiod=5)
            sma_10 = talib.SMA(prices, timeperiod=10)
            sma_20 = talib.SMA(prices, timeperiod=20)
            ema_12 = talib.EMA(prices, timeperiod=12)
            ema_26 = talib.EMA(prices, timeperiod=26)
            
            # 动量指标
            rsi = talib.RSI(prices, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(prices)
            
            # 波动率指标
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
            
            # 趋势强度指标
            adx = talib.ADX(prices, prices, prices, timeperiod=14)
            
            # 价格动量
            momentum = talib.MOM(prices, timeperiod=10)
            roc = talib.ROC(prices, timeperiod=10)
            
            # 增强特征集
            features = {
                # 价格位置特征
                'price_sma5_ratio': prices[-1] / sma_5[-1] if not np.isnan(sma_5[-1]) else 1,
                'price_sma10_ratio': prices[-1] / sma_10[-1] if not np.isnan(sma_10[-1]) else 1,
                'price_sma20_ratio': prices[-1] / sma_20[-1] if not np.isnan(sma_20[-1]) else 1,
                
                # 均线关系
                'sma5_sma10_ratio': sma_5[-1] / sma_10[-1] if not np.isnan(sma_5[-1]) and not np.isnan(sma_10[-1]) else 1,
                'sma10_sma20_ratio': sma_10[-1] / sma_20[-1] if not np.isnan(sma_10[-1]) and not np.isnan(sma_20[-1]) else 1,
                'ema12_ema26_ratio': ema_12[-1] / ema_26[-1] if not np.isnan(ema_12[-1]) and not np.isnan(ema_26[-1]) else 1,
                
                # 趋势特征
                'price_trend_5': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
                'price_trend_10': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0,
                'price_trend_20': (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0,
                
                # 动量指标
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'rsi_divergence': (rsi[-1] - rsi[-5]) if len(rsi) >= 5 and not np.isnan(rsi[-1]) and not np.isnan(rsi[-5]) else 0,
                
                # MACD信号
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                'macd_cross': 1 if (not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]) and macd[-1] > macd_signal[-1]) else 0,
                
                # 波动率特征
                'bb_position': (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if not np.isnan(bb_upper[-1]) else 0.5,
                'bb_width': (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if not np.isnan(bb_upper[-1]) else 0,
                'volatility': np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0,
                
                # 趋势强度
                'adx': adx[-1] if not np.isnan(adx[-1]) else 25,
                'trend_strength': 1 if (not np.isnan(adx[-1]) and adx[-1] > 25) else 0,
                
                # 动量特征
                'momentum': momentum[-1] if not np.isnan(momentum[-1]) else 0,
                'roc': roc[-1] if not np.isnan(roc[-1]) else 0,
                
                # 市场结构
                'higher_high': 1 if (len(prices) >= 5 and prices[-1] > max(prices[-5:-1])) else 0,
                'lower_low': 1 if (len(prices) >= 5 and prices[-1] < min(prices[-5:-1])) else 0,
                'inside_bar': 1 if (len(prices) >= 2 and abs(prices[-1] - prices[-2]) < np.std(prices[-10:])) else 0,
            }
            
            # 影响因子特征
            for factor in ['a', 'b', 'c', 'd']:
                if factor in window_data.columns:
                    factor_values = window_data[factor].values
                    features.update({
                        f'{factor}_current': factor_values[-1],
                        f'{factor}_trend': np.polyfit(range(len(factor_values)), factor_values, 1)[0] if len(factor_values) > 1 else 0,
                        f'{factor}_volatility': np.std(factor_values) / np.mean(factor_values) if np.mean(factor_values) != 0 else 0,
                        f'{factor}_momentum': (factor_values[-1] - factor_values[-5]) / factor_values[-5] if len(factor_values) >= 5 else 0,
                        f'{factor}_extreme': 1 if abs(factor_values[-1]) > 2 * np.std(factor_values) else 0,
                    })
            
            feature_vector = list(features.values())
            return feature_vector, features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def predict_diverse_signals(self, df, current_idx):
        """
        预测多样化信号
        """
        feature_result = self.extract_enhanced_features(df, current_idx)
        if feature_result is None:
            return None
        
        current_features, feature_dict = feature_result
        
        # 为每种信号类型计算概率
        signal_probabilities = {}
        
        for signal_type in [1, 2, 3, 4]:
            if signal_type in self.signal_clusters and self.signal_clusters[signal_type]:
                max_prob = 0
                best_cluster = None
                
                for cluster_info in self.signal_clusters[signal_type]:
                    # 基于特征计算该信号的概率
                    prob = self.calculate_signal_probability(feature_dict, signal_type, cluster_info)
                    if prob > max_prob:
                        max_prob = prob
                        best_cluster = cluster_info
                
                signal_probabilities[signal_type] = {
                    'probability': max_prob,
                    'cluster_info': best_cluster
                }
        
        # 选择最佳信号
        valid_signals = {}
        for signal_type, info in signal_probabilities.items():
            if info['probability'] >= self.signal_thresholds[signal_type]:
                valid_signals[signal_type] = info
        
        if not valid_signals:
            return None
        
        # 返回概率最高的信号
        best_signal = max(valid_signals.items(), key=lambda x: x[1]['probability'])
        
        return {
            'predicted_signal': best_signal[0],
            'confidence': best_signal[1]['probability'],
            'cluster_info': best_signal[1]['cluster_info'],
            'all_probabilities': {k: v['probability'] for k, v in signal_probabilities.items()}
        }
    
    def calculate_signal_probability(self, features, signal_type, cluster_info):
        """
        计算特定信号类型的概率
        """
        base_prob = cluster_info['strength'] * cluster_info['quality']
        
        # 根据市场条件调整概率
        market_adjustment = 1.0
        
        # 趋势信号调整
        if signal_type in [1, 3]:  # 开仓信号
            trend_strength = abs(features.get('price_trend_10', 0))
            if trend_strength > 0.02:  # 强趋势
                market_adjustment *= 1.3
            elif trend_strength < 0.005:  # 弱趋势
                market_adjustment *= 0.7
        
        # 平仓信号调整
        elif signal_type in [2, 4]:  # 平仓信号
            rsi = features.get('rsi', 50)
            bb_position = features.get('bb_position', 0.5)
            
            # 超买超卖区域增加平仓概率
            if rsi > 70 or rsi < 30 or bb_position > 0.8 or bb_position < 0.2:
                market_adjustment *= 1.4
        
        # 波动率调整
        volatility = features.get('volatility', 0)
        if volatility > 0.03:  # 高波动
            market_adjustment *= 1.2
        elif volatility < 0.01:  # 低波动
            market_adjustment *= 0.8
        
        # MACD信号调整
        macd_cross = features.get('macd_cross', 0)
        if signal_type == 1 and macd_cross == 1:  # 做多开仓 + MACD金叉
            market_adjustment *= 1.3
        elif signal_type == 3 and macd_cross == 0:  # 做空开仓 + MACD死叉
            market_adjustment *= 1.3
        
        final_prob = min(base_prob * market_adjustment, 1.0)
        return final_prob
    
    def backtest_enhanced_strategy(self, test_files=None, n_files=3):
        """
        回测增强策略
        """
        if test_files is None:
            label_files = sorted(glob.glob("./label/*.csv"))
            test_files = label_files[-n_files:]
        
        print(f"开始增强策略回测，使用 {len(test_files)} 个文件...")
        
        all_predictions = []
        signal_type_stats = {1: [], 2: [], 3: [], 4: []}
        
        for file_path in test_files:
            print(f"\n回测文件: {os.path.basename(file_path)}")
            
            try:
                df = pd.read_csv(file_path)
                file_predictions = []
                
                for i in range(30, len(df) - 1):
                    prediction = self.predict_diverse_signals(df, i)
                    
                    if prediction:
                        actual_label = df['label'].iloc[i + 1]
                        
                        if actual_label in [1, 2, 3, 4]:
                            is_correct = prediction['predicted_signal'] == actual_label
                            
                            signal_detail = {
                                'file': os.path.basename(file_path),
                                'index': i,
                                'predicted': prediction['predicted_signal'],
                                'actual': actual_label,
                                'confidence': prediction['confidence'],
                                'correct': is_correct,
                                'all_probabilities': prediction['all_probabilities']
                            }
                            
                            file_predictions.append(signal_detail)
                            all_predictions.append(signal_detail)
                            signal_type_stats[prediction['predicted_signal']].append(is_correct)
                
                if file_predictions:
                    file_accuracy = sum(1 for p in file_predictions if p['correct']) / len(file_predictions)
                    signal_diversity = len(set(p['predicted'] for p in file_predictions))
                    print(f"  预测数: {len(file_predictions)}")
                    print(f"  准确率: {file_accuracy:.4f}")
                    print(f"  信号多样性: {signal_diversity} 种不同信号")
                
            except Exception as e:
                print(f"回测文件 {file_path} 时出错: {e}")
                continue
        
        # 整体统计
        if all_predictions:
            overall_accuracy = sum(1 for p in all_predictions if p['correct']) / len(all_predictions)
            signal_diversity = len(set(p['predicted'] for p in all_predictions))
            
            print(f"\n=== 增强策略回测结果 ===")
            print(f"总预测数: {len(all_predictions)}")
            print(f"整体准确率: {overall_accuracy:.4f}")
            print(f"信号多样性: {signal_diversity} 种不同信号")
            
            # 各信号类型表现
            print("\n各信号类型表现:")
            signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
            for signal_type, results in signal_type_stats.items():
                if results:
                    accuracy = sum(results) / len(results)
                    print(f"  {signal_names[signal_type]}: {len(results)}次, 准确率{accuracy:.4f}")
            
            # 保存结果
            os.makedirs(self.models_dir, exist_ok=True)
            results_df = pd.DataFrame(all_predictions)
            results_df.to_csv(os.path.join(self.models_dir, "enhanced_backtest_results.csv"), index=False)
            
            return overall_accuracy, signal_diversity, signal_type_stats
        
        return 0, 0, {}
    
    def run_enhanced_analysis(self):
        """
        运行增强分析
        """
        print("开始增强信号多样性分析...")
        
        # 加载并重新平衡聚类
        self.load_and_rebalance_clusters()
        
        if not any(self.signal_clusters.values()):
            print("未找到合适的聚类数据！")
            return
        
        # 运行回测
        accuracy, diversity, type_stats = self.backtest_enhanced_strategy()
        
        print(f"\n=== 最终评估 ===")
        print(f"策略准确率: {accuracy:.4f}")
        print(f"信号多样性: {diversity} 种")
        
        if diversity >= 3:
            print("✅ 信号多样性良好！")
        elif diversity >= 2:
            print("⚠️  信号多样性中等")
        else:
            print("❌ 信号多样性不足")
        
        if accuracy > 0.6:
            print("✅ 优秀的预测表现！")
        elif accuracy > 0.5:
            print("✅ 良好的预测表现")
        elif accuracy > 0.4:
            print("⚠️  中等预测表现")
        else:
            print("❌ 预测表现不佳")
        
        return accuracy, diversity

def main():
    """
    主函数
    """
    predictor = EnhancedSignalDiversityPredictor()
    accuracy, diversity = predictor.run_enhanced_analysis()
    
    print(f"\n增强信号多样性分析完成")
    print(f"准确率: {accuracy:.4f}, 信号多样性: {diversity}")

if __name__ == "__main__":
    main()