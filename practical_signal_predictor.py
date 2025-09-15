# -*- coding: utf-8 -*-
"""
实用信号预测器
解决信号多样性问题的实际可行方案
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import talib
import warnings
warnings.filterwarnings('ignore')

class PracticalSignalPredictor:
    def __init__(self):
        self.patterns_dir = "./patterns_improved/"
        self.models_dir = "./models_practical/"
        self.cluster_info = {}
        
    def load_cluster_data(self):
        """
        加载聚类数据并分析信号分布
        """
        analysis_file = os.path.join(self.patterns_dir, "cluster_analysis.csv")
        if not os.path.exists(analysis_file):
            raise FileNotFoundError(f"聚类分析文件未找到: {analysis_file}")
        
        cluster_analysis = pd.read_csv(analysis_file)
        print(f"加载了 {len(cluster_analysis)} 个聚类")
        
        # 分析原始信号分布问题
        total_signal_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for _, cluster_row in cluster_analysis.iterrows():
            signal_counts = eval(cluster_row['signal_counts'])
            for signal_type, count in signal_counts.items():
                total_signal_counts[signal_type] += count
        
        print("\n原始数据信号分布:")
        signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        for signal_type, count in total_signal_counts.items():
            print(f"  {signal_names[signal_type]}: {count}")
        
        # 识别主导信号
        dominant_signal = max(total_signal_counts.items(), key=lambda x: x[1])
        print(f"\n主导信号: {signal_names[dominant_signal[0]]} ({dominant_signal[1]} 次)")
        
        # 保存聚类信息
        for _, cluster_row in cluster_analysis.iterrows():
            cluster_id = cluster_row['cluster_id']
            self.cluster_info[cluster_id] = {
                'quality_score': cluster_row['quality_score'],
                'signal_counts': eval(cluster_row['signal_counts']),
                'cluster_size': cluster_row['cluster_size']
            }
        
        return total_signal_counts, dominant_signal
    
    def extract_market_features(self, df, current_idx, window_size=20):
        """
        提取市场特征用于信号预测
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
            rsi = talib.RSI(prices, timeperiod=14)
            macd, macd_signal, _ = talib.MACD(prices)
            
            # 市场状态特征
            current_price = prices[-1]
            price_change_5 = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            price_change_10 = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # 趋势特征
            trend_up = 1 if (not np.isnan(sma_5[-1]) and not np.isnan(sma_10[-1]) and sma_5[-1] > sma_10[-1]) else 0
            strong_trend = 1 if abs(price_change_10) > 0.02 else 0
            
            # RSI状态
            rsi_value = rsi[-1] if not np.isnan(rsi[-1]) else 50
            rsi_overbought = 1 if rsi_value > 70 else 0
            rsi_oversold = 1 if rsi_value < 30 else 0
            
            # MACD状态
            macd_bullish = 1 if (not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]) and macd[-1] > macd_signal[-1]) else 0
            
            # 波动率
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
            high_volatility = 1 if volatility > 0.02 else 0
            
            features = {
                'price_change_5': price_change_5,
                'price_change_10': price_change_10,
                'trend_up': trend_up,
                'strong_trend': strong_trend,
                'rsi': rsi_value,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'macd_bullish': macd_bullish,
                'volatility': volatility,
                'high_volatility': high_volatility
            }
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def predict_signal_with_rules(self, features):
        """
        基于规则的信号预测，确保信号多样性
        """
        if features is None:
            return None
        
        # 信号预测规则
        predictions = []
        
        # 做多开仓条件
        if (features['trend_up'] == 1 and 
            features['rsi'] < 60 and 
            features['macd_bullish'] == 1 and
            features['price_change_5'] > 0.005):
            predictions.append({'signal': 1, 'confidence': 0.7, 'reason': '上升趋势+RSI未超买+MACD金叉'})
        
        # 做多平仓条件
        if (features['rsi_overbought'] == 1 or 
            (features['price_change_5'] < -0.01 and features['trend_up'] == 0)):
            predictions.append({'signal': 2, 'confidence': 0.6, 'reason': 'RSI超买或趋势转弱'})
        
        # 做空开仓条件
        if (features['trend_up'] == 0 and 
            features['rsi'] > 40 and 
            features['macd_bullish'] == 0 and
            features['price_change_5'] < -0.005):
            predictions.append({'signal': 3, 'confidence': 0.7, 'reason': '下降趋势+RSI未超卖+MACD死叉'})
        
        # 做空平仓条件
        if (features['rsi_oversold'] == 1 or 
            (features['price_change_5'] > 0.01 and features['trend_up'] == 1)):
            predictions.append({'signal': 4, 'confidence': 0.6, 'reason': 'RSI超卖或趋势转强'})
        
        # 高波动率时的保守策略
        if features['high_volatility'] == 1:
            # 高波动时倾向于平仓
            if features['rsi'] > 60:
                predictions.append({'signal': 2, 'confidence': 0.5, 'reason': '高波动+RSI偏高'})
            elif features['rsi'] < 40:
                predictions.append({'signal': 4, 'confidence': 0.5, 'reason': '高波动+RSI偏低'})
        
        # 选择最佳预测
        if predictions:
            best_prediction = max(predictions, key=lambda x: x['confidence'])
            return best_prediction
        
        return None
    
    def backtest_practical_strategy(self, test_files=None, n_files=3):
        """
        回测实用策略
        """
        if test_files is None:
            label_files = sorted(glob.glob("./label/*.csv"))
            test_files = label_files[-n_files:]
        
        print(f"\n开始实用策略回测，使用 {len(test_files)} 个文件...")
        
        all_predictions = []
        signal_type_stats = {1: [], 2: [], 3: [], 4: []}
        
        for file_path in test_files:
            print(f"\n回测文件: {os.path.basename(file_path)}")
            
            try:
                df = pd.read_csv(file_path)
                file_predictions = []
                
                for i in range(20, len(df) - 1):
                    features = self.extract_market_features(df, i)
                    prediction = self.predict_signal_with_rules(features)
                    
                    if prediction:
                        actual_label = df['label'].iloc[i + 1]
                        
                        if actual_label in [1, 2, 3, 4]:
                            is_correct = prediction['signal'] == actual_label
                            
                            signal_detail = {
                                'file': os.path.basename(file_path),
                                'index': i,
                                'predicted': prediction['signal'],
                                'actual': actual_label,
                                'confidence': prediction['confidence'],
                                'reason': prediction['reason'],
                                'correct': is_correct
                            }
                            
                            file_predictions.append(signal_detail)
                            all_predictions.append(signal_detail)
                            signal_type_stats[prediction['signal']].append(is_correct)
                
                if file_predictions:
                    file_accuracy = sum(1 for p in file_predictions if p['correct']) / len(file_predictions)
                    signal_diversity = len(set(p['predicted'] for p in file_predictions))
                    signal_dist = Counter(p['predicted'] for p in file_predictions)
                    
                    print(f"  预测数: {len(file_predictions)}")
                    print(f"  准确率: {file_accuracy:.4f}")
                    print(f"  信号多样性: {signal_diversity} 种")
                    print(f"  信号分布: {dict(signal_dist)}")
                
            except Exception as e:
                print(f"回测文件 {file_path} 时出错: {e}")
                continue
        
        # 整体统计
        if all_predictions:
            overall_accuracy = sum(1 for p in all_predictions if p['correct']) / len(all_predictions)
            signal_diversity = len(set(p['predicted'] for p in all_predictions))
            total_signal_dist = Counter(p['predicted'] for p in all_predictions)
            
            print(f"\n=== 实用策略回测结果 ===")
            print(f"总预测数: {len(all_predictions)}")
            print(f"整体准确率: {overall_accuracy:.4f}")
            print(f"信号多样性: {signal_diversity} 种不同信号")
            print(f"总信号分布: {dict(total_signal_dist)}")
            
            # 各信号类型表现
            print("\n各信号类型表现:")
            signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
            for signal_type, results in signal_type_stats.items():
                if results:
                    accuracy = sum(results) / len(results)
                    print(f"  {signal_names[signal_type]}: {len(results)}次, 准确率{accuracy:.4f}")
            
            # 分析预测原因
            print("\n预测原因分析:")
            reason_stats = Counter(p['reason'] for p in all_predictions)
            for reason, count in reason_stats.most_common():
                reason_accuracy = sum(1 for p in all_predictions if p['reason'] == reason and p['correct']) / count
                print(f"  {reason}: {count}次, 准确率{reason_accuracy:.4f}")
            
            # 保存结果
            os.makedirs(self.models_dir, exist_ok=True)
            results_df = pd.DataFrame(all_predictions)
            results_df.to_csv(os.path.join(self.models_dir, "practical_backtest_results.csv"), index=False)
            
            return overall_accuracy, signal_diversity, signal_type_stats, total_signal_dist
        
        return 0, 0, {}, {}
    
    def analyze_original_problem(self):
        """
        分析原始问题
        """
        print("=== 原始问题分析 ===")
        
        # 加载原始聚类数据
        total_signals, dominant_signal = self.load_cluster_data()
        
        print(f"\n🔍 问题诊断:")
        signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        
        # 检查信号分布不均
        total_count = sum(total_signals.values())
        if total_count > 0:
            for signal_type, count in total_signals.items():
                ratio = count / total_count
                print(f"  {signal_names[signal_type]}: {count} ({ratio:.1%})")
            
            # 判断问题严重程度
            max_ratio = max(count / total_count for count in total_signals.values())
            if max_ratio > 0.8:
                print(f"\n❌ 严重问题: 单一信号占比{max_ratio:.1%}，数据严重不平衡")
            elif max_ratio > 0.6:
                print(f"\n⚠️  中等问题: 单一信号占比{max_ratio:.1%}，数据不平衡")
            else:
                print(f"\n✅ 数据分布相对均衡")
        
        return total_signals
    
    def run_complete_analysis(self):
        """
        运行完整分析
        """
        print("开始实用信号预测分析...")
        print("=" * 60)
        
        # 分析原始问题
        original_signals = self.analyze_original_problem()
        
        # 运行实用策略回测
        accuracy, diversity, type_stats, signal_dist = self.backtest_practical_strategy()
        
        print(f"\n=== 解决方案效果 ===")
        print(f"策略准确率: {accuracy:.4f}")
        print(f"信号多样性: {diversity} 种")
        
        if diversity >= 3:
            print("✅ 成功解决信号多样性问题！")
        elif diversity >= 2:
            print("⚠️  部分解决信号多样性问题")
        else:
            print("❌ 信号多样性问题仍然存在")
        
        # 对比分析
        print(f"\n=== 改进对比 ===")
        print(f"原始数据问题: 信号分布极不均衡")
        print(f"解决方案效果: 产生了{diversity}种不同信号")
        
        if accuracy > 0.4:
            print(f"✅ 准确率{accuracy:.1%}超过4分类随机水平(25%)")
        else:
            print(f"⚠️  准确率{accuracy:.1%}需要进一步提升")
        
        return accuracy, diversity

def main():
    """
    主函数
    """
    predictor = PracticalSignalPredictor()
    accuracy, diversity = predictor.run_complete_analysis()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 最终结论:")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   信号多样性: {diversity} 种")
    
    if diversity >= 3 and accuracy > 0.3:
        print(f"\n✅ 成功解决了'训练了个寂寞'的问题！")
        print(f"   • 实现了多样化信号预测")
        print(f"   • 准确率超过随机水平")
        print(f"   • 提供了可解释的预测逻辑")
    else:
        print(f"\n⚠️  问题部分解决，仍需进一步优化")

if __name__ == "__main__":
    main()