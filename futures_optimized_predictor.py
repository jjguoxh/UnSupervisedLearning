# -*- coding: utf-8 -*-
"""
股指期货优化预测器
针对股指期货特点：每天1-2次开仓机会的低频高质量交易
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import talib
import warnings
warnings.filterwarnings('ignore')

# ========= 股指期货专用配置 =========
PATTERNS_DIR = "./patterns_improved/"
MODELS_DIR = "./models_futures/"
PATTERN_LENGTH = 30  # 增加模式长度，捕捉更长期趋势
MIN_CONFIDENCE = 0.4  # 降低置信度阈值，适应低频交易
SIGNAL_QUALITY_THRESHOLD = 0.7  # 高质量信号阈值
DAILY_SIGNAL_LIMIT = 4  # 每日最大信号数量（开仓+平仓）

class FuturesOptimizedPredictor:
    def __init__(self):
        self.cluster_info = {}
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def load_cluster_patterns(self):
        """
        加载并优化聚类模式，专注于高质量信号
        """
        analysis_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
        if not os.path.exists(analysis_file):
            raise FileNotFoundError(f"聚类分析文件未找到: {analysis_file}")
        
        cluster_analysis = pd.read_csv(analysis_file)
        print(f"加载了 {len(cluster_analysis)} 个聚类")
        
        # 选择所有高质量聚类（股指期货需要保留更多模式）
        high_quality_clusters = cluster_analysis[
            cluster_analysis['quality_score'] > 0.3
        ]
        
        print(f"选择了 {len(high_quality_clusters)} 个高质量聚类")
        
        for _, cluster_row in high_quality_clusters.iterrows():
            cluster_id = cluster_row['cluster_id']
            signal_counts = eval(cluster_row['signal_counts'])
            
            # 计算信号平衡度和交易完整性
            long_completeness = min(signal_counts.get(1, 0), signal_counts.get(2, 0))
            short_completeness = min(signal_counts.get(3, 0), signal_counts.get(4, 0))
            total_completeness = long_completeness + short_completeness
            
            # 股指期货特有的质量评估
            futures_quality = self.calculate_futures_quality(
                cluster_row, signal_counts, total_completeness
            )
            
            self.cluster_info[cluster_id] = {
                'quality_score': cluster_row['quality_score'],
                'signal_density': cluster_row['signal_density'],
                'cluster_size': cluster_row['cluster_size'],
                'signal_counts': signal_counts,
                'long_pairs': long_completeness,
                'short_pairs': short_completeness,
                'total_pairs': total_completeness,
                'futures_quality': futures_quality,
                'signal_balance': self.calculate_signal_balance(signal_counts)
            }
        
        print(f"加载了 {len(self.cluster_info)} 个聚类的模式数据")
        return self.cluster_info
    
    def calculate_futures_quality(self, cluster_row, signal_counts, total_completeness):
        """
        计算股指期货专用的质量评分
        """
        base_quality = cluster_row['quality_score']
        
        # 交易完整性奖励（有完整的开平仓对）
        completeness_bonus = min(total_completeness * 0.1, 0.3)
        
        # 聚类稳定性（较大的聚类更稳定）
        stability_bonus = min(np.log(cluster_row['cluster_size']) * 0.05, 0.2)
        
        # 信号多样性（包含多种信号类型）
        signal_diversity = len([v for v in signal_counts.values() if v > 0]) / 4
        diversity_bonus = signal_diversity * 0.15
        
        futures_quality = base_quality + completeness_bonus + stability_bonus + diversity_bonus
        return min(futures_quality, 1.0)
    
    def calculate_signal_balance(self, signal_counts):
        """
        计算信号平衡度
        """
        total_signals = sum(signal_counts.values())
        if total_signals == 0:
            return 0
        
        # 计算各信号类型的分布均匀度
        proportions = [signal_counts.get(i, 0) / total_signals for i in [1, 2, 3, 4]]
        # 使用熵来衡量分布均匀度
        entropy = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
        max_entropy = np.log(4)  # 4种信号类型的最大熵
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def extract_futures_features(self, df, current_idx, window_size=PATTERN_LENGTH):
        """
        提取股指期货专用特征
        """
        if current_idx < window_size:
            return None
        
        start_idx = current_idx - window_size
        end_idx = current_idx
        window_data = df.iloc[start_idx:end_idx]
        
        prices = window_data['index_value'].values.astype(float)
        
        try:
            # 股指期货关键技术指标
            # 趋势指标
            sma_5 = talib.SMA(prices, timeperiod=5)
            sma_20 = talib.SMA(prices, timeperiod=20)
            ema_12 = talib.EMA(prices, timeperiod=12)
            
            # 动量指标
            rsi = talib.RSI(prices, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(prices)
            
            # 波动率指标
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
            atr = talib.ATR(prices, prices, prices, timeperiod=14)  # 使用价格作为高低价代理
            
            # 成交量代理指标（基于价格变化）
            price_changes = np.abs(np.diff(prices))
            volume_proxy = np.mean(price_changes[-10:])  # 最近10期的平均价格变化
            
            # 股指期货特有特征
            futures_features = {
                # 价格位置特征
                'price_current': prices[-1],
                'price_sma5_ratio': prices[-1] / sma_5[-1] if not np.isnan(sma_5[-1]) else 1,
                'price_sma20_ratio': prices[-1] / sma_20[-1] if not np.isnan(sma_20[-1]) else 1,
                'sma5_sma20_ratio': sma_5[-1] / sma_20[-1] if not np.isnan(sma_5[-1]) and not np.isnan(sma_20[-1]) else 1,
                
                # 趋势强度
                'price_trend_5': (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
                'price_trend_10': (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0,
                'price_trend_20': (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0,
                
                # 动量指标
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'rsi_overbought': 1 if (not np.isnan(rsi[-1]) and rsi[-1] > 70) else 0,
                'rsi_oversold': 1 if (not np.isnan(rsi[-1]) and rsi[-1] < 30) else 0,
                
                # MACD信号
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                'macd_bullish': 1 if (not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]) and macd[-1] > macd_signal[-1]) else 0,
                
                # 波动率特征
                'bb_position': (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if not np.isnan(bb_upper[-1]) else 0.5,
                'bb_squeeze': 1 if (not np.isnan(bb_upper[-1]) and (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] < 0.1) else 0,
                'volatility': np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0,
                
                # 市场结构
                'higher_high': 1 if (len(prices) >= 5 and prices[-1] > max(prices[-5:-1])) else 0,
                'lower_low': 1 if (len(prices) >= 5 and prices[-1] < min(prices[-5:-1])) else 0,
                'volume_proxy': volume_proxy,
            }
            
            # 影响因子特征（股指期货的基本面因素）
            factor_features = {}
            for factor in ['a', 'b', 'c', 'd']:
                if factor in window_data.columns:
                    factor_values = window_data[factor].values
                    factor_features.update({
                        f'{factor}_current': factor_values[-1],
                        f'{factor}_trend': np.polyfit(range(len(factor_values)), factor_values, 1)[0] if len(factor_values) > 1 else 0,
                        f'{factor}_volatility': np.std(factor_values) / np.mean(factor_values) if np.mean(factor_values) != 0 else 0,
                        f'{factor}_momentum': (factor_values[-1] - factor_values[-5]) / factor_values[-5] if len(factor_values) >= 5 else 0,
                    })
            
            # 合并所有特征
            all_features = {**futures_features, **factor_features}
            feature_vector = list(all_features.values())
            
            return feature_vector, all_features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def calculate_futures_confidence(self, current_features, cluster_id):
        """
        计算股指期货专用的置信度
        """
        if cluster_id not in self.cluster_info:
            return 0
        
        cluster_info = self.cluster_info[cluster_id]
        
        # 基础置信度（基于聚类质量）
        base_confidence = cluster_info['futures_quality']
        
        # 交易完整性加成
        completeness_ratio = cluster_info['total_pairs'] / cluster_info['cluster_size']
        completeness_bonus = min(completeness_ratio * 0.3, 0.2)
        
        # 信号平衡度加成
        balance_bonus = cluster_info['signal_balance'] * 0.15
        
        # 聚类稳定性加成
        stability_bonus = min(np.log(cluster_info['cluster_size']) * 0.05, 0.15)
        
        # 综合置信度
        total_confidence = base_confidence + completeness_bonus + balance_bonus + stability_bonus
        
        return min(total_confidence, 1.0)
    
    def predict_futures_signal(self, df, current_idx):
        """
        股指期货信号预测
        """
        feature_result = self.extract_futures_features(df, current_idx)
        if feature_result is None:
            return None
        
        current_features, feature_dict = feature_result
        
        # 市场状态判断
        market_state = self.analyze_market_state(feature_dict)
        
        predictions = []
        
        for cluster_id in self.cluster_info.keys():
            confidence = self.calculate_futures_confidence(current_features, cluster_id)
            
            if confidence >= MIN_CONFIDENCE:
                cluster_info = self.cluster_info[cluster_id]
                signal_counts = cluster_info['signal_counts']
                
                # 根据市场状态调整信号偏好
                adjusted_signal_counts = self.adjust_signals_by_market_state(
                    signal_counts, market_state
                )
                
                if adjusted_signal_counts:
                    most_likely_signal = max(adjusted_signal_counts.items(), key=lambda x: x[1])[0]
                    signal_probability = adjusted_signal_counts[most_likely_signal] / sum(adjusted_signal_counts.values())
                    
                    # 最终置信度调整
                    final_confidence = confidence * signal_probability * market_state['confidence_multiplier']
                    
                    prediction = {
                        'cluster_id': cluster_id,
                        'predicted_signal': most_likely_signal,
                        'confidence': final_confidence,
                        'signal_probability': signal_probability,
                        'market_state': market_state['state'],
                        'cluster_quality': cluster_info['futures_quality']
                    }
                    
                    predictions.append(prediction)
        
        # 按置信度排序并限制数量
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 股指期货每日信号限制
        return predictions[:2]  # 最多返回2个最高置信度的预测
    
    def analyze_market_state(self, features):
        """
        分析市场状态
        """
        # 趋势状态
        trend_signals = [
            features.get('price_trend_5', 0),
            features.get('price_trend_10', 0),
            features.get('price_trend_20', 0)
        ]
        
        avg_trend = np.mean(trend_signals)
        
        # 动量状态
        rsi = features.get('rsi', 50)
        macd_bullish = features.get('macd_bullish', 0)
        
        # 波动率状态
        volatility = features.get('volatility', 0)
        bb_position = features.get('bb_position', 0.5)
        
        # 综合市场状态判断
        if avg_trend > 0.01 and rsi < 70 and macd_bullish:
            state = 'bullish'
            confidence_multiplier = 1.2
        elif avg_trend < -0.01 and rsi > 30 and not macd_bullish:
            state = 'bearish'
            confidence_multiplier = 1.2
        elif volatility < 0.02:
            state = 'consolidation'
            confidence_multiplier = 0.8
        elif bb_position > 0.8 or bb_position < 0.2:
            state = 'extreme'
            confidence_multiplier = 1.1
        else:
            state = 'neutral'
            confidence_multiplier = 1.0
        
        return {
            'state': state,
            'confidence_multiplier': confidence_multiplier,
            'trend_strength': abs(avg_trend),
            'volatility': volatility
        }
    
    def adjust_signals_by_market_state(self, signal_counts, market_state):
        """
        根据市场状态调整信号权重
        """
        adjusted_counts = signal_counts.copy()
        state = market_state['state']
        
        if state == 'bullish':
            # 看涨市场，增加做多信号权重
            adjusted_counts[1] = adjusted_counts.get(1, 0) * 1.5  # 做多开仓
            adjusted_counts[4] = adjusted_counts.get(4, 0) * 1.3  # 做空平仓
        elif state == 'bearish':
            # 看跌市场，增加做空信号权重
            adjusted_counts[3] = adjusted_counts.get(3, 0) * 1.5  # 做空开仓
            adjusted_counts[2] = adjusted_counts.get(2, 0) * 1.3  # 做多平仓
        elif state == 'extreme':
            # 极端市场，增加平仓信号权重
            adjusted_counts[2] = adjusted_counts.get(2, 0) * 1.4  # 做多平仓
            adjusted_counts[4] = adjusted_counts.get(4, 0) * 1.4  # 做空平仓
        
        return {k: v for k, v in adjusted_counts.items() if v > 0}
    
    def backtest_futures_strategy(self, test_files=None, n_files=3):
        """
        股指期货策略回测
        """
        if test_files is None:
            label_files = sorted(glob.glob("./label/*.csv"))
            test_files = label_files[-n_files:]
        
        print(f"开始股指期货策略回测，使用 {len(test_files)} 个文件...")
        
        daily_results = []
        all_predictions = []
        
        for file_path in test_files:
            print(f"\n回测文件: {os.path.basename(file_path)}")
            
            try:
                df = pd.read_csv(file_path)
                daily_signals = []
                daily_accuracy = []
                
                # 每日信号计数
                daily_signal_count = 0
                
                for i in range(PATTERN_LENGTH, len(df) - 1):
                    if daily_signal_count >= DAILY_SIGNAL_LIMIT:
                        continue
                    
                    predictions = self.predict_futures_signal(df, i)
                    
                    if predictions:
                        best_prediction = predictions[0]
                        actual_label = df['label'].iloc[i + 1]
                        
                        if actual_label in [1, 2, 3, 4]:
                            is_correct = best_prediction['predicted_signal'] == actual_label
                            
                            signal_detail = {
                                'file': os.path.basename(file_path),
                                'index': i,
                                'predicted': best_prediction['predicted_signal'],
                                'actual': actual_label,
                                'confidence': best_prediction['confidence'],
                                'market_state': best_prediction['market_state'],
                                'correct': is_correct
                            }
                            
                            daily_signals.append(signal_detail)
                            daily_accuracy.append(is_correct)
                            all_predictions.append(signal_detail)
                            daily_signal_count += 1
                
                # 每日统计
                if daily_signals:
                    daily_acc = sum(daily_accuracy) / len(daily_accuracy)
                    daily_results.append({
                        'file': os.path.basename(file_path),
                        'signals_count': len(daily_signals),
                        'accuracy': daily_acc,
                        'avg_confidence': np.mean([s['confidence'] for s in daily_signals])
                    })
                    
                    print(f"  信号数量: {len(daily_signals)}")
                    print(f"  准确率: {daily_acc:.4f}")
                    print(f"  平均置信度: {np.mean([s['confidence'] for s in daily_signals]):.4f}")
                
            except Exception as e:
                print(f"回测文件 {file_path} 时出错: {e}")
                continue
        
        # 整体统计
        if all_predictions:
            overall_accuracy = sum(1 for p in all_predictions if p['correct']) / len(all_predictions)
            avg_daily_signals = np.mean([r['signals_count'] for r in daily_results])
            
            print(f"\n=== 股指期货策略回测结果 ===")
            print(f"总预测数: {len(all_predictions)}")
            print(f"整体准确率: {overall_accuracy:.4f}")
            print(f"平均每日信号数: {avg_daily_signals:.2f}")
            
            # 按市场状态分析
            state_performance = {}
            for state in ['bullish', 'bearish', 'neutral', 'consolidation', 'extreme']:
                state_preds = [p for p in all_predictions if p['market_state'] == state]
                if state_preds:
                    state_acc = sum(1 for p in state_preds if p['correct']) / len(state_preds)
                    state_performance[state] = {
                        'count': len(state_preds),
                        'accuracy': state_acc
                    }
            
            print("\n市场状态表现:")
            for state, perf in state_performance.items():
                print(f"  {state}: {perf['accuracy']:.4f} ({perf['count']} 个信号)")
            
            # 保存结果
            os.makedirs(MODELS_DIR, exist_ok=True)
            results_df = pd.DataFrame(all_predictions)
            results_df.to_csv(os.path.join(MODELS_DIR, "futures_backtest_results.csv"), index=False)
            
            daily_df = pd.DataFrame(daily_results)
            daily_df.to_csv(os.path.join(MODELS_DIR, "daily_performance.csv"), index=False)
            
            return overall_accuracy, state_performance, daily_results
        
        return 0, {}, []
    
    def run_futures_analysis(self):
        """
        运行股指期货分析
        """
        print("开始股指期货优化预测分析...")
        
        # 加载聚类模式
        self.load_cluster_patterns()
        
        if not self.cluster_info:
            print("未加载到聚类模式数据！")
            return
        
        # 运行回测
        accuracy, state_perf, daily_results = self.backtest_futures_strategy()
        
        print(f"\n=== 最终评估 ===")
        print(f"策略准确率: {accuracy:.4f}")
        
        if accuracy > 0.6:
            print("✅ 优秀的预测表现！适合股指期货交易")
        elif accuracy > 0.5:
            print("✅ 良好的预测表现，符合股指期货低频高质量交易特点")
        elif accuracy > 0.4:
            print("⚠️  中等预测表现，需要进一步优化")
        else:
            print("❌ 预测表现不佳，需要重新设计策略")
        
        return accuracy, state_perf

def main():
    """
    主函数
    """
    # 创建股指期货优化预测器
    predictor = FuturesOptimizedPredictor()
    
    # 运行分析
    accuracy, state_perf = predictor.run_futures_analysis()
    
    print(f"\n股指期货预测系统分析完成")
    print(f"系统针对每日1-2次交易机会进行了专门优化")

if __name__ == "__main__":
    main()