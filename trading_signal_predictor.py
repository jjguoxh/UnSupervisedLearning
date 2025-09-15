
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用交易信号预测器
专注于预测1-4交易信号，忽略0信号的平衡问题
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class TradingSignalPredictor:
    def __init__(self, model_dir="model/specialized"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            model_file = os.path.join(self.model_dir, "best_trading_model.pkl")
            scaler_file = os.path.join(self.model_dir, "trading_scaler.pkl")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✓ 模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def extract_features(self, df, index):
        """提取预测特征（与训练时保持一致）"""
        try:
            features = []
            
            # 基础特征
            if 'a' in df.columns:
                features.extend([df.iloc[index]['a'], df.iloc[index]['b'], 
                               df.iloc[index]['c'], df.iloc[index]['d']])
            
            # 价格相关特征
            if 'index_value' in df.columns:
                current_price = df.iloc[index]['index_value']
                
                # 短期价格变化
                price_changes = []
                for lookback in [1, 3, 5, 10]:
                    if index >= lookback:
                        past_price = df.iloc[index-lookback]['index_value']
                        change = (current_price - past_price) / past_price
                        price_changes.append(change)
                    else:
                        price_changes.append(0)
                
                features.extend(price_changes)
                
                # 价格波动率
                if index >= 10:
                    recent_prices = df.iloc[index-10:index]['index_value'].values
                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                    features.append(volatility)
                else:
                    features.append(0)
                
                # 趋势强度
                if index >= 20:
                    long_prices = df.iloc[index-20:index]['index_value'].values
                    trend_strength = (long_prices[-1] - long_prices[0]) / long_prices[0]
                    features.append(trend_strength)
                else:
                    features.append(0)
            
            # 技术指标特征
            if 'a' in df.columns and 'b' in df.columns:
                # 简单移动平均
                if index >= 5:
                    ma_a = np.mean(df.iloc[index-5:index]['a'].values)
                    ma_b = np.mean(df.iloc[index-5:index]['b'].values)
                    features.extend([ma_a, ma_b])
                    
                    # 当前值与移动平均的偏离
                    features.extend([
                        df.iloc[index]['a'] - ma_a,
                        df.iloc[index]['b'] - ma_b
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            return np.array(features).reshape(1, -1) if len(features) > 0 else None
            
        except Exception as e:
            return None
    
    def predict_signal(self, df, index):
        """预测交易信号"""
        if self.model is None or self.scaler is None:
            return 0, 0.0
        
        features = self.extract_features(df, index)
        if features is None:
            return 0, 0.0
        
        try:
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            
            # 获取预测概率（置信度）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.5
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            return 0, 0.0
    
    def batch_predict(self, df, min_confidence=0.6):
        """批量预测整个数据集"""
        predictions = []
        confidences = []
        
        for i in range(len(df)):
            pred, conf = self.predict_signal(df, i)
            
            # 只保留高置信度的交易信号
            if pred != 0 and conf < min_confidence:
                pred = 0
            
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences
    
    def evaluate_predictions(self, df, predictions, actual_labels=None):
        """评估预测结果"""
        signal_names = {0: "等待", 1: "做多开仓", 2: "做多平仓", 3: "做空开仓", 4: "做空平仓"}
        
        # 统计预测信号
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        print("\n预测信号统计:")
        for signal, count in sorted(pred_counts.items()):
            percentage = count / len(predictions) * 100
            print(f"  {signal_names.get(signal, f'未知({signal})')}: {count} 次 ({percentage:.2f}%)")
        
        # 如果有实际标签，计算准确率
        if actual_labels is not None:
            trading_signals_pred = [p for p in predictions if p != 0]
            trading_signals_actual = [a for a in actual_labels if a != 0]
            
            if len(trading_signals_pred) > 0 and len(trading_signals_actual) > 0:
                # 只评估交易信号的准确性
                correct_trading = sum(1 for p, a in zip(predictions, actual_labels) 
                                    if p != 0 and p == a)
                total_trading_pred = sum(1 for p in predictions if p != 0)
                
                if total_trading_pred > 0:
                    trading_precision = correct_trading / total_trading_pred
                    print(f"\n交易信号精确率: {trading_precision:.2%}")
                    print(f"正确交易信号: {correct_trading}/{total_trading_pred}")

def test_predictor():
    """测试预测器"""
    predictor = TradingSignalPredictor()
    
    # 测试数据文件
    test_files = [f for f in os.listdir("label") if f.endswith(".csv")][:3]
    
    for file_name in test_files:
        print(f"\n=== 测试文件: {file_name} ===")
        
        file_path = os.path.join("label", file_name)
        df = pd.read_csv(file_path)
        
        # 批量预测
        predictions, confidences = predictor.batch_predict(df, min_confidence=0.7)
        
        # 评估结果
        actual_labels = df['label'].tolist() if 'label' in df.columns else None
        predictor.evaluate_predictions(df, predictions, actual_labels)

if __name__ == "__main__":
    test_predictor()
