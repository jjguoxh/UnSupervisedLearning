# -*- coding: utf-8 -*-
"""
分析实际交易信号的预测质量
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def analyze_trading_signals():
    """
    分析实际交易信号的预测质量
    """
    print("Analyzing trading signal prediction quality...")
    
    # 获取一个测试文件
    label_files = sorted([f for f in os.listdir("label") if f.endswith(".csv")])
    if not label_files:
        print("No label files found!")
        return
    
    test_file = os.path.join("label", label_files[0])
    print(f"Using test file: {test_file}")
    
    # 加载带标签的数据用于评估
    df = pd.read_csv(test_file)
    if df is None:
        print("Failed to load test data!")
        return
    
    print(f"Loaded data with {len(df)} rows")
    
    # 找到所有实际的交易信号点
    trading_signal_indices = []
    for i in range(len(df)):
        label = df.iloc[i]['label']
        if label != 0:  # 交易信号
            trading_signal_indices.append((i, label))
    
    print(f"\nFound {len(trading_signal_indices)} actual trading signals:")
    for idx, label in trading_signal_indices:
        print(f"  Index {idx}: Label {label}")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 分析每个交易信号点的预测
    print(f"\nAnalyzing predictions at trading signal points:")
    correct_predictions = 0
    total_predictions = 0
    
    for idx, actual_label in trading_signal_indices:
        # 获取该点的预测
        predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:idx+1])
        
        print(f"  Index {idx}:")
        print(f"    Actual label: {actual_label}")
        print(f"    Predicted signal: {predicted_signal}")
        print(f"    Confidence: {confidence:.4f}")
        
        total_predictions += 1
        if actual_label == predicted_signal:
            correct_predictions += 1
            print(f"    >> CORRECT PREDICTION")
        else:
            print(f"    >> INCORRECT PREDICTION")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nTrading signal prediction accuracy:")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # 分析强化学习模型的影响
    print(f"\nAnalyzing RL model impact on trading signals:")
    rl_correct_predictions = 0
    rl_total_predictions = 0
    
    # 尝试加载强化学习模型
    rl_model_path = os.path.join("model", "balanced_model", "rl_trader_model.json")
    use_rl = predictor.load_rl_model(rl_model_path)
    
    if use_rl:
        print("RL model loaded successfully")
        for idx, actual_label in trading_signal_indices:
            # 获取RL模型的预测
            rl_signal, rl_confidence = predictor.predict_signal_with_rl(df.iloc[:idx+1], idx)
            
            print(f"  Index {idx}:")
            print(f"    Actual label: {actual_label}")
            print(f"    Base prediction: {df.iloc[idx]['label'] if 'label' in df.columns else 'N/A'}")
            print(f"    RL prediction: {rl_signal}")
            print(f"    RL confidence: {rl_confidence:.4f}")
            
            rl_total_predictions += 1
            if actual_label == rl_signal:
                rl_correct_predictions += 1
                print(f"    >> RL CORRECT PREDICTION")
            else:
                print(f"    >> RL INCORRECT PREDICTION")
        
        rl_accuracy = rl_correct_predictions / rl_total_predictions if rl_total_predictions > 0 else 0
        print(f"\nRL model trading signal prediction accuracy:")
        print(f"  Correct predictions: {rl_correct_predictions}")
        print(f"  Total predictions: {rl_total_predictions}")
        print(f"  Accuracy: {rl_accuracy:.2%}")
    else:
        print("Failed to load RL model")

if __name__ == "__main__":
    analyze_trading_signals()