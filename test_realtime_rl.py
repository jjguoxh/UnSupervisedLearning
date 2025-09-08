# -*- coding: utf-8 -*-
"""
测试realtime_predictor中的强化学习模型使用
"""

import os
import sys
import pandas as pd

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def test_realtime_rl_prediction():
    """
    测试实时预测中强化学习模型的使用
    """
    print("Testing RL model usage in realtime prediction...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    CURRENT_DIR = os.path.dirname(os.path.abspath('src/realtime_predictor.py'))
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    
    print(f"RL model path: {rl_model_path}")
    print(f"RL model file exists: {os.path.exists(rl_model_path)}")
    
    # 尝试加载模型
    use_rl = predictor.load_rl_model(rl_model_path)
    
    if use_rl:
        print("Successfully loaded RL model!")
        print(f"RL trader exists: {predictor.rl_trader is not None}")
        if predictor.rl_trader:
            print(f"Q-table size: {len(predictor.rl_trader.q_table)}")
    else:
        print("Failed to load RL model!")
        return
    
    # 获取一个测试文件
    label_files = sorted([f for f in os.listdir("label") if f.endswith(".csv")])
    if not label_files:
        print("No label files found!")
        return
    
    test_file = os.path.join("label", label_files[0])
    print(f"Using test file: {test_file}")
    
    # 加载数据
    df = load_realtime_data(test_file)
    if df is None:
        print("Failed to load test data!")
        return
    
    print(f"Loaded data with {len(df)} rows")
    
    # 进行预测（使用RL模型）
    predicted_signal, confidence = predictor.predict_signal_with_rl(df, len(df) - 1)
    print(f"RL prediction - Signal: {predicted_signal}, Confidence: {confidence:.4f}")
    
    # 进行预测（不使用RL模型）
    base_signal, base_confidence = predictor.predict_realtime_signal(df)
    print(f"Base prediction - Signal: {base_signal}, Confidence: {base_confidence:.4f}")
    
    print(f"Predictions match: {predicted_signal == base_signal}")

if __name__ == "__main__":
    test_realtime_rl_prediction()