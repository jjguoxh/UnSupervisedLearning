# -*- coding: utf-8 -*-
"""
分析预测结果，特别是开仓信号的缺失问题
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def analyze_predictions():
    """
    分析预测结果，检查开仓信号缺失问题
    """
    print("Analyzing prediction results for missing opening signals...")
    
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
            print("Q-table keys:", list(predictor.rl_trader.q_table.keys()))
    else:
        print("Failed to load RL model!")
    
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
    
    # 进行多个点的预测分析
    print("\nAnalyzing predictions at multiple points...")
    signals_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    rl_signals_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # 分析最后100个点
    start_idx = max(0, len(df) - 100)
    for i in range(start_idx, len(df)):
        # 基础预测
        base_signal, base_confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
        signals_count[base_signal] += 1
        
        # RL预测（如果加载了RL模型）
        if use_rl:
            rl_signal, rl_confidence = predictor.predict_signal_with_rl(df.iloc[:i+1], i)
            rl_signals_count[rl_signal] += 1
    
    print(f"\nBase prediction signal distribution:")
    for signal, count in signals_count.items():
        print(f"  Signal {signal}: {count} times")
    
    if use_rl:
        print(f"\nRL prediction signal distribution:")
        for signal, count in rl_signals_count.items():
            print(f"  Signal {signal}: {count} times")
    
    # 详细分析几个点
    print(f"\nDetailed analysis of last 10 points:")
    for i in range(max(start_idx, len(df)-10), len(df)):
        print(f"\nPoint {i}:")
        base_signal, base_confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
        print(f"  Base prediction: Signal {base_signal}, Confidence {base_confidence:.4f}")
        
        if use_rl:
            rl_signal, rl_confidence = predictor.predict_signal_with_rl(df.iloc[:i+1], i)
            print(f"  RL prediction: Signal {rl_signal}, Confidence {rl_confidence:.4f}")
            
            # 如果基础预测是开仓信号但RL预测不是，分析原因
            if base_signal in [1, 3] and rl_signal != base_signal:
                print(f"  >> RL model changed opening signal {base_signal} to {rl_signal}")
    
    # 检查模式匹配结果
    print(f"\nAnalyzing pattern matching results...")
    # 检查最后几个点的模式匹配
    for i in range(max(start_idx, len(df)-5), len(df)):
        print(f"\nPattern matching at point {i}:")
        try:
            # 这里我们可以添加更多调试信息
            pattern_result = predictor._match_current_pattern(df.iloc[:i+1], i)
            if pattern_result:
                matched_cluster, similarity, signal_probs = pattern_result
                print(f"  Matched cluster: {matched_cluster}")
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Signal probabilities: {signal_probs}")
            else:
                print(f"  No pattern match")
        except Exception as e:
            print(f"  Error in pattern matching: {e}")

if __name__ == "__main__":
    analyze_predictions()