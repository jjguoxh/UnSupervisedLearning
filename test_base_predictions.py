# -*- coding: utf-8 -*-
"""
直接测试基础预测模型，不使用强化学习模型
"""

import os
import sys
import json
import pandas as pd

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def test_base_predictions():
    """
    测试基础预测模型，不使用强化学习模型
    """
    print("Testing base predictions without RL model...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 注意：我们不加载强化学习模型
    
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
    print("\nAnalyzing base predictions at multiple points...")
    signals_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # 分析最后100个点
    start_idx = max(0, len(df) - 100)
    sequence_predictions = []
    
    for i in range(start_idx, len(df)):
        # 基础预测
        predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
        signals_count[predicted_signal] += 1
        
        sequence_predictions.append({
            'index': i,
            'predicted_signal': predicted_signal,
            'confidence': confidence
        })
    
    print(f"\nBase prediction signal distribution:")
    for signal, count in signals_count.items():
        print(f"  Signal {signal}: {count} times")
    
    # 详细分析几个点
    print(f"\nDetailed analysis of last 10 points:")
    for i in range(max(start_idx, len(df)-10), len(df)):
        predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
        print(f"  Point {i}: Signal {predicted_signal}, Confidence {confidence:.4f}")
    
    # 保存结果
    file_name = os.path.splitext(os.path.basename(test_file))[0]
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"base_prediction_{file_name}.json")
    result = {
        'file': test_file,
        'sequence_predictions': sequence_predictions,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nBase predictions saved to {output_path}")
    except Exception as e:
        print(f"\nError saving base predictions: {e}")

if __name__ == "__main__":
    test_base_predictions()