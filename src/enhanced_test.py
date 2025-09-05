# -*- coding: utf-8 -*-
"""
增强测试脚本 - 使用具有明显交易信号的数据测试预测程序
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')
from realtime_predictor import RealtimePredictor

def find_data_with_signals():
    """
    查找包含交易信号的数据文件
    """
    # 遍历标签目录中的所有文件
    label_dir = "../label"
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".csv")])
    
    # 查找包含交易信号的文件
    for file_name in label_files:
        file_path = os.path.join(label_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            # 检查是否有非零标签
            if (df['label'] != 0).any():
                signal_count = (df['label'] != 0).sum()
                print(f"Found file {file_name} with {signal_count} trading signals")
                return df, file_name
        except Exception as e:
            continue
    
    # 如果没有找到包含信号的文件，返回第一个文件
    if label_files:
        file_path = os.path.join(label_dir, label_files[0])
        df = pd.read_csv(file_path)
        print(f"Using {label_files[0]} as test data (no signals found in other files)")
        return df, label_files[0]
    
    return None, None

def test_with_real_signals():
    """
    使用真实交易信号测试预测程序
    """
    print("Enhanced Real-time Predictor Test")
    print("=" * 40)
    
    # 创建实时预测器
    predictor = RealtimePredictor()
    
    if not predictor.model:
        print("Failed to load model. Exiting.")
        return
    
    # 查找包含交易信号的数据
    test_df, file_name = find_data_with_signals()
    if test_df is None:
        print("No test data found!")
        return
    
    print(f"Using file: {file_name}")
    print(f"Data shape: {test_df.shape}")
    
    # 检查标签分布
    label_counts = test_df['label'].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    
    # 更新数据缓冲区（使用更多数据点）
    predictor.update_data_buffer(test_df)
    
    print(f"Data buffer updated with {len(predictor.data_buffer)} points")
    
    # 进行多次预测，使用不同的数据点
    print("\nMaking predictions...")
    predictions = []
    
    # 使用最后几个数据点进行预测
    for i in range(max(0, len(test_df) - 5), len(test_df)):
        # 重新初始化预测器以避免缓冲区影响
        temp_predictor = RealtimePredictor()
        temp_predictor.update_data_buffer(test_df.head(i+1))
        
        signal, confidence = temp_predictor.predict_signal()
        actual_label = test_df.iloc[i]['label'] if i < len(test_df) else 0
        
        predictions.append({
            'index': i,
            'predicted_signal': signal,
            'actual_signal': actual_label,
            'confidence': confidence
        })
        
        # 保存预测结果
        filepath = temp_predictor.save_prediction(signal, confidence)
        
        print(f"Index {i}: Predicted={signal} (conf: {confidence:.3f}), "
              f"Actual={actual_label}, Saved to: {os.path.basename(filepath)}")
    
    # 计算准确率
    correct = sum(1 for p in predictions if p['predicted_signal'] == p['actual_signal'])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nPrediction Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    # 显示详细预测结果
    print("\nDetailed Predictions:")
    print("-" * 50)
    signal_names = {
        0: "无操作",
        1: "做空开仓",
        2: "做空平仓",
        3: "做多开仓",
        4: "做多平仓"
    }
    
    for pred in predictions:
        print(f"Index {pred['index']:3d}: "
              f"Predicted={signal_names.get(pred['predicted_signal'], 'Unknown')}({pred['predicted_signal']}) "
              f"Actual={signal_names.get(pred['actual_signal'], 'Unknown')}({pred['actual_signal']}) "
              f"Confidence={pred['confidence']:.3f}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_with_real_signals()