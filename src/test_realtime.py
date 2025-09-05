# -*- coding: utf-8 -*-
"""
测试实时预测程序
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')
from realtime_predictor import RealtimePredictor

def create_test_data():
    """
    创建测试数据
    """
    # 从现有数据中加载一些样本
    label_files = sorted([f for f in os.listdir("../label") if f.endswith(".csv")])
    if not label_files:
        print("No label files found!")
        return None
        
    # 使用第一个文件作为测试数据
    test_file = os.path.join("../label", label_files[0])
    df = pd.read_csv(test_file)
    
    # 只取前50行作为测试数据
    test_df = df.head(50)
    return test_df

def test_realtime_predictor():
    """
    测试实时预测器
    """
    print("Testing Real-time Predictor")
    print("=" * 30)
    
    # 创建实时预测器
    predictor = RealtimePredictor()
    
    if not predictor.model:
        print("Failed to load model. Exiting.")
        return
    
    # 创建测试数据
    test_df = create_test_data()
    if test_df is None:
        return
        
    print(f"Created test data with {len(test_df)} rows")
    
    # 更新数据缓冲区
    predictor.update_data_buffer(test_df)
    
    print(f"Data buffer updated with {len(predictor.data_buffer)} points")
    
    # 进行预测
    signal, confidence = predictor.predict_signal()
    
    # 保存预测结果
    filepath = predictor.save_prediction(signal, confidence)
    
    # 打印预测结果
    print(f"Prediction: {predictor.get_signal_description(signal)} ({signal}), "
          f"Confidence: {confidence:.3f}")
    print(f"Prediction saved to: {filepath}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_realtime_predictor()