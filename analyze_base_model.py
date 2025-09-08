# -*- coding: utf-8 -*-
"""
详细分析基础预测模型的质量
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def analyze_base_model_quality():
    """
    详细分析基础预测模型的质量
    """
    print("Analyzing base model quality...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
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
    
    # 检查实际标签分布
    label_counts = df['label'].value_counts().sort_index()
    print(f"\nActual label distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} times")
    
    # 进行预测分析
    print(f"\nAnalyzing predictions...")
    predictions = []
    confidences = []
    
    # 为所有点生成预测
    for i in range(len(df)):
        predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
        predictions.append(predicted_signal)
        confidences.append(confidence)
    
    # 计算准确率
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(df)):
        actual_label = df.iloc[i]['label']
        predicted_label = predictions[i]
        
        if actual_label != 0:  # 只计算非无操作标签的准确率
            total_predictions += 1
            if actual_label == predicted_label:
                correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nPrediction accuracy for non-zero labels:")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # 预测信号分布
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    print(f"\nPredicted signal distribution:")
    for signal, count in pred_counts.items():
        print(f"  Signal {signal}: {count} times")
    
    # 详细分析错误类型
    print(f"\nDetailed error analysis:")
    false_positives = 0  # 预测为信号但实际为0
    false_negatives = 0  # 实际为信号但预测为0或其他信号
    wrong_signals = 0    # 预测信号类型错误
    
    for i in range(len(df)):
        actual_label = df.iloc[i]['label']
        predicted_label = predictions[i]
        
        if actual_label == 0 and predicted_label != 0:
            false_positives += 1
        elif actual_label != 0 and predicted_label == 0:
            false_negatives += 1
        elif actual_label != 0 and predicted_label != 0 and actual_label != predicted_label:
            wrong_signals += 1
    
    print(f"  False positives (predicted signal but actual 0): {false_positives}")
    print(f"  False negatives (actual signal but predicted 0): {false_negatives}")
    print(f"  Wrong signals (predicted signal type wrong): {wrong_signals}")
    
    # 置信度分析
    confidences_array = np.array(confidences)
    print(f"\nConfidence analysis:")
    print(f"  Mean confidence: {np.mean(confidences_array):.4f}")
    print(f"  Std confidence: {np.std(confidences_array):.4f}")
    print(f"  Min confidence: {np.min(confidences_array):.4f}")
    print(f"  Max confidence: {np.max(confidences_array):.4f}")
    
    # 保存详细结果
    result_data = {
        'index': list(range(len(df))),
        'actual_label': df['label'].tolist(),
        'predicted_signal': predictions,
        'confidence': confidences
    }
    
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(test_file))[0]
    output_path = os.path.join(output_dir, f"detailed_analysis_{file_name}.json")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed analysis saved to {output_path}")
    except Exception as e:
        print(f"\nError saving detailed analysis: {e}")

if __name__ == "__main__":
    analyze_base_model_quality()