# -*- coding: utf-8 -*-
"""
临时解决方案：显示基础预测模型的交易信号（不使用RL模型）
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data
import glob

def show_base_model_signals():
    """
    显示基础预测模型的交易信号（不使用RL模型）
    """
    print("Showing base model trading signals (without RL model)...")
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join("label", "*.csv")))
    if not label_files:
        print("No label files found!")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 处理每个文件
    for file_path in label_files[:3]:  # 只处理前3个文件
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        # 加载数据
        df = pd.read_csv(file_path)
        if df is None:
            print(f"Failed to load {file_path}")
            continue
        
        # 生成预测序列
        predictions = []
        confidences = []
        
        # 为最后100个点生成预测
        start_idx = max(0, len(df) - 100)
        for i in range(start_idx, len(df)):
            predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
            predictions.append(predicted_signal)
            confidences.append(confidence)
        
        # 统计预测信号
        signal_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"  Predicted signal distribution:")
        for signal, count in signal_counts.items():
            print(f"    Signal {signal}: {count} times")
        
        # 找到交易信号点
        trading_signal_points = []
        for i, (idx, signal) in enumerate(zip(range(start_idx, len(df)), predictions)):
            if signal != 0:  # 交易信号
                trading_signal_points.append((idx, signal, confidences[i]))
        
        if trading_signal_points:
            print(f"  Found {len(trading_signal_points)} trading signals:")
            for idx, signal, confidence in trading_signal_points:
                print(f"    Index {idx}: Signal {signal}, Confidence {confidence:.4f}")
        else:
            print(f"  No trading signals found in predictions")
        
        # 保存结果
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        result_data = {
            'file': file_path,
            'predictions': [{'index': idx, 'signal': signal, 'confidence': conf} 
                           for idx, signal, conf in zip(range(start_idx, len(df)), predictions, confidences)],
            'trading_signals': [{'index': idx, 'signal': signal, 'confidence': confidence} 
                               for idx, signal, confidence in trading_signal_points]
        }
        
        output_path = os.path.join(output_dir, f"base_model_signals_{file_name}.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"  Results saved to {output_path}")
        except Exception as e:
            print(f"  Error saving results: {e}")

def interactive_signal_viewer():
    """
    交互式信号查看器
    """
    print("Interactive Signal Viewer (Base Model Only)")
    print("=" * 50)
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join("label", "*.csv")))
    if not label_files:
        print("No label files found!")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    while True:
        print("\n" + "-" * 50)
        print("Available files:")
        for i, file_path in enumerate(label_files[:10]):  # 只显示前10个
            print(f"  {i+1}. {os.path.basename(file_path)}")
        
        if len(label_files) > 10:
            print(f"  ... and {len(label_files) - 10} more files")
        
        print("  0. Exit")
        
        try:
            choice = input("\nSelect a file (0 to exit): ").strip()
            if choice == "0":
                print("Exiting...")
                break
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(label_files):
                file_path = label_files[file_index]
                print(f"\nProcessing {os.path.basename(file_path)}...")
                
                # 加载数据
                df = pd.read_csv(file_path)
                if df is None:
                    print("Failed to load file!")
                    continue
                
                # 显示数据摘要
                print(f"  Data points: {len(df)}")
                label_counts = df['label'].value_counts().sort_index()
                print(f"  Actual label distribution:")
                for label, count in label_counts.items():
                    print(f"    Label {label}: {count} times")
                
                # 生成预测序列（最后100个点）
                predictions = []
                confidences = []
                start_idx = max(0, len(df) - 100)
                
                print(f"  Generating predictions for last 100 points...")
                for i in range(start_idx, len(df)):
                    predicted_signal, confidence = predictor.predict_realtime_signal(df.iloc[:i+1])
                    predictions.append(predicted_signal)
                    confidences.append(confidence)
                
                # 显示预测摘要
                signal_counts = pd.Series(predictions).value_counts().sort_index()
                print(f"  Predicted signal distribution:")
                for signal, count in signal_counts.items():
                    print(f"    Signal {signal}: {count} times")
                
                # 显示详细预测
                show_details = input("  Show detailed predictions? (y/n): ").strip().lower()
                if show_details == 'y':
                    print(f"  Detailed predictions (last 20 points):")
                    for i in range(max(start_idx, len(df)-20), len(df)):
                        pred_idx = i - start_idx
                        actual_label = df.iloc[i]['label']
                        predicted_signal = predictions[pred_idx]
                        confidence = confidences[pred_idx]
                        match = "✓" if actual_label == predicted_signal else "✗"
                        print(f"    Index {i}: Actual={actual_label}, Predicted={predicted_signal}, Confidence={confidence:.4f} {match}")
                
                # 保存结果
                save_results = input("  Save results? (y/n): ").strip().lower()
                if save_results == 'y':
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_dir = "predictions"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    result_data = {
                        'file': file_path,
                        'predictions': [{'index': idx, 'signal': signal, 'confidence': conf} 
                                       for idx, signal, conf in zip(range(start_idx, len(df)), predictions, confidences)],
                    }
                    
                    output_path = os.path.join(output_dir, f"interactive_base_model_{file_name}.json")
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data, f, ensure_ascii=False, indent=2)
                        print(f"  Results saved to {output_path}")
                    except Exception as e:
                        print(f"  Error saving results: {e}")
            else:
                print("Invalid selection!")
        except ValueError:
            print("Invalid input!")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Show base model signals for all files")
    print("2. Interactive signal viewer")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        show_base_model_signals()
    elif choice == "2":
        interactive_signal_viewer()
    else:
        print("Invalid choice!")