# -*- coding: utf-8 -*-
"""
分析和改进训练数据质量
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_training_data_quality():
    """
    分析训练数据质量
    """
    print("Analyzing training data quality...")
    
    # 获取所有标签文件
    label_files = sorted([f for f in os.listdir("label") if f.endswith(".csv")])
    if not label_files:
        print("No label files found!")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # 统计所有文件的标签分布
    total_labels = []
    signal_distribution = Counter()
    
    for file_name in label_files[:10]:  # 分析前10个文件
        file_path = os.path.join("label", file_name)
        try:
            df = pd.read_csv(file_path)
            labels = df['label'].tolist()
            total_labels.extend(labels)
            signal_distribution.update(labels)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    print(f"\nOverall label distribution:")
    for label, count in sorted(signal_distribution.items()):
        percentage = count / len(total_labels) * 100
        print(f"  Label {label}: {count} times ({percentage:.2f}%)")
    
    # 分析信号平衡性
    trading_signals = [label for label in total_labels if label != 0]
    print(f"\nTrading signal analysis:")
    print(f"  Total trading signals: {len(trading_signals)}")
    print(f"  Total data points: {len(total_labels)}")
    print(f"  Trading signal ratio: {len(trading_signals)/len(total_labels)*100:.2f}%")
    
    if len(trading_signals) > 0:
        trading_dist = Counter(trading_signals)
        print(f"  Trading signal distribution:")
        for signal, count in sorted(trading_dist.items()):
            percentage = count / len(trading_signals) * 100
            print(f"    Signal {signal}: {count} times ({percentage:.2f}%)")

def improve_data_quality():
    """
    改进数据质量的策略
    """
    print("\nData quality improvement strategies:")
    print("1. 数据增强:")
    print("   - 增加更多历史数据用于训练")
    print("   - 使用数据合成技术生成更多样例")
    print("   - 平衡各类信号的样本数量")
    
    print("\n2. 特征工程:")
    print("   - 增加更多技术指标作为特征")
    print("   - 使用滑动窗口统计特征")
    print("   - 添加市场状态特征（趋势、波动率等）")
    
    print("\n3. 标签优化:")
    print("   - 重新审视标签生成逻辑")
    print("   - 考虑使用更精细的信号分类")
    print("   - 引入置信度标签")

if __name__ == "__main__":
    analyze_training_data_quality()
    improve_data_quality()