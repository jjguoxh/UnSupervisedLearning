# -*- coding: utf-8 -*-
"""
预测结果分析工具
分析预测结果的准确性和标签分布
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
PATTERNS_DIR = "../patterns/"  # 模式数据目录
PREDICTIONS_DIR = "../predictions/"  # 预测结果目录

def analyze_label_distribution():
    """
    分析标签分布情况
    """
    print("=== 标签分布分析 ===")
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 统计所有标签
    all_labels = []
    for file_path in label_files[:10]:  # 只分析前10个文件
        try:
            df = pd.read_csv(file_path)
            labels = df['label'].values
            all_labels.extend(labels)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 计算标签分布
    label_counts = Counter(all_labels)
    total_labels = len(all_labels)
    
    label_names = {
        0: "无操作",
        1: "做多开仓",
        2: "做多平仓",
        3: "做空开仓",
        4: "做空平仓"
    }
    
    print(f"总标签数: {total_labels}")
    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        percentage = count / total_labels * 100
        name = label_names.get(label, f"未知标签{label}")
        print(f"  {name}({label}): {count} ({percentage:.2f}%)")
    
    return label_counts

def analyze_prediction_results():
    """
    分析预测结果
    """
    print("\n=== 预测结果分析 ===")
    
    # 读取预测结果汇总文件
    predictions_file = os.path.join(PREDICTIONS_DIR, "predictions_summary.csv")
    if not os.path.exists(predictions_file):
        print("未找到预测结果文件")
        return
    
    try:
        df = pd.read_csv(predictions_file)
        print(f"找到 {len(df)} 个预测结果")
        
        # 统计预测标签分布
        pred_counts = Counter(df['predicted_signal'])
        total_predictions = len(df)
        
        label_names = {
            0: "无操作",
            1: "做多开仓",
            2: "做多平仓",
            3: "做空开仓",
            4: "做空平仓"
        }
        
        print("预测标签分布:")
        for label, count in sorted(pred_counts.items()):
            percentage = count / total_predictions * 100
            name = label_names.get(label, f"未知标签{label}")
            print(f"  {name}({label}): {count} ({percentage:.2f}%)")
        
        # 分析置信度分布
        if 'confidence' in df.columns:
            avg_confidence = df['confidence'].mean()
            min_confidence = df['confidence'].min()
            max_confidence = df['confidence'].max()
            
            print(f"\n置信度统计:")
            print(f"  平均置信度: {avg_confidence:.3f}")
            print(f"  最低置信度: {min_confidence:.3f}")
            print(f"  最高置信度: {max_confidence:.3f}")
        
        return pred_counts
    except Exception as e:
        print(f"读取预测结果时出错: {e}")
        return {}

def analyze_cluster_quality():
    """
    分析聚类质量
    """
    print("\n=== 聚类质量分析 ===")
    
    # 读取聚类分析结果
    cluster_file = os.path.join(PATTERNS_DIR, "cluster_analysis.csv")
    if not os.path.exists(cluster_file):
        print("未找到聚类分析文件")
        return
    
    try:
        df = pd.read_csv(cluster_file)
        print(f"找到 {len(df)} 个聚类")
        
        # 分析信号密度
        avg_signal_density = df['signal_density'].mean()
        max_signal_density = df['signal_density'].max()
        min_signal_density = df['signal_density'].min()
        
        print(f"信号密度统计:")
        print(f"  平均信号密度: {avg_signal_density:.3f}")
        print(f"  最高信号密度: {max_signal_density:.3f}")
        print(f"  最低信号密度: {min_signal_density:.3f}")
        
        # 分析各类信号的分布
        total_long_open = df['long_open'].sum()
        total_long_close = df['long_close'].sum()
        total_short_open = df['short_open'].sum()
        total_short_close = df['short_close'].sum()
        
        print(f"\n信号分布:")
        print(f"  做多开仓: {total_long_open}")
        print(f"  做多平仓: {total_long_close}")
        print(f"  做空开仓: {total_short_open}")
        print(f"  做空平仓: {total_short_close}")
        
    except Exception as e:
        print(f"读取聚类分析时出错: {e}")

def check_model_files():
    """
    检查模型文件
    """
    print("\n=== 模型文件检查 ===")
    
    model_dir = "../model/"
    if not os.path.exists(model_dir):
        print("未找到模型目录")
        return
    
    model_files = os.listdir(model_dir)
    print(f"模型目录中的文件: {model_files}")
    
    # 检查预测模型文件
    model_file = os.path.join(model_dir, "pattern_predictor_model.json")
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        print(f"预测模型文件大小: {file_size} 字节")
    else:
        print("未找到预测模型文件")

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 预测结果分析工具")
    print("=" * 50)
    
    # 分析标签分布
    label_counts = analyze_label_distribution()
    
    # 分析预测结果
    pred_counts = analyze_prediction_results()
    
    # 分析聚类质量
    analyze_cluster_quality()
    
    # 检查模型文件
    check_model_files()
    
    # 提供改进建议
    print("\n=== 改进建议 ===")
    if label_counts and pred_counts:
        # 检查标签分布是否不平衡
        short_open_count = label_counts.get(3, 0)
        total_labels = sum(label_counts.values())
        short_open_ratio = short_open_count / total_labels if total_labels > 0 else 0
        
        if short_open_ratio > 0.5:
            print("⚠️  警告: 做空开仓标签(3)在训练数据中占比过高 ({:.2f}%)".format(short_open_ratio * 100))
            print("   建议: 检查标签生成逻辑，确保标签分布平衡")
        
        # 检查预测是否偏向某一类
        pred_short_open_count = pred_counts.get(3, 0)
        total_predictions = sum(pred_counts.values())
        pred_short_open_ratio = pred_short_open_count / total_predictions if total_predictions > 0 else 0
        
        if pred_short_open_ratio > 0.7:
            print("⚠️  警告: 预测结果过度偏向做空开仓标签(3) ({:.2f}%)".format(pred_short_open_ratio * 100))
            print("   建议: 调整模型参数或重新训练模型")
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()