# -*- coding: utf-8 -*-
"""
预测问题诊断工具
专门分析模型持续预测"做空开仓"的问题
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

# ========= 配置参数 =========
# 使用相对于脚本位置的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "..", "label/")  # 标签数据目录
PATTERNS_DIR = os.path.join(CURRENT_DIR, "..", "patterns/")  # 模式数据目录
PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "..", "predictions/")  # 预测结果目录
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "model/")  # 模型目录

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
    for file_path in label_files[:20]:  # 只分析前20个文件
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
        
        # 检查最近的预测结果
        print(f"\n最近的预测结果:")
        recent_predictions = df.tail(10)
        for idx, row in recent_predictions.iterrows():
            print(f"  {row['timestamp']}: Predicted={label_names.get(row['predicted_signal'], row['predicted_signal'])} ({row['predicted_signal']}), Confidence={row['confidence']:.3f}")
        
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
        
        total_signals = total_long_open + total_long_close + total_short_open + total_short_close
        
        print(f"\n信号分布:")
        print(f"  做多开仓: {total_long_open} ({total_long_open/total_signals*100:.2f}%)")
        print(f"  做多平仓: {total_long_close} ({total_long_close/total_signals*100:.2f}%)")
        print(f"  做空开仓: {total_short_open} ({total_short_open/total_signals*100:.2f}%)")
        print(f"  做空平仓: {total_short_close} ({total_short_close/total_signals*100:.2f}%)")
        
        # 检查是否存在模式偏向
        if total_short_open / total_signals > 0.4:
            print("⚠️  警告: 模式中做空开仓信号占比过高")
        
    except Exception as e:
        print(f"读取聚类分析时出错: {e}")

def check_model_files():
    """
    检查模型文件
    """
    print("\n=== 模型文件检查 ===")
    
    if not os.path.exists(MODEL_DIR):
        print("未找到模型目录")
        return
    
    model_files = os.listdir(MODEL_DIR)
    print(f"模型目录中的文件: {model_files}")
    
    # 检查预测模型文件
    model_file = os.path.join(MODEL_DIR, "pattern_predictor_model.json")
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        print(f"预测模型文件大小: {file_size} 字节")
        
        # 如果文件较小，可能表示模型没有充分训练
        if file_size < 10000:
            print("⚠️  警告: 模型文件过小，可能未充分训练")
    else:
        print("未找到预测模型文件")

def diagnose_prediction_issues():
    """
    诊断预测问题
    """
    print("\n=== 预测问题诊断 ===")
    
    # 1. 检查标签分布
    label_counts = analyze_label_distribution()
    
    # 2. 检查预测结果
    pred_counts = analyze_prediction_results()
    
    # 3. 检查模型模式
    analyze_cluster_quality()
    
    # 4. 检查模型文件
    check_model_files()
    
    # 5. 提供具体建议
    print("\n=== 问题诊断与改进建议 ===")
    
    if label_counts and pred_counts:
        # 检查训练数据和预测结果的差异
        short_open_in_training = label_counts.get(3, 0) / sum(label_counts.values()) if sum(label_counts.values()) > 0 else 0
        short_open_in_prediction = pred_counts.get(3, 0) / sum(pred_counts.values()) if sum(pred_counts.values()) > 0 else 0
        
        print(f"做空开仓在训练数据中的比例: {short_open_in_training:.2%}")
        print(f"做空开仓在预测结果中的比例: {short_open_in_prediction:.2%}")
        
        if abs(short_open_in_prediction - short_open_in_training) > 0.3:
            print("⚠️  警告: 预测结果与训练数据分布存在显著差异")
            print("   建议:")
            print("   1. 重新检查标签生成逻辑，确保标签分布合理")
            print("   2. 调整模型参数，增加训练轮次")
            print("   3. 检查模式识别算法，确保模式质量")
        
        # 检查预测置信度
        # 读取预测结果文件检查置信度
        predictions_file = os.path.join(PREDICTIONS_DIR, "predictions_summary.csv")
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            if 'confidence' in df.columns and len(df) > 0:
                avg_confidence = df['confidence'].mean()
                print(f"平均预测置信度: {avg_confidence:.3f}")
                if avg_confidence < 0.1:
                    print("⚠️  警告: 预测置信度过低，模型可能未充分学习")
                    print("   建议:")
                    print("   1. 增加训练数据量")
                    print("   2. 调整模型超参数")
                    print("   3. 检查特征工程")
    
    print("\n其他可能的改进措施:")
    print("1. 检查标签生成逻辑，确保标签0正确表示无操作状态")
    print("2. 增加更多样化的训练数据")
    print("3. 调整聚类算法参数，提高模式质量")
    print("4. 优化特征提取方法，提高特征表达能力")
    print("5. 考虑使用集成学习方法提高预测稳定性")

def main():
    """
    主函数
    """
    print("无监督学习交易信号识别系统 - 预测问题诊断工具")
    print("=" * 60)
    
    # 执行详细诊断
    diagnose_prediction_issues()
    
    print("\n" + "=" * 60)
    print("诊断完成!")

if __name__ == "__main__":
    main()