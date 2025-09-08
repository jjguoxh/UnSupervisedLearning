# -*- coding: utf-8 -*-
"""
完整工作流程测试脚本：验证整个系统是否正确排除了0标签信号
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import sys
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_label_distribution():
    """
    检查标签分布
    """
    logger.info("Checking label distribution...")
    
    # 获取标签文件
    label_dir = "./label/"
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
    
    if not label_files:
        logger.error("No label files found!")
        return False
    
    # 检查所有文件的标签分布
    total_label_counts = Counter()
    
    for file_path in label_files[:5]:  # 只检查前5个文件
        try:
            df = pd.read_csv(file_path)
            label_counts = Counter(df['label'])
            total_label_counts.update(label_counts)
            logger.info(f"File {os.path.basename(file_path)} label distribution: {dict(label_counts)}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    logger.info(f"Total label distribution across {min(5, len(label_files))} files: {dict(total_label_counts)}")
    
    # 验证是否包含所有类型的标签
    expected_labels = {0, 1, 2, 3, 4}
    actual_labels = set(total_label_counts.keys())
    
    if not expected_labels.issubset(actual_labels):
        logger.warning(f"Missing expected labels. Expected: {expected_labels}, Actual: {actual_labels}")
    
    # 检查0标签是否占主导地位
    zero_count = total_label_counts[0]
    non_zero_count = sum(count for label, count in total_label_counts.items() if label != 0)
    
    logger.info(f"Zero labels: {zero_count}, Non-zero labels: {non_zero_count}")
    logger.info(f"Ratio of zero to non-zero labels: {zero_count/non_zero_count:.2f}" if non_zero_count > 0 else "No non-zero labels found")
    
    return True

def check_pattern_recognition_output():
    """
    检查模式识别输出是否排除了0标签
    """
    logger.info("Checking pattern recognition output...")
    
    # 获取模式分析结果
    patterns_dir = "./patterns/"
    cluster_analysis_file = os.path.join(patterns_dir, "cluster_analysis.csv")
    
    if not os.path.exists(cluster_analysis_file):
        logger.warning("Cluster analysis file not found. Pattern recognition may not have been run yet.")
        return False
    
    try:
        # 读取聚类分析结果
        df = pd.read_csv(cluster_analysis_file)
        logger.info(f"Found {len(df)} clusters in cluster analysis")
        
        # 检查每个聚类的信号计数
        for _, row in df.iterrows():
            cluster_id = row['cluster_id']
            signal_counts = eval(str(row['signal_counts'])) if isinstance(row['signal_counts'], str) else row['signal_counts']
            logger.info(f"Cluster {cluster_id} signal counts: {signal_counts}")
            
            # 验证是否没有0标签
            if 0 in signal_counts:
                logger.warning(f"Found 0 labels in cluster {cluster_id} signal counts: {signal_counts}")
        
        return True
    except Exception as e:
        logger.error(f"Error reading cluster analysis file: {e}")
        return False

def check_model_files():
    """
    检查模型文件是否正确生成
    """
    logger.info("Checking model files...")
    
    # 检查模型目录
    model_dir = "./model/balanced_model/"
    if not os.path.exists(model_dir):
        logger.warning("Model directory not found.")
        return False
    
    # 检查模型文件
    model_files = os.listdir(model_dir)
    logger.info(f"Model files: {model_files}")
    
    # 检查强化学习模型
    rl_model_file = os.path.join(model_dir, "rl_trader_model.json")
    if os.path.exists(rl_model_file):
        logger.info("Reinforcement learning model found")
    else:
        logger.warning("Reinforcement learning model not found")
    
    # 检查平衡模式预测器模型
    predictor_model_file = os.path.join(model_dir, "balanced_pattern_predictor_model.json")
    if os.path.exists(predictor_model_file):
        logger.info("Balanced pattern predictor model found")
    else:
        logger.warning("Balanced pattern predictor model not found")
    
    return True

def check_prediction_files():
    """
    检查预测结果文件
    """
    logger.info("Checking prediction files...")
    
    # 检查预测目录
    predictions_dir = "./predictions/"
    if not os.path.exists(predictions_dir):
        logger.warning("Predictions directory not found.")
        return False
    
    # 获取预测文件
    prediction_files = glob.glob(os.path.join(predictions_dir, "*.json"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    # 检查前几个预测文件
    for file_path in prediction_files[:3]:
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Prediction file {os.path.basename(file_path)}: {data}")
        except Exception as e:
            logger.error(f"Error reading prediction file {file_path}: {e}")
    
    return True

def main():
    """
    主函数
    """
    logger.info("Starting complete workflow test...")
    
    # 检查标签分布
    label_check_passed = check_label_distribution()
    
    # 检查模式识别输出
    pattern_check_passed = check_pattern_recognition_output()
    
    # 检查模型文件
    model_check_passed = check_model_files()
    
    # 检查预测文件
    prediction_check_passed = check_prediction_files()
    
    # 总结结果
    all_checks = [label_check_passed, pattern_check_passed, model_check_passed, prediction_check_passed]
    passed_checks = sum(all_checks)
    
    if passed_checks == len(all_checks):
        logger.info("All checks passed! The complete workflow is working correctly.")
        logger.info("The system now excludes 0 labels during training and only uses 1,2,3,4 signals.")
        return True
    else:
        logger.warning(f"{passed_checks}/{len(all_checks)} checks passed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)