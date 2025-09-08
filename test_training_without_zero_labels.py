# -*- coding: utf-8 -*-
"""
测试脚本：验证在训练时去除0特征信号，仅保留1,2,3,4信号的训练
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

from src.pattern_recognition import process_single_file as pattern_process_single_file
from src.trading_pattern_learning import process_single_file as learning_process_single_file

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pattern_recognition_without_zero_labels():
    """
    测试模式识别是否正确排除了0标签信号
    """
    logger.info("Testing pattern recognition without zero labels...")
    
    # 获取标签文件
    label_dir = "./label/"
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
    
    if not label_files:
        logger.error("No label files found!")
        return False
    
    # 测试第一个文件
    test_file = label_files[0]
    logger.info(f"Testing with file: {test_file}")
    
    try:
        # 加载数据
        df = pd.read_csv(test_file)
        logger.info(f"Loaded data with {len(df)} rows")
        
        # 检查标签分布
        label_counts = Counter(df['label'])
        logger.info(f"Original label distribution: {dict(label_counts)}")
        
        # 处理单个文件
        patterns, clusters, cluster_analysis = pattern_process_single_file(test_file)
        
        if patterns is None:
            logger.error("Failed to process file in pattern recognition")
            return False
            
        logger.info(f"Found {len(patterns)} patterns")
        logger.info(f"Found {len(clusters)} clusters")
        
        # 检查所有模式中的信号标签
        pattern_labels = [pattern['signal_label'] for pattern in patterns]
        pattern_label_counts = Counter(pattern_labels)
        logger.info(f"Pattern label distribution: {dict(pattern_label_counts)}")
        
        # 验证是否没有0标签
        if 0 in pattern_label_counts:
            logger.error("Found 0 labels in patterns, which should have been excluded!")
            return False
        else:
            logger.info("Successfully excluded 0 labels from patterns")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in pattern recognition test: {e}")
        return False

def test_trading_pattern_learning_without_zero_labels():
    """
    测试交易模式学习是否正确排除了0标签信号
    """
    logger.info("Testing trading pattern learning without zero labels...")
    
    # 获取标签文件
    label_dir = "./label/"
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
    
    if not label_files:
        logger.error("No label files found!")
        return False
    
    # 测试第一个文件
    test_file = label_files[0]
    logger.info(f"Testing with file: {test_file}")
    
    try:
        # 处理单个文件
        windows, cluster_labels, cluster_profit, features_normalized, signal_indices = learning_process_single_file(test_file)
        
        if len(windows) == 0:
            logger.error("No windows found in trading pattern learning")
            return False
            
        logger.info(f"Found {len(windows)} windows")
        logger.info(f"Found {len(cluster_labels)} cluster labels")
        logger.info(f"Found {len(signal_indices)} signal indices")
        
        # 加载原始数据以检查信号标签
        df = pd.read_csv(test_file)
        signal_labels = [df['label'].iloc[idx] for idx in signal_indices]
        signal_label_counts = Counter(signal_labels)
        logger.info(f"Signal label distribution in windows: {dict(signal_label_counts)}")
        
        # 验证是否没有0标签
        if 0 in signal_label_counts:
            logger.error("Found 0 labels in signal windows, which should have been excluded!")
            return False
        else:
            logger.info("Successfully excluded 0 labels from signal windows")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in trading pattern learning test: {e}")
        return False

def main():
    """
    主函数
    """
    logger.info("Starting test for training without zero labels...")
    
    # 测试模式识别
    pattern_test_passed = test_pattern_recognition_without_zero_labels()
    
    # 测试交易模式学习
    learning_test_passed = test_trading_pattern_learning_without_zero_labels()
    
    # 总结结果
    if pattern_test_passed and learning_test_passed:
        logger.info("All tests passed! Training now excludes 0 labels and only uses 1,2,3,4 signals.")
        return True
    else:
        logger.error("Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)