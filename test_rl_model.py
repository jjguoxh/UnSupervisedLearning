# -*- coding: utf-8 -*-
"""
测试强化学习模型的保存和加载
"""

import os
import sys
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pattern_predictor_balanced import BalancedPatternPredictor
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rl_model_save_load():
    """
    测试强化学习模型的保存和加载
    """
    logger.info("Testing RL model save and load...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 获取测试数据文件
    label_files = sorted(glob.glob(os.path.join(".", "label", "*.csv")))
    logger.info(f"Found {len(label_files)} label files")
    
    if not label_files:
        logger.error("No label files found!")
        return False
    
    # 训练强化学习模型（使用第一个文件）
    logger.info("Training reinforcement learning model...")
    success = predictor.train_rl_model(label_files[0])
    
    if not success:
        logger.error("Failed to train RL model")
        return False
    
    # 保存强化学习模型
    logger.info("Saving reinforcement learning model...")
    save_success = predictor.save_rl_model()
    
    if not save_success:
        logger.error("Failed to save RL model")
        return False
    
    # 检查模型文件是否存在
    model_dir = os.path.join(".", "model", "balanced_model")
    model_file = os.path.join(model_dir, "rl_trader_model.json")
    
    if os.path.exists(model_file):
        logger.info(f"RL model saved successfully to {model_file}")
        file_size = os.path.getsize(model_file)
        logger.info(f"Model file size: {file_size} bytes")
    else:
        logger.error(f"RL model file not found at {model_file}")
        return False
    
    # 创建新的预测器并加载模型
    logger.info("Creating new predictor and loading RL model...")
    new_predictor = BalancedPatternPredictor()
    load_success = new_predictor.load_rl_model()
    
    if load_success:
        logger.info("RL model loaded successfully")
    else:
        logger.error("Failed to load RL model")
        return False
    
    return True

def main():
    """
    主函数
    """
    logger.info("Starting RL model save/load test...")
    
    success = test_rl_model_save_load()
    
    if success:
        logger.info("RL model save/load test passed!")
        return True
    else:
        logger.error("RL model save/load test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)