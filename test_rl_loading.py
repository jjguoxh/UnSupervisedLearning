# -*- coding: utf-8 -*-
"""
测试realtime_predictor中的强化学习模型加载功能
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor

def test_rl_model_loading():
    """
    测试强化学习模型加载
    """
    print("Testing RL model loading in realtime_predictor...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    CURRENT_DIR = os.path.dirname(os.path.abspath('src/realtime_predictor.py'))
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    
    print(f"RL model path: {rl_model_path}")
    print(f"RL model file exists: {os.path.exists(rl_model_path)}")
    
    # 尝试加载模型
    success = predictor.load_rl_model(rl_model_path)
    
    if success:
        print("Successfully loaded RL model!")
        print(f"RL trader exists: {predictor.rl_trader is not None}")
        if predictor.rl_trader:
            print(f"Q-table size: {len(predictor.rl_trader.q_table)}")
    else:
        print("Failed to load RL model!")
        print("Checking if file is readable...")
        if os.path.exists(rl_model_path):
            try:
                with open(rl_model_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"File content length: {len(content)}")
                    print("First 200 chars:", content[:200])
            except Exception as e:
                print(f"Error reading file: {e}")

if __name__ == "__main__":
    test_rl_model_loading()