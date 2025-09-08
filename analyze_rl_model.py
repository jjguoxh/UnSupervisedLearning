# -*- coding: utf-8 -*-
"""
分析强化学习模型的Q表内容
"""

import os
import sys
import json
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor

def analyze_rl_model():
    """
    分析强化学习模型的Q表内容
    """
    print("Analyzing RL model Q-table content...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    CURRENT_DIR = os.path.dirname(os.path.abspath('src/realtime_predictor.py'))
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    
    print(f"RL model path: {rl_model_path}")
    print(f"RL model file exists: {os.path.exists(rl_model_path)}")
    
    # 尝试加载模型
    use_rl = predictor.load_rl_model(rl_model_path)
    
    if use_rl:
        print("Successfully loaded RL model!")
        print(f"RL trader exists: {predictor.rl_trader is not None}")
        if predictor.rl_trader:
            print(f"Q-table size: {len(predictor.rl_trader.q_table)}")
            print("Q-table keys:", list(predictor.rl_trader.q_table.keys()))
            
            # 详细打印Q表内容
            for state, values in predictor.rl_trader.q_table.items():
                print(f"  State '{state}': Ignore={values[0]:.4f}, Execute={values[1]:.4f}")
                recommended_action = "Execute" if np.argmax(values) == 1 else "Ignore"
                print(f"    Recommended action: {recommended_action}")
    else:
        print("Failed to load RL model!")
        
        # 直接读取JSON文件查看内容
        print("\nReading RL model file directly:")
        try:
            with open(rl_model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            print("Model data keys:", list(model_data.keys()))
            print("Q-table keys:", list(model_data['q_table'].keys()))
            for state, values in model_data['q_table'].items():
                print(f"  State '{state}': Ignore={values[0]:.4f}, Execute={values[1]:.4f}")
        except Exception as e:
            print(f"Error reading model file directly: {e}")

if __name__ == "__main__":
    analyze_rl_model()