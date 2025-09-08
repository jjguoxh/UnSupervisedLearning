# -*- coding: utf-8 -*-
"""
简单的强化学习模型保存测试
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_rl_trader import SimpleRLTrader

def test_simple_rl_model_save():
    """
    测试简单强化学习模型的保存
    """
    print("Testing simple RL model save...")
    
    # 创建简单的强化学习交易器
    trader = SimpleRLTrader(learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    
    # 添加一些测试数据到Q表
    trader.q_table["1_5_0_10"] = np.array([0.5, 0.8])
    trader.q_table["3_7_0_10"] = np.array([0.3, 0.9])
    
    # 保存模型
    model_dir = os.path.join(".", "model", "balanced_model")
    model_path = os.path.join(model_dir, "rl_trader_model.json")
    
    print(f"Attempting to save RL model to {model_path}")
    
    try:
        # 将Q表转换为可序列化的格式
        q_table_serializable = {}
        for state, values in trader.q_table.items():
            q_table_serializable[state] = values.tolist()
        
        model_data = {
            'q_table': q_table_serializable,
            'learning_rate': trader.learning_rate,
            'discount_factor': trader.discount_factor,
            'epsilon': trader.epsilon
        }
        
        # 确保目录存在
        print(f"Ensuring model directory exists: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Writing model data to file: {model_path}")
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"RL model saved to {model_path}")
        
        # 检查文件是否存在
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Model file size: {file_size} bytes")
            
            # 读取并验证内容
            with open(model_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            print(f"Loaded model data keys: {list(loaded_data.keys())}")
            print(f"Q-table states: {list(loaded_data['q_table'].keys())}")
            return True
        else:
            print(f"Model file not found at {model_path}")
            return False
            
    except Exception as e:
        print(f"Error saving RL model: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_simple_rl_model_save()
    print(f"Test {'passed' if success else 'failed'}!")
    sys.exit(0 if success else 1)