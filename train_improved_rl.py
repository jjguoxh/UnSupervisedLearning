# -*- coding: utf-8 -*-
"""
改进的强化学习训练脚本
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_rl_trader import SimpleRLTrader
from src.pattern_predictor_balanced import BalancedPatternPredictor, load_test_data

def improved_reward_function(action, predicted_signal, actual_signal, price_change, position=0, balance=10000):
    """
    改进的奖励函数
    """
    base_reward = 0
    
    if action == 0:  # 忽略信号
        # 如果信号是正确的但被忽略了，给予较小的负奖励
        if predicted_signal != 0 and predicted_signal == actual_signal:
            base_reward = -0.05  # 降低惩罚
        else:
            base_reward = 0.01  # 给予小的正奖励，鼓励谨慎
    else:  # 执行信号
        if predicted_signal == 0:
            base_reward = -0.1  # 降低惩罚
        elif predicted_signal == actual_signal:
            # 根据价格变化计算收益
            if predicted_signal in [1, 3]:  # 开仓信号
                base_reward = 0.2  # 提高开仓奖励
            elif predicted_signal in [2, 4]:  # 平仓信号
                # 考虑持仓情况和实际收益
                if (predicted_signal == 2 and position == 1) or (predicted_signal == 4 and position == -1):
                    profit_reward = abs(price_change) * 0.8  # 提高收益奖励系数
                    base_reward = 0.3 + profit_reward  # 提高平仓奖励
                else:
                    base_reward = 0.15  # 提高正确信号但持仓不匹配的奖励
        else:
            base_reward = -0.15  # 降低错误执行信号的惩罚
    
    # 考虑资金管理
    if balance < 1000:  # 资金过少时更谨慎
        base_reward *= 0.7
        
    return base_reward

class ImprovedSimpleRLTrader(SimpleRLTrader):
    """
    改进的简单强化学习交易器
    """
    def calculate_reward(self, action, predicted_signal, actual_signal, price_change, position=0, balance=10000):
        """
        使用改进的奖励函数
        """
        return improved_reward_function(action, predicted_signal, actual_signal, price_change, position, balance)

def train_improved_rl_model():
    """
    训练改进的强化学习模型
    """
    print("Training improved RL model...")
    
    # 获取数据文件
    data_files = sorted([f for f in os.listdir("label") if f.endswith(".csv")])
    if not data_files:
        print("No label files found!")
        return False
    
    # 使用多个文件进行训练
    train_files = data_files[:min(5, len(data_files))]  # 使用前5个文件进行训练
    print(f"Training on {len(train_files)} files: {train_files}")
    
    # 加载所有训练数据
    all_train_data = []
    predictor = BalancedPatternPredictor()
    
    for file_name in train_files:
        file_path = os.path.join("label", file_name)
        print(f"Processing {file_path}...")
        
        # 加载数据
        df = load_test_data(file_path)
        if df is None:
            print(f"Failed to load {file_path}")
            continue
        
        # 生成预测信号
        predictions = []
        confidences = []
        
        for i in range(len(df)):
            predicted_signal, confidence = predictor.predict_signal(df, i)
            predictions.append(predicted_signal)
            confidences.append(confidence)
        
        df['predicted_signal'] = predictions
        df['prediction_confidence'] = confidences
        
        all_train_data.append(df)
    
    if not all_train_data:
        print("No training data loaded!")
        return False
    
    # 合并所有训练数据
    combined_data = pd.concat(all_train_data, ignore_index=True)
    print(f"Combined training data size: {len(combined_data)}")
    
    # 创建并训练改进的强化学习交易器
    trader = ImprovedSimpleRLTrader(learning_rate=0.2, discount_factor=0.95, epsilon=0.3)
    
    # 增加训练轮数
    trader.train(combined_data, episodes=100)
    
    # 显示Q表摘要
    print("Q-table summary:")
    q_summary = trader.get_q_table_summary()
    for state, values in q_summary.items():
        print(f"  State '{state}': Ignore={values['ignore']:.4f}, Execute={values['execute']:.4f}, Recommended={values['recommended_action']}")
    
    # 保存模型
    model_dir = os.path.join("model", "balanced_model")
    model_path = os.path.join(model_dir, "improved_rl_trader_model.json")
    
    print(f"Saving improved RL model to {model_path}...")
    
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
        os.makedirs(model_dir, exist_ok=True)
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"Improved RL model saved to {model_path}")
        
        # 验证保存的模型
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"Model file size: {file_size} bytes")
            return True
        else:
            print(f"Model file not found at {model_path}")
            return False
            
    except Exception as e:
        print(f"Error saving improved RL model: {e}")
        return False

def test_improved_rl_model():
    """
    测试改进的强化学习模型
    """
    print("Testing improved RL model...")
    
    # 加载改进的模型
    model_dir = os.path.join("model", "balanced_model")
    model_path = os.path.join(model_dir, "improved_rl_trader_model.json")
    
    if not os.path.exists(model_path):
        print(f"Improved RL model not found at {model_path}")
        return False
    
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        print("Loaded improved model data:")
        print(f"  Learning rate: {model_data['learning_rate']}")
        print(f"  Discount factor: {model_data['discount_factor']}")
        print(f"  Epsilon: {model_data['epsilon']}")
        print(f"  Q-table states: {len(model_data['q_table'])}")
        
        for state, values in model_data['q_table'].items():
            print(f"  State '{state}': Ignore={values[0]:.4f}, Execute={values[1]:.4f}")
            recommended_action = "Execute" if np.argmax(values) == 1 else "Ignore"
            print(f"    Recommended action: {recommended_action}")
            
        return True
    except Exception as e:
        print(f"Error loading improved RL model: {e}")
        return False

if __name__ == "__main__":
    print("=== Improved RL Model Training ===")
    success = train_improved_rl_model()
    
    if success:
        print("\n=== Testing Improved RL Model ===")
        test_improved_rl_model()
        print("\nImproved RL model training and testing completed!")
    else:
        print("\nFailed to train improved RL model!")