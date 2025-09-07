# -*- coding: utf-8 -*-
"""
深度强化学习交易器
使用神经网络学习复杂的交易策略
"""

import numpy as np
import pandas as pd
import random
import os
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, using simple Q-learning")

class DeepRLTrader:
    """
    深度强化学习交易器
    """
    def __init__(self, state_size=10, action_size=2, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        
        # 参数
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # 折扣因子
        
        # 创建模型
        if TENSORFLOW_AVAILABLE:
            self.model = self._build_model()
        else:
            # 如果没有TensorFlow，使用简单的Q表
            from collections import defaultdict
            self.q_table = defaultdict(lambda: np.zeros(action_size))
            self.model = None
    
    def _build_model(self):
        """
        构建神经网络模型
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        存储经验
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前状态选择动作
        """
        # ε-贪婪策略
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 使用模型预测动作
        if self.model is not None:
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return np.argmax(act_values[0])
        else:
            # 使用Q表
            state_key = tuple(state)
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            else:
                return random.randrange(self.action_size)
    
    def replay(self, batch_size=32):
        """
        经验回放训练
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                if self.model is not None:
                    target = (reward + self.gamma * 
                             np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0]))
                else:
                    # Q表更新
                    state_key = tuple(state)
                    next_state_key = tuple(next_state)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    if next_state_key not in self.q_table:
                        self.q_table[next_state_key] = np.zeros(self.action_size)
                    target = (reward + self.gamma * np.max(self.q_table[next_state_key]))
            
            if self.model is not None:
                target_f = self.model.predict(state.reshape(1, -1), verbose=0)
                target_f[0][action] = target
                self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            else:
                # 更新Q表
                state_key = tuple(state)
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.action_size)
                self.q_table[state_key][action] = (
                    self.q_table[state_key][action] + 
                    0.1 * (target - self.q_table[state_key][action])
                )
        
        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def state_to_vector(predicted_signal, confidence, position, balance, price_change, 
                   a, b, c, d, index_value):
    """
    将状态转换为向量
    """
    # 标准化特征
    features = [
        predicted_signal / 4.0,  # 标准化信号 (0-4)
        confidence,  # 置信度 (0-1)
        position,  # 持仓状态 (-1, 0, 1)
        balance / 10000.0,  # 标准化余额
        price_change / 100.0,  # 标准化价格变化
        a, b, c, d,  # 技术指标
        index_value / 1000.0  # 标准化价格
    ]
    
    return np.array(features)

def train_deep_rl_agent(data, episodes=100):
    """
    训练深度强化学习代理
    """
    logger.info("开始训练深度强化学习代理")
    
    # 创建代理
    agent = DeepRLTrader(state_size=10, action_size=2)
    
    scores = []
    for e in range(episodes):
        total_reward = 0
        position = 0
        balance = 10000
        
        # 初始状态
        if len(data) > 0:
            state = state_to_vector(
                data.iloc[0]['predicted_signal'],
                data.iloc[0]['prediction_confidence'],
                position,
                balance,
                0,  # 初始价格变化
                data.iloc[0]['a'],
                data.iloc[0]['b'],
                data.iloc[0]['c'],
                data.iloc[0]['d'],
                data.iloc[0]['index_value']
            )
        else:
            continue
        
        for i in range(1, len(data)):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作并获取奖励
            next_row = data.iloc[i]
            price_change = next_row['index_value'] - data.iloc[i-1]['index_value']
            
            # 简化的奖励计算
            reward = 0
            if action == 1 and next_row['predicted_signal'] != 0 and next_row['predicted_signal'] == next_row['label']:
                reward = 0.1
            elif action == 1 and (next_row['predicted_signal'] == 0 or next_row['predicted_signal'] != next_row['label']):
                reward = -0.1
            
            # 更新持仓和余额（简化）
            if action == 1:
                position = 1 if next_row['predicted_signal'] in [1, 3] else 0
            
            # 下一个状态
            next_state = state_to_vector(
                next_row['predicted_signal'],
                next_row['prediction_confidence'],
                position,
                balance,
                price_change,
                next_row['a'],
                next_row['b'],
                next_row['c'],
                next_row['d'],
                next_row['index_value']
            )
            
            # 存储经验
            done = (i == len(data) - 1)
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_reward += reward
        
        # 经验回放训练
        agent.replay(32)
        
        scores.append(total_reward)
        if (e + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            logger.info(f"Episode {e+1}/{episodes}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    logger.info("训练完成")
    return agent

def main():
    """
    主函数
    """
    import glob
    from simple_rl_trader import load_data_with_predictions
    
    # 获取数据文件
    data_files = sorted(glob.glob(os.path.join(".", "label", "*.csv")))
    
    logger.info(f"找到 {len(data_files)} 个数据文件")
    
    if not data_files:
        logger.error("未找到数据文件!")
        return
    
    # 使用第一个文件进行训练
    logger.info("=== 训练阶段 ===")
    train_data = load_data_with_predictions(data_files[0])
    if train_data is None:
        logger.error("训练数据加载失败")
        return
    
    # 训练深度强化学习代理
    agent = train_deep_rl_agent(train_data, episodes=50)
    
    logger.info("深度强化学习代理训练完成")

if __name__ == "__main__":
    main()