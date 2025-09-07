# -*- coding: utf-8 -*-
"""
基于强化学习的交易策略优化器
使用预测信号作为输入，学习最优交易策略
"""

import numpy as np
import pandas as pd
import random
import os
import json
import logging
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnvironment:
    """
    交易环境类
    """
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """
        重置环境状态
        """
        self.balance = self.initial_balance
        self.position = 0  # 0: 无持仓, 1: 多头, -1: 空头
        self.position_price = 0
        self.current_step = 0
        self.net_worth_history = [self.initial_balance]
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """
        获取当前状态
        """
        if self.current_step >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_step]
        state = {
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price,
            'index_value': row['index_value'],
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'd': row['d'],
            'x': row['x'],
            'predicted_signal': row.get('predicted_signal', 0),
            'actual_signal': row.get('label', 0)
        }
        return state
    
    def step(self, action):
        """
        执行动作并返回结果
        动作定义: 0-持有, 1-做多, 2-做空, 3-平仓
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}
        
        # 获取当前状态
        current_state = self._get_state()
        current_price = current_state['index_value']
        
        # 执行交易动作
        reward = 0
        done = False
        
        # 计算交易成本
        transaction_fee = 0
        
        # 根据动作执行交易
        if action == 1 and self.position == 0:  # 开多仓
            self.position = 1
            self.position_price = current_price
            transaction_fee = self.balance * self.transaction_cost
            self.balance -= transaction_fee
            self.trades.append({
                'step': self.current_step,
                'action': 'long_open',
                'price': current_price,
                'balance': self.balance
            })
        elif action == 2 and self.position == 0:  # 开空仓
            self.position = -1
            self.position_price = current_price
            transaction_fee = self.balance * self.transaction_cost
            self.balance -= transaction_fee
            self.trades.append({
                'step': self.current_step,
                'action': 'short_open',
                'price': current_price,
                'balance': self.balance
            })
        elif action == 3 and self.position != 0:  # 平仓
            if self.position == 1:  # 平多仓
                profit = (current_price - self.position_price) * self.balance / self.position_price
                transaction_fee = (self.balance + profit) * self.transaction_cost
                self.balance += profit - transaction_fee
                self.trades.append({
                    'step': self.current_step,
                    'action': 'long_close',
                    'price': current_price,
                    'profit': profit,
                    'balance': self.balance
                })
            elif self.position == -1:  # 平空仓
                profit = (self.position_price - current_price) * self.balance / self.position_price
                transaction_fee = (self.balance + profit) * self.transaction_cost
                self.balance += profit - transaction_fee
                self.trades.append({
                    'step': self.current_step,
                    'action': 'short_close',
                    'price': current_price,
                    'profit': profit,
                    'balance': self.balance
                })
            self.position = 0
            self.position_price = 0
        
        # 移动到下一步
        self.current_step += 1
        
        # 计算奖励
        # 奖励基于资产净值的变化
        previous_net_worth = self.net_worth_history[-1] if self.net_worth_history else self.initial_balance
        current_net_worth = self.balance  # 简化处理，不考虑未实现盈亏
        reward = (current_net_worth - previous_net_worth) / previous_net_worth if previous_net_worth > 0 else 0
        self.net_worth_history.append(current_net_worth)
        
        # 检查是否结束
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # 获取下一个状态
        next_state = self._get_state()
        
        return next_state, reward, done, {}

class QLearningAgent:
    """
    Q学习代理
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4种动作: 0-持有, 1-做多, 2-做空, 3-平仓
    
    def _state_to_key(self, state):
        """
        将状态转换为Q表的键
        """
        if state is None:
            return "terminal"
        
        # 简化状态表示，使用关键特征
        position = state.get('position', 0)
        predicted_signal = state.get('predicted_signal', 0)
        # 将连续值离散化
        price_level = int(state.get('index_value', 0) // 10)  # 价格级别
        balance_level = int(state.get('balance', 0) // 1000)  # 余额级别
        
        return f"{position}_{predicted_signal}_{price_level}_{balance_level}"
    
    def act(self, state):
        """
        根据当前状态选择动作
        """
        state_key = self._state_to_key(state)
        
        # ε-贪婪策略
        if random.random() <= self.epsilon:
            return random.randrange(4)  # 随机动作
        else:
            return np.argmax(self.q_table[state_key])  # 贪婪动作
    
    def learn(self, state, action, reward, next_state):
        """
        Q学习更新
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Q学习更新公式
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error
    
    def decay_epsilon(self):
        """
        降低探索率
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_q_learning_agent(data_file, episodes=100):
    """
    训练Q学习代理
    """
    logger.info(f"Training Q-learning agent on {data_file}")
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded data with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # 创建交易环境
    env = TradingEnvironment(df)
    
    # 创建Q学习代理
    agent = QLearningAgent()
    
    # 训练循环
    scores = []
    for e in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        
        # 交易循环
        while True:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 学习
            agent.learn(state, action, reward, next_state)
            
            # 更新状态
            state = next_state
            
            # 检查是否结束
            if done:
                break
        
        # 降低探索率
        agent.decay_epsilon()
        
        # 记录结果
        scores.append(total_reward)
        if (e + 1) % 10 == 0:
            logger.info(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Final Balance: {env.balance:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    logger.info(f"Training completed. Average score: {np.mean(scores):.2f}")
    return agent

def evaluate_q_learning_agent(agent, data_file):
    """
    评估Q学习代理
    """
    logger.info(f"Evaluating Q-learning agent on {data_file}")
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded data with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # 创建交易环境
    env = TradingEnvironment(df)
    
    # 评估
    state = env.reset()
    total_reward = 0
    actions = []
    
    # 关闭探索
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    while True:
        # 使用训练好的代理选择动作
        action = agent.act(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # 恢复探索率
    agent.epsilon = original_epsilon
    
    logger.info(f"Evaluation completed. Total Reward: {total_reward:.2f}, Final Balance: {env.balance:.2f}")
    logger.info(f"Actions distribution: {pd.Series(actions).value_counts().to_dict()}")
    logger.info(f"Number of trades: {len(env.trades)}")
    
    return {
        'total_reward': total_reward,
        'final_balance': env.balance,
        'actions': actions,
        'trades': env.trades,
        'net_worth_history': env.net_worth_history
    }

def integrate_with_predictor():
    """
    将强化学习与预测器集成
    """
    logger.info("Integrating reinforcement learning with pattern predictor")
    
    import glob
    from pattern_predictor_balanced import BalancedPatternPredictor, load_test_data
    
    # 获取数据文件
    data_files = sorted(glob.glob(os.path.join("..", "predict", "*.csv")))
    logger.info(f"Found {len(data_files)} data files")
    
    if not data_files:
        logger.error("No data files found!")
        return
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 为第一个文件生成预测信号
    df = load_test_data(data_files[0])
    if df is None:
        logger.error("Failed to load test data")
        return
    
    # 进行预测
    logger.info("Generating predictions for reinforcement learning training")
    predictions = []
    for i in range(len(df)):
        predicted_signal, confidence = predictor.predict_signal(df, i)
        predictions.append(predicted_signal)
    
    # 将预测信号添加到数据中
    df['predicted_signal'] = predictions
    
    # 保存带预测信号的数据
    temp_file = os.path.join("..", "temp", "rl_training_data.csv")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    df.to_csv(temp_file, index=False)
    logger.info(f"Saved data with predictions to {temp_file}")
    
    # 训练Q学习代理
    agent = train_q_learning_agent(temp_file, episodes=50)
    
    if agent is None:
        logger.error("Failed to train agent")
        return
    
    # 评估代理（在其他文件上）
    results = []
    for i, data_file in enumerate(data_files[1:3]):  # 只评估前几个文件
        logger.info(f"\nEvaluating on file {i+1}/{min(3, len(data_files)-1)}")
        
        # 为评估文件生成预测信号
        eval_df = load_test_data(data_file)
        if eval_df is None:
            continue
            
        eval_predictions = []
        for j in range(len(eval_df)):
            predicted_signal, confidence = predictor.predict_signal(eval_df, j)
            eval_predictions.append(predicted_signal)
        
        # 将预测信号添加到数据中
        eval_df['predicted_signal'] = eval_predictions
        
        # 保存带预测信号的评估数据
        eval_temp_file = os.path.join("..", "temp", f"rl_eval_data_{i}.csv")
        os.makedirs(os.path.dirname(eval_temp_file), exist_ok=True)
        eval_df.to_csv(eval_temp_file, index=False)
        
        result = evaluate_q_learning_agent(agent, eval_temp_file)
        if result:
            results.append(result)
    
    # 输出总体结果
    if results:
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_balance = np.mean([r['final_balance'] for r in results])
        logger.info(f"\nOverall Results:")
        logger.info(f"Average Total Reward: {avg_reward:.2f}")
        logger.info(f"Average Final Balance: {avg_balance:.2f}")
    
    return agent

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Reinforcement Learning Trader')
    parser.add_argument('--mode', choices=['train', 'integrate'], default='integrate',
                       help='运行模式: train(仅训练), integrate(与预测器集成)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 获取数据文件
        import glob
        data_files = sorted(glob.glob(os.path.join("..", "predict", "*.csv")))
        if data_files:
            agent = train_q_learning_agent(data_files[0], episodes=50)
    else:
        # 与预测器集成
        agent = integrate_with_predictor()

if __name__ == "__main__":
    main()