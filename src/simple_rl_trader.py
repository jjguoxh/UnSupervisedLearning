# -*- coding: utf-8 -*-
"""
简单的强化学习交易优化器
基于预测信号优化交易决策
"""

import numpy as np
import pandas as pd
import random
import os
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRLTrader:
    """
    简单的强化学习交易器
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q表：状态(预测信号+其他特征) -> 动作(交易决策) -> 价值
        # 使用更丰富的状态表示
        self.q_table = defaultdict(lambda: np.zeros(2))
        
        # 统计信息
        self.total_trades = 0
        self.successful_trades = 0
    
    def get_state(self, predicted_signal, confidence=0, position=0, balance=10000):
        """
        获取状态 - 增强版本
        """
        # 将连续值离散化
        confidence_level = int(confidence * 10)  # 置信度级别 (0-10)
        position_state = position  # 持仓状态 (-1, 0, 1)
        balance_level = int(balance // 1000)  # 余额级别
        
        # 组合状态
        state = f"{int(predicted_signal)}_{confidence_level}_{position_state}_{balance_level}"
        return state
    
    def choose_action(self, state):
        """
        选择动作
        """
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # 随机动作
        else:
            return np.argmax(self.q_table[state])  # 最优动作
    
    def update_q_value(self, state, action, reward, next_state):
        """
        更新Q值
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def calculate_reward(self, action, predicted_signal, actual_signal, price_change, position=0, balance=10000):
        """
        计算奖励 - 改进版本
        """
        # 基础奖励
        base_reward = 0
        
        if action == 0:  # 忽略信号
            # 如果信号是正确的但被忽略了，给予负奖励
            if predicted_signal != 0 and predicted_signal == actual_signal:
                base_reward = -0.1
            else:
                base_reward = 0  # 正确忽略无用信号
        else:  # 执行信号
            if predicted_signal == 0:
                base_reward = -0.2  # 惩罚执行无信号的决策
            elif predicted_signal == actual_signal:
                # 根据价格变化计算收益
                if predicted_signal in [1, 3]:  # 开仓信号
                    base_reward = 0.1  # 开仓奖励
                elif predicted_signal in [2, 4]:  # 平仓信号
                    # 考虑持仓情况和实际收益
                    if (predicted_signal == 2 and position == 1) or (predicted_signal == 4 and position == -1):
                        profit_reward = abs(price_change) * 0.5
                        base_reward = 0.2 + profit_reward  # 平仓奖励 + 收益奖励
                    else:
                        base_reward = 0.1  # 正确信号但持仓不匹配
            else:
                base_reward = -0.3  # 惩罚错误执行信号
        
        # 考虑资金管理
        if balance < 1000:  # 资金过少时更谨慎
            base_reward *= 0.5
            
        return base_reward
    
    def train(self, data, episodes=100):
        """
        训练强化学习模型
        """
        logger.info("开始训练强化学习模型")
        
        # 初始化持仓和余额
        position = 0  # 0: 无持仓, 1: 多头, -1: 空头
        balance = 10000  # 初始资金
        
        for episode in range(episodes):
            total_reward = 0
            correct_decisions = 0
            total_decisions = 0
            
            # 重置持仓和余额
            position = 0
            balance = 10000
            
            # 遍历数据
            for i in range(1, len(data)):
                # 获取当前状态
                predicted_signal = data.iloc[i]['predicted_signal']
                actual_signal = data.iloc[i]['label']
                price_change = data.iloc[i]['index_value'] - data.iloc[i-1]['index_value']
                
                state = self.get_state(predicted_signal, position=position, balance=balance)
                
                # 选择动作
                action = self.choose_action(state)
                
                # 计算奖励
                reward = self.calculate_reward(action, predicted_signal, actual_signal, price_change, position, balance)
                total_reward += reward
                
                # 更新持仓和余额
                if action == 1:  # 执行信号
                    if predicted_signal == 1:  # 做多开仓
                        position = 1
                    elif predicted_signal == 2:  # 做多平仓
                        if position == 1:  # 只有在多头持仓时才能平仓
                            balance += price_change * 100  # 假设每手100单位
                            position = 0
                    elif predicted_signal == 3:  # 做空开仓
                        position = -1
                    elif predicted_signal == 4:  # 做空平仓
                        if position == -1:  # 只有在空头持仓时才能平仓
                            balance += (-price_change) * 100  # 做空时价格下跌盈利
                            position = 0
                
                # 统计正确决策
                if (action == 1 and predicted_signal != 0 and predicted_signal == actual_signal) or \
                   (action == 0 and (predicted_signal == 0 or predicted_signal != actual_signal)):
                    correct_decisions += 1
                total_decisions += 1
                
                # 获取下一个状态
                if i < len(data) - 1:
                    next_predicted_signal = data.iloc[i+1]['predicted_signal']
                    next_state = self.get_state(next_predicted_signal, position=position, balance=balance)
                    
                    # 更新Q值
                    self.update_q_value(state, action, reward, next_state)
            
            # 记录训练进度
            accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Accuracy: {accuracy:.2%}")
        
        logger.info("训练完成")
    
    def predict(self, predicted_signal):
        """
        使用训练好的模型进行预测
        """
        state = self.get_state(predicted_signal)
        action = np.argmax(self.q_table[state])
        return action  # 0-忽略信号, 1-执行信号
    
    def get_q_table_summary(self):
        """
        获取Q表摘要
        """
        summary = {}
        for state in sorted(self.q_table.keys()):
            summary[state] = {
                'ignore': self.q_table[state][0],
                'execute': self.q_table[state][1],
                'recommended_action': 'execute' if np.argmax(self.q_table[state]) == 1 else 'ignore'
            }
        return summary

def load_data_with_predictions(data_file):
    """
    加载数据并生成预测信号
    """
    from pattern_predictor_balanced import BalancedPatternPredictor, load_test_data
    
    # 加载数据
    df = load_test_data(data_file)
    if df is None:
        return None
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 生成预测信号
    logger.info(f"为 {data_file} 生成预测信号")
    predictions = []
    confidences = []
    
    for i in range(len(df)):
        predicted_signal, confidence = predictor.predict_signal(df, i)
        predictions.append(predicted_signal)
        confidences.append(confidence)
    
    df['predicted_signal'] = predictions
    df['prediction_confidence'] = confidences
    
    return df

def evaluate_performance(data, trader):
    """
    评估性能
    """
    logger.info("评估强化学习交易器性能")
    
    correct_executions = 0
    total_executions = 0
    correct_ignores = 0
    total_ignores = 0
    
    actions = []
    
    for i in range(len(data)):
        predicted_signal = data.iloc[i]['predicted_signal']
        actual_signal = data.iloc[i]['label']
        
        # 使用训练好的模型选择动作
        action = trader.predict(predicted_signal)
        actions.append(action)
        
        if action == 1:  # 执行信号
            total_executions += 1
            if predicted_signal != 0 and predicted_signal == actual_signal:
                correct_executions += 1
        else:  # 忽略信号
            total_ignores += 1
            if predicted_signal == 0 or predicted_signal != actual_signal:
                correct_ignores += 1
    
    execution_accuracy = correct_executions / total_executions if total_executions > 0 else 0
    ignore_accuracy = correct_ignores / total_ignores if total_ignores > 0 else 0
    overall_accuracy = (correct_executions + correct_ignores) / len(data) if len(data) > 0 else 0
    
    logger.info(f"执行决策准确率: {execution_accuracy:.2%} ({correct_executions}/{total_executions})")
    logger.info(f"忽略决策准确率: {ignore_accuracy:.2%} ({correct_ignores}/{total_ignores})")
    logger.info(f"总体准确率: {overall_accuracy:.2%}")
    logger.info(f"动作分布: {pd.Series(actions).value_counts().to_dict()}")
    
    return {
        'execution_accuracy': execution_accuracy,
        'ignore_accuracy': ignore_accuracy,
        'overall_accuracy': overall_accuracy,
        'actions': actions
    }

def main():
    """
    主函数
    """
    import glob
    
    # 获取数据文件（从正确的目录）
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
    
    # 创建并训练强化学习交易器
    trader = SimpleRLTrader(learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    trader.train(train_data, episodes=50)
    
    # 显示Q表摘要
    logger.info("Q表摘要:")
    q_summary = trader.get_q_table_summary()
    for state, values in q_summary.items():
        logger.info(f"  信号 {state}: 忽略={values['ignore']:.3f}, 执行={values['execute']:.3f}, 推荐={values['recommended_action']}")
    
    # 评估性能
    logger.info("=== 性能评估 ===")
    performance = evaluate_performance(train_data, trader)
    
    # 在其他文件上进行测试
    logger.info("=== 测试阶段 ===")
    test_performances = []
    for i, data_file in enumerate(data_files[1:3]):  # 测试前几个文件
        logger.info(f"测试文件 {i+1}/{min(2, len(data_files)-1)}: {os.path.basename(data_file)}")
        test_data = load_data_with_predictions(data_file)
        if test_data is not None:
            test_performance = evaluate_performance(test_data, trader)
            test_performances.append(test_performance)
    
    # 输出测试结果摘要
    if test_performances:
        avg_execution_accuracy = np.mean([p['execution_accuracy'] for p in test_performances])
        avg_ignore_accuracy = np.mean([p['ignore_accuracy'] for p in test_performances])
        avg_overall_accuracy = np.mean([p['overall_accuracy'] for p in test_performances])
        
        logger.info("=== 测试结果摘要 ===")
        logger.info(f"平均执行决策准确率: {avg_execution_accuracy:.2%}")
        logger.info(f"平均忽略决策准确率: {avg_ignore_accuracy:.2%}")
        logger.info(f"平均总体准确率: {avg_overall_accuracy:.2%}")

if __name__ == "__main__":
    main()