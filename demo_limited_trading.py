#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示限制开仓次数功能
"""

import pandas as pd
import numpy as np
from generate_trading_analysis import TradingAnalysisVisualizer

def demo_limited_trading():
    """演示限制开仓次数和选择高置信度信号的功能"""
    print("=== 演示限制开仓次数功能 ===")
    
    # 创建测试数据 - 模拟一天的交易数据
    np.random.seed(42)
    prices = [100]
    for i in range(99):
        change = np.random.normal(0, 0.5)
        prices.append(prices[-1] + change)
    
    test_data = {'index_value': prices}
    df = pd.DataFrame(test_data)
    
    # 创建多个开仓信号，模拟系统产生很多交易机会
    test_predictions = [0] * 100
    test_confidences = [0.5] * 100
    
    # 添加6个做多开仓信号，置信度不同
    open_positions = [10, 20, 30, 40, 50, 60]
    close_positions = [15, 25, 35, 45, 55, 65]
    confidences_values = [0.95, 0.75, 0.85, 0.65, 0.90, 0.70]  # 不同置信度
    
    for i, (open_pos, close_pos, conf) in enumerate(zip(open_positions, close_positions, confidences_values)):
        test_predictions[open_pos] = 1  # 做多开仓
        test_predictions[close_pos] = 2  # 做多平仓
        test_confidences[open_pos] = conf
        test_confidences[close_pos] = 0.8
    
    print(f"原始信号中有 {sum(1 for p in test_predictions if p == 1)} 个做多开仓信号")
    print(f"开仓信号置信度: {[confidences_values[i] for i in range(len(confidences_values))]}")
    
    # 创建分析器
    visualizer = TradingAnalysisVisualizer()
    
    print("\n=== 不限制开仓次数的结果 ===")
    # 测试不限制开仓次数（设置为很大的数）
    unlimited_predictions, unlimited_trades = visualizer.filter_trading_signals(
        test_predictions, df['index_value'].values, test_confidences, max_daily_trades=10
    )
    
    unlimited_opens = sum(1 for p in unlimited_predictions if p == 1)
    print(f"实际开仓次数: {unlimited_opens}")
    print(f"完成交易次数: {len(unlimited_trades)}")
    if unlimited_trades:
        total_profit = sum(trade['profit'] for trade in unlimited_trades)
        avg_confidence = sum(trade['confidence'] for trade in unlimited_trades) / len(unlimited_trades)
        print(f"总盈亏: {total_profit:.2f}")
        print(f"平均置信度: {avg_confidence:.3f}")
    
    print("\n=== 限制每日最多3次开仓的结果 ===")
    # 测试限制每日最多3次开仓
    limited_predictions, limited_trades = visualizer.filter_trading_signals(
        test_predictions, df['index_value'].values, test_confidences, max_daily_trades=3
    )
    
    limited_opens = sum(1 for p in limited_predictions if p == 1)
    print(f"实际开仓次数: {limited_opens}")
    print(f"完成交易次数: {len(limited_trades)}")
    if limited_trades:
        total_profit = sum(trade['profit'] for trade in limited_trades)
        avg_confidence = sum(trade['confidence'] for trade in limited_trades) / len(limited_trades)
        print(f"总盈亏: {total_profit:.2f}")
        print(f"平均置信度: {avg_confidence:.3f}")
        
        print("\n选中的交易详情:")
        for i, trade in enumerate(limited_trades, 1):
            print(f"  交易{i}: 开仓价格{trade['open_price']:.2f} -> 平仓价格{trade['close_price']:.2f} | 盈亏:{trade['profit']:+.2f} | 置信度:{trade['confidence']:.3f}")
    
    print("\n=== 对比分析 ===")
    print(f"通过限制开仓次数，从 {unlimited_opens} 次开仓减少到 {limited_opens} 次")
    if unlimited_trades and limited_trades:
        unlimited_avg_conf = sum(trade['confidence'] for trade in unlimited_trades) / len(unlimited_trades)
        limited_avg_conf = sum(trade['confidence'] for trade in limited_trades) / len(limited_trades)
        print(f"平均置信度从 {unlimited_avg_conf:.3f} 提升到 {limited_avg_conf:.3f}")
        
        unlimited_profit = sum(trade['profit'] for trade in unlimited_trades)
        limited_profit = sum(trade['profit'] for trade in limited_trades)
        print(f"总盈亏对比: 不限制 {unlimited_profit:.2f} vs 限制后 {limited_profit:.2f}")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    demo_limited_trading()