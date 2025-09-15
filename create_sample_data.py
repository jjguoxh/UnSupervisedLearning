# -*- coding: utf-8 -*-
"""
创建示例股指期货数据
用于演示预测和可视化功能
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_trading_data(filename, base_price=3000, num_points=200):
    """
    创建示例交易数据
    """
    np.random.seed(hash(filename) % 1000)  # 基于文件名的随机种子
    
    # 生成时间序列
    x_values = list(range(num_points))
    
    # 生成价格数据（模拟股指期货走势）
    prices = [base_price]
    
    # 模拟价格波动
    trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # 趋势方向
    volatility = np.random.uniform(0.5, 2.0)  # 波动率
    
    for i in range(1, num_points):
        # 基础趋势
        trend_change = trend * np.random.uniform(0.1, 0.5)
        
        # 随机波动
        random_change = np.random.normal(0, volatility)
        
        # 均值回归
        mean_reversion = (base_price - prices[-1]) * 0.001
        
        # 计算价格变化
        price_change = trend_change + random_change + mean_reversion
        
        # 限制单次变化幅度
        price_change = np.clip(price_change, -20, 20)
        
        new_price = prices[-1] + price_change
        
        # 确保价格在合理范围内
        new_price = max(base_price * 0.8, min(base_price * 1.2, new_price))
        
        prices.append(new_price)
        
        # 偶尔改变趋势
        if i % 50 == 0:
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
    
    # 生成标签（模拟交易信号）
    labels = []
    position = 0  # 0: 空仓, 1: 多头, -1: 空头
    last_trade_i = 0  # 记录上次交易时间
    
    for i in range(num_points):
        if i < 30:
            labels.append(0)  # 前30个点不交易
            continue
        
        # 计算技术指标
        recent_prices = prices[max(0, i-20):i+1]
        price_change = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
        
        # 更积极的交易逻辑（生成更多交易信号）
        if position == 0:  # 空仓状态
            if price_change > 0.002 and np.random.random() > 0.4:  # 上涨趋势，开多
                labels.append(1)  # 做多开仓
                position = 1
            elif price_change < -0.002 and np.random.random() > 0.4:  # 下跌趋势，开空
                labels.append(3)  # 做空开仓
                position = -1
            else:
                labels.append(0)  # 观察
        
        elif position == 1:  # 多头持仓
            if (price_change < -0.001 and np.random.random() > 0.5) or (i - last_trade_i > 20 and np.random.random() > 0.6):  # 止损或时间止损
                labels.append(2)  # 做多平仓
                position = 0
                last_trade_i = i
            else:
                labels.append(0)  # 持仓
        
        elif position == -1:  # 空头持仓
            if (price_change > 0.001 and np.random.random() > 0.5) or (i - last_trade_i > 20 and np.random.random() > 0.6):  # 止损或时间止损
                labels.append(4)  # 做空平仓
                position = 0
                last_trade_i = i
            else:
                labels.append(0)  # 持仓
    
    # 创建DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'index_value': prices,
        'label': labels
    })
    
    return df

def create_multiple_sample_files():
    """
    创建多个示例文件
    """
    # 确保result目录存在
    os.makedirs('./result', exist_ok=True)
    
    # 创建不同类型的市场数据
    sample_configs = [
        {'filename': 'trading_day_001.csv', 'base_price': 3000, 'num_points': 240},
        {'filename': 'trading_day_002.csv', 'base_price': 3050, 'num_points': 220},
        {'filename': 'trading_day_003.csv', 'base_price': 2980, 'num_points': 260},
        {'filename': 'trading_day_004.csv', 'base_price': 3120, 'num_points': 200},
        {'filename': 'trading_day_005.csv', 'base_price': 2950, 'num_points': 280},
    ]
    
    print("=== 创建示例股指期货数据 ===")
    
    for config in sample_configs:
        filename = config['filename']
        base_price = config['base_price']
        num_points = config['num_points']
        
        print(f"📊 创建文件: {filename}")
        
        # 生成数据
        df = create_sample_trading_data(filename, base_price, num_points)
        
        # 保存文件
        filepath = f"./result/{filename}"
        df.to_csv(filepath, index=False)
        
        # 统计信息
        signal_counts = df['label'].value_counts().sort_index()
        trading_signals = signal_counts[signal_counts.index != 0].sum()
        
        print(f"   📈 价格范围: {df['index_value'].min():.2f} - {df['index_value'].max():.2f}")
        print(f"   🎯 交易信号: {trading_signals} 个")
        print(f"   📏 数据点数: {len(df)}")
        
        # 显示信号分布
        signal_names = {0: '观察', 1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        for label, count in signal_counts.items():
            if count > 0:
                print(f"      {signal_names.get(label, f'标签{label}')}: {count}")
        
        print()
    
    print(f"✅ 已创建 {len(sample_configs)} 个示例文件到 ./result/ 目录")
    print("📁 文件列表:")
    for config in sample_configs:
        print(f"   - {config['filename']}")
    
    print("\n🚀 现在可以运行 predict_and_visualize.py 进行预测和可视化！")

def main():
    create_multiple_sample_files()

if __name__ == "__main__":
    main()