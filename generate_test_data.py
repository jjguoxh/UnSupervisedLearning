# -*- coding: utf-8 -*-
"""
生成测试数据用于验证模式识别流程
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_trading_data(n_samples=1000, filename="test_data.csv"):
    """
    生成合成的交易数据
    """
    np.random.seed(42)
    
    # 生成时间序列
    start_time = datetime.now() - timedelta(seconds=n_samples)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]
    x_values = [(t - start_time).total_seconds() for t in timestamps]
    
    # 生成基础价格走势（随机游走）
    price_changes = np.random.normal(0, 1, n_samples)
    prices = np.cumsum(price_changes) + 100  # 基础价格从100开始
    
    # 生成影响因子 a, b, c, d
    a = np.random.normal(0, 0.5, n_samples)
    b = np.random.normal(0, 0.3, n_samples)
    c = np.random.normal(0, 0.4, n_samples)
    d = np.random.normal(0, 0.2, n_samples)
    
    # 生成交易标签（模拟真实的交易信号）
    labels = np.zeros(n_samples, dtype=int)
    
    # 模拟交易策略：基于价格趋势和技术指标
    for i in range(20, n_samples-20):
        # 计算短期和长期移动平均
        short_ma = np.mean(prices[i-5:i])
        long_ma = np.mean(prices[i-20:i])
        
        # 计算价格动量
        momentum = prices[i] - prices[i-10]
        
        # 生成交易信号的概率
        signal_prob = np.random.random()
        
        # 做多信号条件
        if (short_ma > long_ma and momentum > 0 and a[i] > 0.2 and signal_prob < 0.05):
            labels[i] = 1  # 做多开仓
            # 寻找平仓点
            for j in range(i+1, min(i+15, n_samples)):
                if prices[j] > prices[i] * 1.02 or j == i+10:  # 盈利2%或持有10期
                    labels[j] = 2  # 做多平仓
                    break
        
        # 做空信号条件
        elif (short_ma < long_ma and momentum < 0 and b[i] < -0.2 and signal_prob < 0.03):
            labels[i] = 3  # 做空开仓
            # 寻找平仓点
            for j in range(i+1, min(i+15, n_samples)):
                if prices[j] < prices[i] * 0.98 or j == i+10:  # 盈利2%或持有10期
                    labels[j] = 4  # 做空平仓
                    break
    
    # 创建DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'index_value': prices,
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'label': labels
    })
    
    # 保存到label目录
    os.makedirs('./label', exist_ok=True)
    filepath = os.path.join('./label', filename)
    df.to_csv(filepath, index=False)
    
    print(f"Generated {n_samples} samples with {np.sum(labels != 0)} trading signals")
    print(f"Signal distribution:")
    for label in [1, 2, 3, 4]:
        count = np.sum(labels == label)
        print(f"  Label {label}: {count} ({count/n_samples*100:.2f}%)")
    
    print(f"Data saved to {filepath}")
    return df

def generate_multiple_files(n_files=5, samples_per_file=1000):
    """
    生成多个测试数据文件
    """
    for i in range(n_files):
        filename = f"test_data_{i+1:02d}.csv"
        generate_synthetic_trading_data(samples_per_file, filename)
    
    print(f"\nGenerated {n_files} test data files in ./label/ directory")

if __name__ == "__main__":
    # 生成测试数据
    generate_multiple_files(n_files=3, samples_per_file=800)
    
    # 验证数据质量
    print("\n=== Data Quality Check ===")
    label_files = [f for f in os.listdir('./label') if f.endswith('.csv')]
    
    for file in label_files:
        filepath = os.path.join('./label', file)
        df = pd.read_csv(filepath)
        
        print(f"\n{file}:")
        print(f"  Samples: {len(df)}")
        print(f"  Trading signals: {np.sum(df['label'] != 0)}")
        print(f"  Signal rate: {np.sum(df['label'] != 0)/len(df)*100:.2f}%")
        
        # 检查数据完整性
        if df.isnull().any().any():
            print(f"  ⚠️ Warning: Contains null values")
        else:
            print(f"  ✓ No null values")
        
        # 检查标签分布
        label_dist = df['label'].value_counts().sort_index()
        print(f"  Label distribution: {dict(label_dist)}")