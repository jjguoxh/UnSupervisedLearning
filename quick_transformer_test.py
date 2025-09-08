# -*- coding: utf-8 -*-
"""
快速测试Transformer模型的脚本
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_predictor import TradingSignalTransformer

def create_test_data():
    """创建测试数据"""
    # 创建一个简单的测试数据集
    data = {
        'x': np.random.randn(100),
        'a': np.random.randn(100),
        'b': np.random.randn(100),
        'c': np.random.randn(100),
        'd': np.random.randn(100),
        'index_value': np.cumsum(np.random.randn(100)) + 1000  # 累积和模拟指数值
    }
    df = pd.DataFrame(data)
    return df

def main():
    print("快速测试Transformer模型...")
    
    # 创建测试数据
    print("1. 创建测试数据...")
    df = create_test_data()
    print(f"   数据形状: {df.shape}")
    
    # 创建预测器
    print("2. 创建Transformer预测器...")
    predictor = TradingSignalTransformer(max_seq_len=2000)
    
    # 加载模型（如果存在）
    model_path = os.path.join("model", "balanced_model", "transformer_predictor.pth")
    if os.path.exists(model_path):
        print("3. 加载预训练模型...")
        if predictor.load_model(model_path):
            print("   模型加载成功")
        else:
            print("   模型加载失败")
    else:
        print("3. 未找到预训练模型，跳过加载")
    
    # 进行预测
    print("4. 进行预测...")
    try:
        predictions, confidences = predictor.predict(df)
        print(f"   预测完成，结果长度: {len(predictions)}")
        print(f"   平均置信度: {np.mean(confidences):.4f}")
        
        # 显示前10个预测结果
        print("   前10个预测结果:")
        for i in range(min(10, len(predictions))):
            print(f"     点 {i}: 信号 {predictions[i]}, 置信度 {confidences[i]:.4f}")
            
        print("✓ 测试完成")
        return 0
    except Exception as e:
        print(f"   预测过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())