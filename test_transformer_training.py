# -*- coding: utf-8 -*-
"""
测试Transformer模型训练的简化脚本
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_predictor import train_transformer_model, test_transformer_model

def main():
    print("开始测试Transformer模型训练...")
    
    try:
        # 训练模型
        print("\n1. 训练Transformer模型...")
        train_transformer_model()
        print("✓ Transformer模型训练完成")
        
        # 测试模型
        print("\n2. 测试Transformer模型...")
        test_transformer_model()
        print("✓ Transformer模型测试完成")
        
        print("\n所有步骤已完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())