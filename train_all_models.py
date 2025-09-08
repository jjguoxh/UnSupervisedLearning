# -*- coding: utf-8 -*-
"""
训练所有模型的脚本
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from improved_predictor import test_improved_predictor
from transformer_predictor import train_transformer_model

if __name__ == "__main__":
    print("开始训练所有模型...")
    
    # 训练改进的预测模型
    print("\n1. 训练改进的预测模型...")
    test_improved_predictor()
    
    # 训练Transformer模型
    print("\n2. 训练Transformer模型...")
    train_transformer_model()
    
    print("\n所有模型训练完成!")