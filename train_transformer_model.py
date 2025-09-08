# -*- coding: utf-8 -*-
"""
训练Transformer模型的脚本
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_predictor import train_transformer_model

if __name__ == "__main__":
    print("开始训练Transformer模型...")
    train_transformer_model()
    print("Transformer模型训练完成!")