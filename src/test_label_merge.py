# -*- coding: utf-8 -*-
"""
测试标签合并功能
"""

import numpy as np
import pandas as pd

def test_label_merge():
    """
    测试标签合并功能
    """
    # 创建测试数据
    test_labels = np.array([0, 1, 5, 2, 0, 3, 6, 4, 0, 1, 5, 5, 2, 0, 3, 6, 6, 4])
    print("原始标签:", test_labels)
    
    # 合并标签：1和5合并为1，3和6合并为3
    merged_labels = test_labels.copy()
    merged_labels[merged_labels == 5] = 1  # 将标签5合并到标签1
    merged_labels[merged_labels == 6] = 3  # 将标签6合并到标签3
    
    print("合并后标签:", merged_labels)
    
    # 验证合并结果
    expected_labels = np.array([0, 1, 1, 2, 0, 3, 3, 4, 0, 1, 1, 1, 2, 0, 3, 3, 3, 4])
    print("期望标签:", expected_labels)
    
    # 检查是否匹配
    if np.array_equal(merged_labels, expected_labels):
        print("✓ 标签合并测试通过")
    else:
        print("✗ 标签合并测试失败")
    
    # 统计各类标签数量
    unique, counts = np.unique(merged_labels, return_counts=True)
    print("标签统计:")
    for label, count in zip(unique, counts):
        label_names = {
            0: "无操作",
            1: "做多开仓",  # 包括原来的标签1和5
            2: "做多平仓",
            3: "做空开仓",  # 包括原来的标签3和6
            4: "做空平仓"
        }
        print(f"  {label_names.get(label, '未知')}({label}): {count} 个")

if __name__ == "__main__":
    test_label_merge()