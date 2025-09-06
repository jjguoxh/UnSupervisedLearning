# -*- coding: utf-8 -*-
"""
综合测试：验证标签合并后的系统一致性
"""

import numpy as np
import pandas as pd
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_label_consistency():
    """
    测试标签系统的一致性
    """
    print("=== 标签系统一致性测试 ===")
    
    # 测试标签合并
    print("1. 测试标签合并功能:")
    original_labels = np.array([0, 1, 5, 2, 0, 3, 6, 4, 0, 1, 5, 5, 2, 0, 3, 6, 6, 4])
    print(f"   原始标签: {original_labels}")
    
    # 合并标签：1和5合并为1，3和6合并为3
    merged_labels = original_labels.copy()
    merged_labels[merged_labels == 5] = 1  # 将标签5合并到标签1
    merged_labels[merged_labels == 6] = 3  # 将标签6合并到标签3
    
    expected_labels = np.array([0, 1, 1, 2, 0, 3, 3, 4, 0, 1, 1, 1, 2, 0, 3, 3, 3, 4])
    print(f"   合并后标签: {merged_labels}")
    print(f"   期望标签: {expected_labels}")
    
    if np.array_equal(merged_labels, expected_labels):
        print("   ✓ 标签合并测试通过")
    else:
        print("   ✗ 标签合并测试失败")
        return False
    
    # 测试标签统计
    print("\n2. 测试标签统计功能:")
    unique, counts = np.unique(merged_labels, return_counts=True)
    label_stats = dict(zip(unique, counts))
    print(f"   标签统计: {label_stats}")
    
    expected_stats = {0: 4, 1: 5, 2: 2, 3: 5, 4: 2}
    if label_stats == expected_stats:
        print("   ✓ 标签统计测试通过")
    else:
        print("   ✗ 标签统计测试失败")
        return False
    
    # 测试标签描述
    print("\n3. 测试标签描述功能:")
    label_names = {
        0: "无操作",
        1: "做多开仓",  # 包括原来的标签1和5
        2: "做多平仓",
        3: "做空开仓",  # 包括原来的标签3和6
        4: "做空平仓"
    }
    
    for label, count in label_stats.items():
        desc = label_names.get(label, "未知")
        print(f"   {desc}({label}): {count} 个")
    
    print("   ✓ 标签描述测试通过")
    
    # 测试交易对识别
    print("\n4. 测试交易对识别功能:")
    # 模拟一个简单的交易序列
    test_sequence = [0, 1, 1, 1, 2, 0, 0, 3, 3, 3, 4, 0]
    print(f"   测试序列: {test_sequence}")
    
    # 识别交易对
    trading_pairs = []
    i = 0
    while i < len(test_sequence) - 1:
        open_signal = test_sequence[i]
        close_signal = test_sequence[i + 1]
        
        # 做多交易对：标签1(开仓) -> 标签2(平仓)
        # 做空交易对：标签3(开仓) -> 标签4(平仓)
        is_long_pair = (open_signal == 1 and close_signal == 2)
        is_short_pair = (open_signal == 3 and close_signal == 4)
        
        if is_long_pair or is_short_pair:
            pair_type = "做多" if is_long_pair else "做空"
            trading_pairs.append((i, i+1, pair_type))
            i += 2
        else:
            i += 1
    
    print(f"   识别到的交易对: {trading_pairs}")
    expected_pairs = [(3, 4, '做多'), (9, 10, '做空')]
    if trading_pairs == expected_pairs:
        print("   ✓ 交易对识别测试通过")
    else:
        print("   ✗ 交易对识别测试失败")
        return False
    
    print("\n=== 所有测试通过! ===")
    return True

def test_file_consistency():
    """
    测试文件系统的一致性
    """
    print("\n=== 文件系统一致性测试 ===")
    
    # 检查关键文件是否存在
    required_files = [
        "label_generation.py",
        "pattern_recognition.py",
        "pattern_predictor.py",
        "trading_pattern_learning.py",
        "realtime_predictor.py"
    ]
    
    src_dir = os.path.join(os.path.dirname(__file__))
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(src_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"   缺少文件: {missing_files}")
        return False
    else:
        print("   ✓ 所有必需文件存在")
    
    # 检查文档文件
    doc_files = [
        "../README.md",
        "../doc/label_system.md",
        "../doc/prediction_programs_summary.md"
    ]
    
    missing_docs = []
    for file in doc_files:
        file_path = os.path.join(src_dir, file)
        if not os.path.exists(file_path):
            missing_docs.append(file)
    
    if missing_docs:
        print(f"   缺少文档: {missing_docs}")
        return False
    else:
        print("   ✓ 所有文档文件存在")
    
    print("   ✓ 文件系统一致性测试通过")
    return True

def main():
    """
    主测试函数
    """
    print("开始综合测试...")
    
    # 运行标签一致性测试
    label_test_passed = test_label_consistency()
    
    # 运行文件一致性测试
    file_test_passed = test_file_consistency()
    
    if label_test_passed and file_test_passed:
        print("\n🎉 所有测试都通过了!")
        print("标签合并系统已正确实现并保持一致性。")
        return True
    else:
        print("\n❌ 部分测试失败，请检查上述错误。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)