import sys
import os
import pandas as pd
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

def test_predictions():
    """测试预测功能"""
    # 创建预测器
    predictor = BalancedPatternPredictor()
    print(f"成功加载 {len(predictor.cluster_models)} 个聚类模型")
    
    # 获取测试文件
    label_dir = os.path.join(os.path.dirname(__file__), 'label')
    if not os.path.exists(label_dir):
        print("未找到label目录")
        return
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
    if not label_files:
        print("label目录中没有CSV文件")
        return
    
    # 测试多个文件
    test_files = label_files[:5]  # 只测试前5个文件
    all_signal_distributions = {}
    
    for i, file_name in enumerate(test_files):
        print(f"\n=== 测试文件 {i+1}/{len(test_files)}: {file_name} ===")
        
        # 使用文件进行测试
        test_file = os.path.join(label_dir, file_name)
        print(f"使用文件进行测试: {test_file}")
        
        # 加载数据
        df = load_realtime_data(test_file)
        if df is None:
            print("加载数据失败")
            continue
        
        print(f"数据形状: {df.shape}")
        
        # 进行单点预测
        predicted_signal, confidence = predictor.predict_realtime_signal(df)
        print(f"单点预测结果: 信号={predicted_signal}, 置信度={confidence:.4f}")
        
        # 进行序列预测
        sequence_predictions = predictor.predict_realtime_sequence(df, sequence_length=min(50, len(df)))
        print(f"序列预测完成，共 {len(sequence_predictions)} 个预测点")
        
        # 统计信号分布
        signal_counts = {}
        for pred in sequence_predictions:
            signal = pred['predicted_signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        print("信号分布:")
        signal_names = {0: "无操作", 1: "做多开仓", 2: "做多平仓", 3: "做空开仓", 4: "做空平仓"}
        for signal, count in sorted(signal_counts.items()):
            print(f"  {signal_names.get(signal, f'未知信号{signal}')}: {count}")
        
        # 记录所有文件的信号分布
        for signal, count in signal_counts.items():
            if signal not in all_signal_distributions:
                all_signal_distributions[signal] = 0
            all_signal_distributions[signal] += count
    
    # 打印所有文件的总体信号分布
    print(f"\n=== {len(test_files)}个文件的总体信号分布 ===")
    for signal, count in sorted(all_signal_distributions.items()):
        print(f"  {signal_names.get(signal, f'未知信号{signal}')}: {count}")

if __name__ == "__main__":
    test_predictions()