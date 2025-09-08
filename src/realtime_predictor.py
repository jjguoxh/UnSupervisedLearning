# -*- coding: utf-8 -*-
"""
实时预测程序
基于已训练的模式模型进行实时交易信号预测
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import logging
import warnings
import sys
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pattern_predictor_balanced import BalancedPatternPredictor, load_realtime_data

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..", "realtime_data/")  # 实时数据目录
PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "..", "predictions/")  # 预测结果目录
VISUALIZATION_DIR = os.path.join(CURRENT_DIR, "..", "visualization/")  # 可视化结果目录

def monitor_directory_and_predict():
    """
    监控目录并进行实时预测
    """
    logger.info("Starting directory monitoring for real-time prediction...")
    logger.info(f"Monitoring directory: {DATA_DIR}")
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    if predictor.load_rl_model(rl_model_path):
        logger.info("Successfully loaded reinforcement learning model")
        use_rl = True
    else:
        logger.warning("Failed to load reinforcement learning model, using base predictions only")
        use_rl = False
    
    # 记录已处理的文件
    processed_files = set()
    
    try:
        while True:
            # 获取目录中的所有CSV文件
            data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
            
            # 处理新文件
            for data_file in data_files:
                if data_file in processed_files:
                    continue
                
                logger.info(f"Processing new data file: {data_file}")
                
                # 加载数据
                df = load_realtime_data(data_file)
                if df is None:
                    logger.error(f"Failed to load data from {data_file}")
                    continue
                
                # 进行实时预测（最后一个点）
                if use_rl:
                    predicted_signal, confidence = predictor.predict_signal_with_rl(df, len(df) - 1)
                else:
                    predicted_signal, confidence = predictor.predict_realtime_signal(df)
                
                # 进行序列预测（最后100个点）
                sequence_predictions = []
                start_idx = max(10, len(df) - 100)
                for i in range(start_idx, len(df)):
                    if use_rl:
                        pred_signal, pred_conf = predictor.predict_signal_with_rl(df, i)
                    else:
                        pred_signal, pred_conf = predictor.predict_realtime_signal(df)
                    sequence_predictions.append({
                        'index': i,
                        'predicted_signal': pred_signal,
                        'confidence': pred_conf
                    })
                
                # 保存预测结果
                file_name = os.path.splitext(os.path.basename(data_file))[0]
                single_prediction_path = os.path.join(PREDICTIONS_DIR, f"realtime_prediction_{file_name}.json")
                
                single_result = {
                    'file': data_file,
                    'predicted_signal': int(predicted_signal),
                    'confidence': float(confidence),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                try:
                    with open(single_prediction_path, 'w', encoding='utf-8') as f:
                        json.dump(single_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Single point prediction saved to {single_prediction_path}")
                except Exception as e:
                    logger.error(f"Error saving single point prediction: {e}")
                
                # 保存序列预测结果
                sequence_prediction_path = os.path.join(PREDICTIONS_DIR, f"realtime_sequence_prediction_{file_name}.json")
                sequence_results = {
                    'file': data_file,
                    'predictions': sequence_predictions,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                try:
                    with open(sequence_prediction_path, 'w', encoding='utf-8') as f:
                        json.dump(sequence_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"Sequence prediction saved to {sequence_prediction_path}")
                except Exception as e:
                    logger.error(f"Error saving sequence prediction: {e}")
                
                # 生成可视化结果（使用实时预测适配的可视化函数）
                output_path = os.path.join(VISUALIZATION_DIR, f"realtime_prediction_{file_name}.png")
                visualize_realtime_predictions(df, sequence_predictions, output_path)
                
                logger.info(f"Visualization saved to {output_path}")
                
                # 标记为已处理
                processed_files.add(data_file)
            
            # 等待一段时间再检查
            import time
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        logger.info("Directory monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error in directory monitoring: {e}")

def monitor_directory_and_predict_with_rl(use_rl):
    """
    监控目录并进行实时预测（支持RL模型）
    """
    logger.info("Starting directory monitoring for real-time prediction with RL support...")
    logger.info(f"Monitoring directory: {DATA_DIR}")
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型（如果尚未加载）
    if use_rl:
        rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
        if not predictor.load_rl_model(rl_model_path):
            logger.warning("Failed to load reinforcement learning model in monitor function")
            use_rl = False
    
    # 记录已处理的文件
    processed_files = set()
    
    try:
        while True:
            # 获取目录中的所有CSV文件
            data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
            
            # 处理新文件
            for data_file in data_files:
                if data_file in processed_files:
                    continue
                
                logger.info(f"Processing new data file: {data_file}")
                
                # 加载数据
                df = load_realtime_data(data_file)
                if df is None:
                    logger.error(f"Failed to load data from {data_file}")
                    continue
                
                # 进行实时预测（最后一个点）
                if use_rl:
                    predicted_signal, confidence = predictor.predict_signal_with_rl(df, len(df) - 1)
                else:
                    predicted_signal, confidence = predictor.predict_realtime_signal(df)
                
                # 进行序列预测（最后100个点）
                sequence_predictions = []
                start_idx = max(10, len(df) - 100)
                for i in range(start_idx, len(df)):
                    if use_rl:
                        pred_signal, pred_conf = predictor.predict_signal_with_rl(df, i)
                    else:
                        pred_signal, pred_conf = predictor.predict_realtime_signal(df)
                    sequence_predictions.append({
                        'index': i,
                        'predicted_signal': pred_signal,
                        'confidence': pred_conf
                    })
                
                # 保存预测结果
                file_name = os.path.splitext(os.path.basename(data_file))[0]
                single_prediction_path = os.path.join(PREDICTIONS_DIR, f"realtime_prediction_{file_name}.json")
                
                single_result = {
                    'file': data_file,
                    'predicted_signal': int(predicted_signal),
                    'confidence': float(confidence),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                try:
                    with open(single_prediction_path, 'w', encoding='utf-8') as f:
                        json.dump(single_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Single point prediction saved to {single_prediction_path}")
                except Exception as e:
                    logger.error(f"Error saving single point prediction: {e}")
                
                # 保存序列预测结果
                sequence_prediction_path = os.path.join(PREDICTIONS_DIR, f"realtime_sequence_prediction_{file_name}.json")
                sequence_results = {
                    'file': data_file,
                    'predictions': sequence_predictions,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                try:
                    with open(sequence_prediction_path, 'w', encoding='utf-8') as f:
                        json.dump(sequence_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"Sequence prediction saved to {sequence_prediction_path}")
                except Exception as e:
                    logger.error(f"Error saving sequence prediction: {e}")
                
                # 生成可视化结果（使用实时预测适配的可视化函数）
                output_path = os.path.join(VISUALIZATION_DIR, f"realtime_prediction_{file_name}.png")
                visualize_realtime_predictions(df, sequence_predictions, output_path)
                
                logger.info(f"Visualization saved to {output_path}")
                
                # 标记为已处理
                processed_files.add(data_file)
            
            # 等待一段时间再检查
            import time
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        logger.info("Directory monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error in directory monitoring: {e}")

def visualize_realtime_predictions(df, predictions, output_path=None):
    """
    为实时预测结果生成可视化图表（不依赖实际标签）
    
    Parameters:
    df: DataFrame - 包含测试数据的DataFrame
    predictions: list - 预测结果列表
    output_path: str - 输出图像文件路径，默认为None（显示图像而不保存）
    """
    logger.info("Generating visualization of real-time predictions...")
    
    # 准备数据
    indices = [pred['index'] for pred in predictions]
    predicted_signals = [pred['predicted_signal'] for pred in predictions]
    index_values = [df.iloc[i]['index_value'] for i in indices]
    
    # 合并连续的同向开仓信号
    # 过滤预测信号，合并连续的同向开仓信号
    filtered_predictions = []
    last_long_position = False  # 是否已发出做多开仓信号
    last_short_position = False  # 是否已发出做空开仓信号
    
    for i, (idx, pred_signal) in enumerate(zip(indices, predicted_signals)):
        # 处理预测信号的合并逻辑
        if pred_signal == 1:  # 做多开仓
            if not last_long_position:  # 如果当前没有做多开仓信号
                filtered_predictions.append({
                    'index': idx,
                    'predicted_signal': pred_signal,
                    'index_value': index_values[i]
                })
                last_long_position = True
            # 如果已经发出了做多开仓信号，则忽略这个做多开仓信号
        elif pred_signal == 3:  # 做空开仓
            if not last_short_position:  # 如果当前没有做空开仓信号
                filtered_predictions.append({
                    'index': idx,
                    'predicted_signal': pred_signal,
                    'index_value': index_values[i]
                })
                last_short_position = True
            # 如果已经发出了做空开仓信号，则忽略这个做空开仓信号
        elif pred_signal == 2:  # 做多平仓
            filtered_predictions.append({
                'index': idx,
                'predicted_signal': pred_signal,
                'index_value': index_values[i]
            })
            last_long_position = False  # 重置做多开仓信号状态
        elif pred_signal == 4:  # 做空平仓
            filtered_predictions.append({
                'index': idx,
                'predicted_signal': pred_signal,
                'index_value': index_values[i]
            })
            last_short_position = False  # 重置做空开仓信号状态
        else:  # 无操作信号
            filtered_predictions.append({
                'index': idx,
                'predicted_signal': pred_signal,
                'index_value': index_values[i]
            })
            # 保持当前信号状态不变
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 绘制指数值曲线（上下翻转y值）
    y_values = np.array(index_values)
    y_flipped = -y_values  # 上下翻转y值
    ax1.plot(indices, y_flipped, 'b-', linewidth=1, label='指数值')
    ax1.set_xlabel('时间索引')
    ax1.set_ylabel('指数值 (翻转)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 标识过滤后的预测信号
    long_open_indices = []    # 做多开仓
    long_close_indices = []   # 做多平仓
    short_open_indices = []   # 做空开仓
    short_close_indices = []  # 做空平仓
    
    for pred in filtered_predictions:
        idx = pred['index']
        pred_signal = pred['predicted_signal']
        index_val = pred['index_value']
        
        if pred_signal == 1:  # 做多开仓
            long_open_indices.append((idx, -index_val))  # 上下翻转y值
        elif pred_signal == 2:  # 做多平仓
            long_close_indices.append((idx, -index_val))  # 上下翻转y值
        elif pred_signal == 3:  # 做空开仓
            short_open_indices.append((idx, -index_val))  # 上下翻转y值
        elif pred_signal == 4:  # 做空平仓
            short_close_indices.append((idx, -index_val))  # 上下翻转y값
    
    # 在图表上标识各种信号
    # 修正信号颜色：做多用绿色，做空用红色
    if long_open_indices:
        lo_idx, lo_val = zip(*long_open_indices)
        ax1.scatter(lo_idx, lo_val, color='green', marker='^', s=100, label='预测做多开仓', zorder=5)
        
    if long_close_indices:
        lc_idx, lc_val = zip(*long_close_indices)
        ax1.scatter(lc_idx, lc_val, color='green', marker='v', s=100, label='预测做多平仓', zorder=5)
        
    if short_open_indices:
        so_idx, so_val = zip(*short_open_indices)
        ax1.scatter(so_idx, so_val, color='red', marker='^', s=100, label='预测做空开仓', zorder=5)
        
    if short_close_indices:
        sc_idx, sc_val = zip(*short_close_indices)
        ax1.scatter(sc_idx, sc_val, color='red', marker='v', s=100, label='预测做空平仓', zorder=5)
    
    # 添加图例
    ax1.legend(loc='upper left')
    
    # 设置标题
    plt.title('实时预测结果可视化（合并连续开仓信号）', fontsize=16)
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    logger.info("Real-time visualization generation completed.")

def simulate_realtime_data_stream():
    """
    模拟实时数据流进行预测
    """
    logger.info("Starting simulated real-time data stream...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    if predictor.load_rl_model(rl_model_path):
        logger.info("Successfully loaded reinforcement learning model")
        use_rl = True
    else:
        logger.warning("Failed to load reinforcement learning model, using base predictions only")
        use_rl = False
    
    # 获取一些历史数据用于模拟
    label_files = sorted(glob.glob(os.path.join(CURRENT_DIR, "..", "label/", "*.csv")))
    if not label_files:
        logger.error("No label files found for simulation!")
        return
    
    # 使用第一个文件作为模拟数据源
    df = load_realtime_data(label_files[0])
    if df is None:
        logger.error("Failed to load data for simulation!")
        return
    
    logger.info(f"Using {label_files[0]} as simulation data source")
    
    # 模拟实时预测
    try:
        for i in range(len(df) - 100, len(df)):
            # 取前i个数据点作为当前数据
            current_df = df.iloc[:i+1].copy()
            
            # 进行预测
            if use_rl:
                predicted_signal, confidence = predictor.predict_signal_with_rl(current_df, len(current_df) - 1)
            else:
                predicted_signal, confidence = predictor.predict_realtime_signal(current_df)
            
            logger.info(f"Time step {i}: Predicted Signal = {predicted_signal}, Confidence = {confidence:.4f}")
            
            # 模拟延迟
            import time
            time.sleep(0.1)  # 100ms延迟
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    except Exception as e:
        logger.error(f"Error in simulation: {e}")

def interactive_prediction_mode():
    """
    交互式预测模式
    """
    logger.info("Starting interactive prediction mode...")
    
    # 创建预测器
    predictor = BalancedPatternPredictor()
    
    # 尝试加载强化学习模型
    rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
    if predictor.load_rl_model(rl_model_path):
        logger.info("Successfully loaded reinforcement learning model")
        use_rl = True
    else:
        logger.warning("Failed to load reinforcement learning model, using base predictions only")
        use_rl = False
    
    while True:
        try:
            print("\n" + "="*60)
            print("交互式实时预测模式")
            print("="*60)
            print("1. 从文件加载数据并预测")
            print("2. 目录监控模式")
            print("3. 数据模拟模式")
            print("4. 退出")
            print("-"*60)
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == "1":
                file_path = input("请输入CSV文件路径或目录路径: ").strip()
                if not os.path.exists(file_path):
                    print("路径不存在!")
                    continue
                
                # 确保可视化目录存在
                os.makedirs(VISUALIZATION_DIR, exist_ok=True)
                
                # 检查是否为目录
                if os.path.isdir(file_path):
                    # 处理目录中的所有CSV文件
                    csv_files = glob.glob(os.path.join(file_path, "*.csv"))
                    if not csv_files:
                        print("目录中没有找到CSV文件!")
                        continue
                    
                    print(f"找到 {len(csv_files)} 个CSV文件，开始批量预测...")
                    for i, csv_file in enumerate(csv_files):
                        print(f"处理文件 {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
                        
                        # 加载数据
                        df = load_realtime_data(csv_file)
                        if df is None:
                            print(f"加载数据失败: {csv_file}")
                            continue
                        
                        # 进行预测
                        if use_rl:
                            predicted_signal, confidence = predictor.predict_signal_with_rl(df, len(df) - 1)
                        else:
                            predicted_signal, confidence = predictor.predict_realtime_signal(df)
                        print(f"  预测结果:")
                        print(f"    预测信号: {predicted_signal}")
                        print(f"    置信度: {confidence:.4f}")
                        
                        # 进行序列预测（增加预测点数）
                        sequence_predictions = []
                        start_idx = max(10, min(200, len(df)) - 100)
                        for j in range(start_idx, min(start_idx + 200, len(df))):
                            if use_rl:
                                pred_signal, pred_conf = predictor.predict_signal_with_rl(df, j)
                            else:
                                pred_signal, pred_conf = predictor.predict_realtime_signal(df)
                            sequence_predictions.append({
                                'index': j,
                                'predicted_signal': pred_signal,
                                'confidence': pred_conf
                            })
                        print(f"    序列预测点数: {len(sequence_predictions)}")
                        
                        # 显示信号分布
                        signal_counts = {}
                        for pred in sequence_predictions:
                            signal = pred['predicted_signal']
                            signal_counts[signal] = signal_counts.get(signal, 0) + 1
                        print(f"    信号分布: {signal_counts}")
                        
                        # 保存结果
                        file_name = os.path.splitext(os.path.basename(csv_file))[0]
                        output_path = os.path.join(PREDICTIONS_DIR, f"interactive_prediction_{file_name}.json")
                        result = {
                            'file': csv_file,
                            'predicted_signal': int(predicted_signal),
                            'confidence': float(confidence),
                            'sequence_predictions': sequence_predictions,
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                        
                        try:
                            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                            print(f"    预测结果已保存到: {output_path}")
                        except Exception as e:
                            print(f"    保存预测结果时出错: {e}")
                        
                        # 生成可视化结果
                        try:
                            visualization_path = os.path.join(VISUALIZATION_DIR, f"interactive_prediction_{file_name}.png")
                            visualize_single_file_predictions(df, predicted_signal, sequence_predictions, visualization_path)
                            print(f"    可视化图表已保存到: {visualization_path}")
                        except Exception as e:
                            print(f"    生成可视化图表时出错: {e}")
                else:
                    # 处理单个文件
                    # 加载数据
                    df = load_realtime_data(file_path)
                    if df is None:
                        print("加载数据失败!")
                        continue
                    
                    # 进行预测
                    if use_rl:
                        predicted_signal, confidence = predictor.predict_signal_with_rl(df, len(df) - 1)
                    else:
                        predicted_signal, confidence = predictor.predict_realtime_signal(df)
                    print(f"\n预测结果:")
                    print(f"  预测信号: {predicted_signal}")
                    print(f"  置信度: {confidence:.4f}")
                    
                    # 进行序列预测（增加预测点数）
                    sequence_predictions = []
                    start_idx = max(10, min(200, len(df)) - 100)
                    for j in range(start_idx, min(start_idx + 200, len(df))):
                        if use_rl:
                            pred_signal, pred_conf = predictor.predict_signal_with_rl(df, j)
                        else:
                            pred_signal, pred_conf = predictor.predict_realtime_signal(df)
                        sequence_predictions.append({
                            'index': j,
                            'predicted_signal': pred_signal,
                            'confidence': pred_conf
                        })
                    print(f"  序列预测点数: {len(sequence_predictions)}")
                    
                    # 显示信号分布
                    signal_counts = {}
                    for pred in sequence_predictions:
                        signal = pred['predicted_signal']
                        signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    print(f"  信号分布: {signal_counts}")
                    
                    # 保存结果
                    save_choice = input("\n是否保存预测结果? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        output_path = os.path.join(PREDICTIONS_DIR, f"interactive_prediction_{file_name}.json")
                        result = {
                            'file': file_path,
                            'predicted_signal': int(predicted_signal),
                            'confidence': float(confidence),
                            'sequence_predictions': sequence_predictions,
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                        
                        try:
                            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                            print(f"预测结果已保存到: {output_path}")
                        except Exception as e:
                            print(f"保存预测结果时出错: {e}")
                    
                    # 生成可视化结果
                    try:
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        visualization_path = os.path.join(VISUALIZATION_DIR, f"interactive_prediction_{file_name}.png")
                        visualize_single_file_predictions(df, predicted_signal, sequence_predictions, visualization_path)
                        print(f"可视化图表已保存到: {visualization_path}")
                    except Exception as e:
                        print(f"生成可视化图表时出错: {e}")
            elif choice == "2":
                print("启动目录监控模式...")
                print(f"请将CSV文件放入以下目录进行实时预测: {DATA_DIR}")
                # 传递use_rl参数给监控函数
                monitor_directory_and_predict_with_rl(use_rl)
                
            elif choice == "3":
                print("启动数据模拟模式...")
                # 在模拟函数中已经处理了RL模型加载
                simulate_realtime_data_stream()
                
            elif choice == "4":
                print("退出交互模式.")
                break
                
            else:
                print("无效选择，请重新输入.")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断.")
            break
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='实时预测程序')
    parser.add_argument('--mode', choices=['monitor', 'simulate', 'interactive'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--file', help='用于预测的CSV文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        # 创建预测器并尝试加载RL模型
        predictor = BalancedPatternPredictor()
        rl_model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/rl_trader_model.json")
        use_rl = predictor.load_rl_model(rl_model_path)
        if use_rl:
            logger.info("Successfully loaded reinforcement learning model for monitor mode")
        else:
            logger.warning("Failed to load reinforcement learning model for monitor mode")
        monitor_directory_and_predict_with_rl(use_rl)
    elif args.mode == 'simulate':
        simulate_realtime_data_stream()
    elif args.mode == 'interactive':
        interactive_prediction_mode()
    else:
        interactive_prediction_mode()

if __name__ == "__main__":
    main()