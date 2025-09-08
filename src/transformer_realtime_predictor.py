# -*- coding: utf-8 -*-
"""
使用Transformer模型的实时预测程序
基于已训练的Transformer模型进行实时交易信号预测
目标：每个交易日(单独的csv文件)至少有一个开仓和一个相应平仓交易信号
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
import torch

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_predictor import TradingSignalTransformer
from pattern_predictor_balanced import load_realtime_data

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
    logger.info("Starting directory monitoring for real-time prediction with Transformer model...")
    logger.info(f"Monitoring directory: {DATA_DIR}")
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # 创建预测器并加载Transformer模型
    predictor = TradingSignalTransformer()
    model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/transformer_predictor.pth")
    if not predictor.load_model(model_path):
        logger.error("Failed to load Transformer model!")
        return
    
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
                
                # 进行序列预测（整个文件）
                predictions, confidences = predictor.predict(df)
                
                # 构建预测结果
                sequence_predictions = []
                for i in range(len(predictions)):
                    sequence_predictions.append({
                        'index': i,
                        'predicted_signal': predictions[i],
                        'confidence': confidences[i]
                    })
                
                # 确保至少有一个开仓和一个平仓信号
                sequence_predictions = ensure_trading_signals(sequence_predictions, df)
                
                # 保存序列预测结果
                file_name = os.path.splitext(os.path.basename(data_file))[0]
                sequence_prediction_path = os.path.join(PREDICTIONS_DIR, f"transformer_sequence_prediction_{file_name}.json")
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
                
                # 生成可视化结果
                output_path = os.path.join(VISUALIZATION_DIR, f"transformer_prediction_{file_name}.png")
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

def ensure_trading_signals(predictions, df):
    """
    确保每个交易日至少有一个开仓和一个平仓信号
    
    Args:
        predictions: 预测结果列表
        df: 原始数据DataFrame
        
    Returns:
        modified_predictions: 修改后的预测结果列表
    """
    modified_predictions = predictions.copy()
    
    # 检查是否已有开仓和平仓信号
    has_long_open = any(pred['predicted_signal'] == 1 for pred in modified_predictions)
    has_long_close = any(pred['predicted_signal'] == 2 for pred in modified_predictions)
    has_short_open = any(pred['predicted_signal'] == 3 for pred in modified_predictions)
    has_short_close = any(pred['predicted_signal'] == 4 for pred in modified_predictions)
    
    # 如果没有开仓信号，添加一个
    if not has_long_open and not has_short_open:
        # 选择一个高置信度的点作为开仓信号
        max_conf_idx = max(range(len(modified_predictions)), 
                          key=lambda i: modified_predictions[i]['confidence'])
        # 根据趋势判断开仓方向
        if len(df) > 10:
            # 简单趋势判断：如果最后几个点呈上升趋势，则做多，否则做空
            recent_values = df['index_value'].tail(5).values
            trend = recent_values[-1] - recent_values[0]
            signal = 1 if trend > 0 else 3  # 做多开仓或做空开仓
            modified_predictions[max_conf_idx]['predicted_signal'] = signal
            logger.info(f"Added opening signal {signal} at index {max_conf_idx}")
    
    # 如果没有平仓信号，添加一个
    if not has_long_close and not has_short_close:
        # 选择一个距离开仓信号较远的点作为平仓信号
        open_signal_indices = [i for i, pred in enumerate(modified_predictions) 
                              if pred['predicted_signal'] in [1, 3]]
        if open_signal_indices:
            open_idx = open_signal_indices[0]
            # 选择距离开仓信号较远的点
            close_idx = len(modified_predictions) - 1 if open_idx < len(modified_predictions) // 2 else 0
            # 平仓信号应该与开仓信号对应
            open_signal = modified_predictions[open_idx]['predicted_signal']
            close_signal = 2 if open_signal == 1 else 4  # 做多平仓或做空平仓
            modified_predictions[close_idx]['predicted_signal'] = close_signal
            logger.info(f"Added closing signal {close_signal} at index {close_idx}")
    
    return modified_predictions

def visualize_realtime_predictions(df, predictions, output_path=None):
    """
    为实时预测结果生成可视化图表
    
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
    
    # 过滤预测信号，只保留交易信号（做多开仓、做多平仓、做空开仓、做空平仓）
    filtered_predictions = []
    for i, (idx, pred_signal) in enumerate(zip(indices, predicted_signals)):
        # 只处理交易信号，过滤掉无操作信号(0)
        if pred_signal != 0:
            filtered_predictions.append({
                'index': idx,
                'predicted_signal': pred_signal,
                'index_value': index_values[i]
            })
    
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
    plt.title('Transformer模型的实时预测结果可视化', fontsize=16)
    
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
    
    # 创建预测器并加载Transformer模型
    predictor = TradingSignalTransformer()
    model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/transformer_predictor.pth")
    if not predictor.load_model(model_path):
        logger.error("Failed to load Transformer model!")
        return
    
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
    
    # 进行预测
    try:
        predictions, confidences = predictor.predict(df)
        
        # 确保至少有一个开仓和一个平仓信号
        pred_list = [{'index': i, 'predicted_signal': predictions[i], 'confidence': confidences[i]} 
                     for i in range(len(predictions))]
        pred_list = ensure_trading_signals(pred_list, df)
        
        logger.info(f"Generated {len([p for p in pred_list if p['predicted_signal'] != 0])} trading signals")
        
        # 显示信号分布
        signal_counts = Counter(pred['predicted_signal'] for pred in pred_list)
        logger.info(f"Signal distribution: {dict(signal_counts)}")
        
        # 显示一些高置信度的信号
        high_conf_signals = [pred for pred in pred_list if pred['confidence'] > 0.5]
        logger.info(f"High confidence signals (>0.5): {len(high_conf_signals)}")
        
        for pred in high_conf_signals[:5]:  # 显示前5个高置信度信号
            logger.info(f"  Index {pred['index']}: Signal {pred['predicted_signal']}, Confidence {pred['confidence']:.4f}")
            
    except Exception as e:
        logger.error(f"Error in simulation: {e}")

def interactive_prediction_mode():
    """
    交互式预测模式
    """
    logger.info("Starting interactive prediction mode...")
    
    # 创建预测器并加载Transformer模型
    predictor = TradingSignalTransformer()
    model_path = os.path.join(CURRENT_DIR, "..", "model/balanced_model/transformer_predictor.pth")
    if not predictor.load_model(model_path):
        logger.error("Failed to load Transformer model!")
        return
    
    while True:
        try:
            print("\n" + "="*60)
            print("交互式实时预测模式（Transformer模型版）")
            print("="*60)
            print("1. 从文件加载数据并预测")
            print("2. 目录监控模式")
            print("3. 数据模拟模式")
            print("4. 退出")
            print("-"*60)
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == "1":
                file_path = input("请输入CSV文件路径: ").strip()
                if not os.path.exists(file_path):
                    print("路径不存在!")
                    continue
                
                # 加载数据
                df = load_realtime_data(file_path)
                if df is None:
                    print("加载数据失败!")
                    continue
                
                # 进行预测
                predictions, confidences = predictor.predict(df)
                
                # 构建预测结果
                pred_list = [{'index': i, 'predicted_signal': predictions[i], 'confidence': confidences[i]} 
                             for i in range(len(predictions))]
                
                # 确保至少有一个开仓和一个平仓信号
                pred_list = ensure_trading_signals(pred_list, df)
                
                print(f"\n预测结果:")
                print(f"  总预测点数: {len(pred_list)}")
                
                # 显示信号分布
                signal_counts = Counter(pred['predicted_signal'] for pred in pred_list)
                print(f"  信号分布: {dict(signal_counts)}")
                
                # 显示交易信号
                trading_signals = [pred for pred in pred_list if pred['predicted_signal'] != 0]
                print(f"  交易信号数: {len(trading_signals)}")
                for signal in trading_signals[:10]:  # 显示前10个交易信号
                    print(f"    Index {signal['index']}: Signal {signal['predicted_signal']}, Confidence {signal['confidence']:.4f}")
                
                # 保存结果
                save_choice = input("\n是否保存预测结果? (y/n): ").strip().lower()
                if save_choice == 'y':
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_path = os.path.join(PREDICTIONS_DIR, f"transformer_interactive_prediction_{file_name}.json")
                    result = {
                        'file': file_path,
                        'predictions': pred_list,
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
                    visualization_path = os.path.join(VISUALIZATION_DIR, f"transformer_interactive_prediction_{file_name}.png")
                    visualize_realtime_predictions(df, pred_list, visualization_path)
                    print(f"可视化图表已保存到: {visualization_path}")
                except Exception as e:
                    print(f"生成可视化图表时出错: {e}")
                    
            elif choice == "2":
                print("启动目录监控模式...")
                print(f"请将CSV文件放入以下目录进行实时预测: {DATA_DIR}")
                monitor_directory_and_predict()
                
            elif choice == "3":
                print("启动数据模拟模式...")
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
    parser = argparse.ArgumentParser(description='使用Transformer模型的实时预测程序')
    parser.add_argument('--mode', choices=['monitor', 'simulate', 'interactive'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--file', help='用于预测的CSV文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_directory_and_predict()
    elif args.mode == 'simulate':
        simulate_realtime_data_stream()
    elif args.mode == 'interactive':
        interactive_prediction_mode()
    else:
        interactive_prediction_mode()

if __name__ == "__main__":
    main()