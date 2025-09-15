# -*- coding: utf-8 -*-
"""
增强版实时预测器
使用增强版深度学习模型进行实时交易信号预测
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
from datetime import datetime
import time

# 导入增强版预测器
from enhanced_deep_learning_predictor import EnhancedDeepLearningPredictor

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "realtime_data/")  # 实时数据目录
PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "predictions/")  # 预测结果目录
VISUALIZATION_DIR = os.path.join(CURRENT_DIR, "visualization/")  # 可视化结果目录
MODELS_DIR = os.path.join(CURRENT_DIR, "models_enhanced/")  # 模型目录

def load_realtime_data(file_path: str) -> pd.DataFrame:
    """
    加载实时数据
    """
    try:
        df = pd.read_csv(file_path)
        
        # 检查必要的列
        required_columns = ['x', 'a', 'b', 'c', 'd', 'index_value']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"数据文件 {file_path} 缺少必要的列")
            return None
        
        # 数据清洗
        df = df.dropna(subset=required_columns)
        
        if len(df) < 50:  # 至少需要50个数据点
            logger.warning(f"数据文件 {file_path} 数据点不足")
            return None
        
        logger.info(f"成功加载数据: {file_path}, 数据点数: {len(df)}")
        return df
        
    except Exception as e:
        logger.error(f"加载数据文件 {file_path} 失败: {e}")
        return None

def ensure_trading_signals(predictions: list, df: pd.DataFrame) -> list:
    """
    确保每个交易日至少有一个开仓和一个平仓信号
    """
    if not predictions:
        return predictions
    
    # 统计信号类型
    signal_counts = Counter(p['predicted_signal'] for p in predictions)
    
    # 检查是否有开仓信号（1或3）
    has_open = signal_counts.get(1, 0) > 0 or signal_counts.get(3, 0) > 0
    
    # 检查是否有平仓信号（2或4）
    has_close = signal_counts.get(2, 0) > 0 or signal_counts.get(4, 0) > 0
    
    # 如果缺少开仓或平仓信号，智能添加
    if not has_open or not has_close:
        # 找到置信度最高的预测点
        max_conf_idx = max(range(len(predictions)), key=lambda i: predictions[i]['confidence'])
        
        if not has_open:
            # 根据趋势判断开仓方向
            recent_values = df['index_value'].tail(10).values
            if len(recent_values) >= 2:
                trend = recent_values[-1] - recent_values[0]
                signal = 1 if trend > 0 else 3  # 做多开仓或做空开仓
            else:
                signal = 1  # 默认做多开仓
            
            predictions[max_conf_idx]['predicted_signal'] = signal
            logger.info(f"添加开仓信号 {signal} 在索引 {max_conf_idx}")
        
        if not has_close:
            # 添加对应的平仓信号
            open_signals = [p['predicted_signal'] for p in predictions if p['predicted_signal'] in [1, 3]]
            if open_signals:
                last_open = open_signals[-1]
                close_signal = 2 if last_open == 1 else 4  # 对应的平仓信号
            else:
                close_signal = 2  # 默认做多平仓
            
            # 找一个不同的位置添加平仓信号
            close_idx = (max_conf_idx + len(predictions) // 2) % len(predictions)
            predictions[close_idx]['predicted_signal'] = close_signal
            logger.info(f"添加平仓信号 {close_signal} 在索引 {close_idx}")
    
    return predictions

def create_visualization(df: pd.DataFrame, predictions: list, output_path: str):
    """
    创建可视化图表
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # 主图：价格走势
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['index_value'], 'b-', linewidth=1, alpha=0.7, label='价格')
        
        # 信号颜色和标记配置
        signal_colors = {1: 'green', 2: 'green', 3: 'red', 4: 'red'}
        signal_markers = {1: '^', 2: 'v', 3: 'v', 4: '^'}
        signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        
        # 绘制预测信号
        for pred in predictions:
            idx = pred['index']
            signal = pred['predicted_signal']
            confidence = pred['confidence']
            
            if idx < len(df):
                plt.scatter(idx, df.iloc[idx]['index_value'], 
                          c=signal_colors[signal], marker=signal_markers[signal], 
                          s=100, alpha=0.8, edgecolors='black', linewidth=1,
                          label=f"{signal_names[signal]} (置信度: {confidence:.3f})")
        
        plt.title('增强版深度学习模型 - 交易信号预测', fontsize=14, fontweight='bold')
        plt.xlabel('时间点')
        plt.ylabel('价格')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 子图：技术指标
        plt.subplot(2, 1, 2)
        
        # 绘制一些基础技术指标
        if len(df) > 20:
            # 简单移动平均
            sma_5 = df['index_value'].rolling(window=5).mean()
            sma_20 = df['index_value'].rolling(window=20).mean()
            
            plt.plot(df.index, sma_5, 'orange', linewidth=1, alpha=0.7, label='SMA(5)')
            plt.plot(df.index, sma_20, 'purple', linewidth=1, alpha=0.7, label='SMA(20)')
        
        plt.title('技术指标', fontsize=12)
        plt.xlabel('时间点')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图表已保存: {output_path}")
        
    except Exception as e:
        logger.error(f"创建可视化图表失败: {e}")

def predict_single_file(predictor: EnhancedDeepLearningPredictor, file_path: str) -> dict:
    """
    对单个文件进行预测
    """
    try:
        # 加载数据
        df = load_realtime_data(file_path)
        if df is None:
            return None
        
        # 提取特征
        features, _ = predictor.extract_enhanced_features(df)
        if len(features) == 0:
            logger.warning(f"文件 {file_path} 无有效特征")
            return None
        
        # 进行预测
        predictions, confidences = predictor.predict_ensemble(features)
        
        # 构建预测结果
        prediction_results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            prediction_results.append({
                'index': i,
                'predicted_signal': pred + 1,  # 转换回1-4
                'confidence': float(conf),
                'timestamp': datetime.now().isoformat()
            })
        
        # 确保信号完整性
        prediction_results = ensure_trading_signals(prediction_results, df)
        
        # 统计信号
        signal_stats = Counter(p['predicted_signal'] for p in prediction_results)
        
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'data_points': len(df),
            'predictions': prediction_results,
            'signal_statistics': dict(signal_stats),
            'total_signals': len(prediction_results),
            'avg_confidence': np.mean(confidences),
            'prediction_time': datetime.now().isoformat()
        }
        
        # 保存预测结果
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # JSON结果
        json_path = os.path.join(PREDICTIONS_DIR, f"enhanced_prediction_{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 可视化
        viz_path = os.path.join(VISUALIZATION_DIR, f"enhanced_prediction_{base_name}.png")
        create_visualization(df, prediction_results, viz_path)
        
        logger.info(f"预测完成: {file_path}")
        logger.info(f"信号统计: {signal_stats}")
        logger.info(f"平均置信度: {np.mean(confidences):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"预测文件 {file_path} 时出错: {e}")
        return None

def monitor_directory_and_predict(predictor: EnhancedDeepLearningPredictor):
    """
    监控目录并进行实时预测
    """
    logger.info("开始目录监控模式...")
    logger.info(f"监控目录: {DATA_DIR}")
    
    processed_files = set()
    
    try:
        while True:
            # 获取目录中的所有CSV文件
            data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
            
            # 处理新文件
            for data_file in data_files:
                if data_file in processed_files:
                    continue
                
                logger.info(f"发现新文件: {data_file}")
                result = predict_single_file(predictor, data_file)
                
                if result:
                    processed_files.add(data_file)
                    logger.info(f"文件处理完成: {data_file}")
                else:
                    logger.warning(f"文件处理失败: {data_file}")
            
            # 等待新文件
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("监控已停止")
    except Exception as e:
        logger.error(f"监控过程出错: {e}")

def simulate_realtime_prediction(predictor: EnhancedDeepLearningPredictor):
    """
    使用历史数据模拟实时预测
    """
    logger.info("开始数据模拟模式...")
    
    # 使用label目录中的文件进行模拟
    label_files = glob.glob("label/*.csv")
    if not label_files:
        logger.error("没有找到标签文件用于模拟")
        return
    
    # 使用最后几个文件进行模拟
    test_files = label_files[-5:]
    logger.info(f"使用 {len(test_files)} 个文件进行模拟")
    
    results = []
    for file_path in test_files:
        logger.info(f"模拟预测: {file_path}")
        result = predict_single_file(predictor, file_path)
        if result:
            results.append(result)
        
        # 模拟实时间隔
        time.sleep(2)
    
    # 输出模拟结果统计
    if results:
        total_signals = sum(r['total_signals'] for r in results)
        avg_confidence = np.mean([r['avg_confidence'] for r in results])
        
        logger.info(f"\n=== 模拟预测完成 ===")
        logger.info(f"处理文件数: {len(results)}")
        logger.info(f"总信号数: {total_signals}")
        logger.info(f"平均置信度: {avg_confidence:.4f}")
        
        # 信号类型统计
        all_signals = []
        for r in results:
            all_signals.extend([p['predicted_signal'] for p in r['predictions']])
        
        signal_distribution = Counter(all_signals)
        logger.info(f"信号分布: {dict(signal_distribution)}")

def interactive_prediction(predictor: EnhancedDeepLearningPredictor):
    """
    交互式预测模式
    """
    logger.info("开始交互模式...")
    
    while True:
        try:
            print("\n=== 增强版深度学习预测器 - 交互模式 ===")
            print("1. 预测单个文件")
            print("2. 批量预测目录")
            print("3. 查看模型信息")
            print("4. 退出")
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == '1':
                file_path = input("请输入文件路径: ").strip()
                if os.path.exists(file_path):
                    result = predict_single_file(predictor, file_path)
                    if result:
                        print(f"预测完成！信号统计: {result['signal_statistics']}")
                    else:
                        print("预测失败")
                else:
                    print("文件不存在")
            
            elif choice == '2':
                dir_path = input("请输入目录路径: ").strip()
                if os.path.exists(dir_path):
                    csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
                    print(f"找到 {len(csv_files)} 个CSV文件")
                    
                    for file_path in csv_files:
                        print(f"预测: {file_path}")
                        predict_single_file(predictor, file_path)
                    
                    print("批量预测完成")
                else:
                    print("目录不存在")
            
            elif choice == '3':
                print(f"\n=== 模型信息 ===")
                print(f"模型数量: {len(predictor.models)}")
                print(f"输入维度: {predictor.input_dim}")
                print(f"设备: {predictor.device}")
                print(f"模型目录: {predictor.models_dir}")
            
            elif choice == '4':
                print("退出交互模式")
                break
            
            else:
                print("无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"操作出错: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='增强版深度学习实时预测器')
    parser.add_argument('--mode', choices=['monitor', 'simulate', 'interactive'], 
                       default='interactive', help='运行模式')
    
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # 创建预测器并加载模型
    predictor = EnhancedDeepLearningPredictor()
    
    model_path = os.path.join(MODELS_DIR, "enhanced_predictor")
    if not predictor.load_models(model_path):
        logger.error("无法加载增强版模型！")
        logger.info("请先运行训练脚本: python enhanced_deep_learning_predictor.py")
        return
    
    logger.info(f"成功加载增强版模型，包含 {len(predictor.models)} 个子模型")
    
    # 根据模式运行
    if args.mode == 'monitor':
        monitor_directory_and_predict(predictor)
    elif args.mode == 'simulate':
        simulate_realtime_prediction(predictor)
    elif args.mode == 'interactive':
        interactive_prediction(predictor)

if __name__ == "__main__":
    main()