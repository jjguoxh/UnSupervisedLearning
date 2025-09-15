# -*- coding: utf-8 -*-
"""
训练数据预处理脚本
将现有的标签数据转换为适合增强版深度学习模型的格式
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "label/")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data/")

def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于价格变化和技术指标生成交易信号
    """
    df = df.copy()
    
    # 计算价格变化率
    df['price_change'] = df['index_value'].pct_change()
    df['price_change_ma'] = df['price_change'].rolling(window=5).mean()
    
    # 计算移动平均线
    df['sma_5'] = df['index_value'].rolling(window=5).mean()
    df['sma_20'] = df['index_value'].rolling(window=20).mean()
    
    # 计算RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['index_value'])
    
    # 计算MACD
    exp1 = df['index_value'].ewm(span=12).mean()
    exp2 = df['index_value'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # 初始化信号列
    df['signal'] = 0
    
    # 生成交易信号的逻辑
    for i in range(20, len(df)):  # 从第20行开始，确保有足够的历史数据
        current_price = df.iloc[i]['index_value']
        prev_price = df.iloc[i-1]['index_value']
        
        # 获取技术指标
        rsi = df.iloc[i]['rsi']
        macd = df.iloc[i]['macd']
        macd_signal = df.iloc[i]['macd_signal']
        sma_5 = df.iloc[i]['sma_5']
        sma_20 = df.iloc[i]['sma_20']
        
        # 价格变化
        price_change = (current_price - prev_price) / prev_price
        
        # 检查前一个信号
        prev_signal = df.iloc[i-1]['signal'] if i > 0 else 0
        
        # 做多开仓信号 (1)
        if (rsi < 30 and  # RSI超卖
            macd > macd_signal and  # MACD金叉
            sma_5 > sma_20 and  # 短期均线在长期均线之上
            price_change > 0.001 and  # 价格上涨
            prev_signal not in [1, 2]):  # 避免重复开仓
            df.iloc[i, df.columns.get_loc('signal')] = 1
        
        # 做多平仓信号 (2)
        elif (rsi > 70 and  # RSI超买
              macd < macd_signal and  # MACD死叉
              prev_signal == 1):  # 之前有做多开仓
            df.iloc[i, df.columns.get_loc('signal')] = 2
        
        # 做空开仓信号 (3)
        elif (rsi > 70 and  # RSI超买
              macd < macd_signal and  # MACD死叉
              sma_5 < sma_20 and  # 短期均线在长期均线之下
              price_change < -0.001 and  # 价格下跌
              prev_signal not in [3, 4]):  # 避免重复开仓
            df.iloc[i, df.columns.get_loc('signal')] = 3
        
        # 做空平仓信号 (4)
        elif (rsi < 30 and  # RSI超卖
              macd > macd_signal and  # MACD金叉
              prev_signal == 3):  # 之前有做空开仓
            df.iloc[i, df.columns.get_loc('signal')] = 4
        
        # 保持前一个信号（如果没有新信号）
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0
    
    return df

def enhance_signals_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强信号分布，确保每种信号类型都有足够的样本
    """
    df = df.copy()
    
    # 统计当前信号分布
    signal_counts = df['signal'].value_counts()
    logger.info(f"原始信号分布: {dict(signal_counts)}")
    
    # 如果某些信号类型太少，增加一些
    min_samples = 50  # 每种信号至少50个样本
    
    for signal_type in [1, 2, 3, 4]:
        current_count = signal_counts.get(signal_type, 0)
        
        if current_count < min_samples:
            needed = min_samples - current_count
            logger.info(f"信号类型 {signal_type} 需要增加 {needed} 个样本")
            
            # 找到价格变化较大的点，添加相应信号
            df_copy = df.copy()
            df_copy['abs_change'] = abs(df_copy['index_value'].pct_change())
            
            # 选择变化最大的点
            candidates = df_copy[df_copy['signal'] == 0].nlargest(needed * 2, 'abs_change')
            
            # 随机选择一些点设置为目标信号
            selected_indices = np.random.choice(candidates.index, 
                                              min(needed, len(candidates)), 
                                              replace=False)
            
            for idx in selected_indices:
                df.loc[idx, 'signal'] = signal_type
    
    # 重新统计信号分布
    final_counts = df['signal'].value_counts()
    logger.info(f"增强后信号分布: {dict(final_counts)}")
    
    return df

def process_single_file(file_path: str) -> bool:
    """
    处理单个文件
    """
    try:
        # 读取原始数据
        df = pd.read_csv(file_path)
        
        # 检查必要的列
        required_columns = ['x', 'a', 'b', 'c', 'd', 'index_value']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"文件 {file_path} 缺少必要的列，跳过")
            return False
        
        # 数据清洗
        df = df.dropna(subset=required_columns)
        
        if len(df) < 100:  # 至少需要100个数据点
            logger.warning(f"文件 {file_path} 数据点不足，跳过")
            return False
        
        # 生成交易信号
        df_with_signals = generate_trading_signals(df)
        
        # 增强信号分布
        df_enhanced = enhance_signals_distribution(df_with_signals)
        
        # 保存处理后的文件
        base_name = os.path.basename(file_path)
        output_path = os.path.join(PROCESSED_DIR, f"processed_{base_name}")
        
        # 只保留训练需要的列
        columns_to_save = ['x', 'a', 'b', 'c', 'd', 'index_value', 'signal']
        df_final = df_enhanced[columns_to_save].copy()
        
        df_final.to_csv(output_path, index=False)
        
        # 统计信息
        signal_stats = df_final['signal'].value_counts()
        total_signals = sum(signal_stats[signal_stats.index != 0])
        
        logger.info(f"处理完成: {file_path}")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"数据点数: {len(df_final)}")
        logger.info(f"有效信号数: {total_signals}")
        logger.info(f"信号分布: {dict(signal_stats)}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return False

def create_sample_data():
    """
    创建一些示例训练数据（如果没有足够的真实数据）
    """
    logger.info("创建示例训练数据...")
    
    np.random.seed(42)
    
    for i in range(5):
        # 生成模拟的交易数据
        n_points = 1000
        
        # 基础价格走势
        base_price = 3000
        price_trend = np.cumsum(np.random.randn(n_points) * 0.01)
        prices = base_price + price_trend
        
        # 生成特征数据
        x = np.arange(n_points)
        a = np.random.randn(n_points) * 0.5 + 0.3  # 开盘相关
        b = np.random.randn(n_points) * 0.3 + 0.5  # 最高价相关
        c = np.random.randn(n_points) * 0.4 - 0.2  # 最低价相关
        d = np.random.randn(n_points) * 0.3 + 0.1  # 收盘相关
        
        # 创建DataFrame
        df = pd.DataFrame({
            'x': x,
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'index_value': prices
        })
        
        # 生成交易信号
        df_with_signals = generate_trading_signals(df)
        df_enhanced = enhance_signals_distribution(df_with_signals)
        
        # 保存文件
        output_path = os.path.join(PROCESSED_DIR, f"sample_training_data_{i+1:02d}.csv")
        columns_to_save = ['x', 'a', 'b', 'c', 'd', 'index_value', 'signal']
        df_enhanced[columns_to_save].to_csv(output_path, index=False)
        
        logger.info(f"创建示例数据: {output_path}")

def main():
    """
    主函数
    """
    logger.info("开始训练数据预处理...")
    
    # 确保输出目录存在
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    
    if not label_files:
        logger.warning(f"在 {LABEL_DIR} 目录中没有找到标签文件")
        logger.info("将创建示例训练数据")
        create_sample_data()
        return
    
    logger.info(f"找到 {len(label_files)} 个标签文件")
    
    # 处理每个文件
    processed_count = 0
    for file_path in label_files:
        if process_single_file(file_path):
            processed_count += 1
    
    logger.info(f"成功处理 {processed_count}/{len(label_files)} 个文件")
    
    # 如果处理的文件太少，创建一些示例数据
    if processed_count < 3:
        logger.info("处理的文件数量不足，创建额外的示例数据")
        create_sample_data()
    
    # 统计处理结果
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    logger.info(f"\n=== 预处理完成 ===")
    logger.info(f"输出目录: {PROCESSED_DIR}")
    logger.info(f"处理后文件数: {len(processed_files)}")
    
    # 统计总体信号分布
    total_signals = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
    total_samples = 0
    
    for file_path in processed_files:
        try:
            df = pd.read_csv(file_path)
            signal_counts = df['signal'].value_counts()
            total_samples += len(df)
            
            for signal, count in signal_counts.items():
                total_signals[signal] = total_signals.get(signal, 0) + count
                
        except Exception as e:
            logger.error(f"统计文件 {file_path} 时出错: {e}")
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"总信号分布: {dict(total_signals)}")
    
    # 计算信号比例
    valid_signals = sum(total_signals[i] for i in [1, 2, 3, 4])
    if valid_signals > 0:
        logger.info(f"有效信号比例: {valid_signals/total_samples:.2%}")
        
        for signal_type in [1, 2, 3, 4]:
            ratio = total_signals[signal_type] / valid_signals
            signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
            logger.info(f"{signal_names[signal_type]}: {total_signals[signal_type]} ({ratio:.1%})")
    
    logger.info("\n现在可以运行增强版模型训练:")
    logger.info("python train_enhanced_model.py")

if __name__ == "__main__":
    main()