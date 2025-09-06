# -*- coding: utf-8 -*-
"""
序列模式分析器
分析交易信号序列模式
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
LABEL_DIR = "../label/"  # 标签数据目录
OUTPUT_DIR = "../sequence_patterns/"  # 序列模式输出目录
CONTEXT_WINDOW = 50  # 信号前后上下文窗口大小

def load_data(file_path):
    """
    加载单个CSV文件数据
    """
    df = pd.read_csv(file_path)
    return df

def extract_trading_sequences(df):
    """
    从数据中提取交易序列
    """
    # 找到所有交易信号
    signals = df[df['label'] != 0].copy()
    signals['original_index'] = signals.index
    
    # 将信号转换为序列
    sequences = []
    signal_list = signals.to_dict('records')
    
    for i in range(len(signal_list)):
        signal = signal_list[i]
        
        # 提取信号前后的时间序列上下文
        idx = signal['original_index']
        start_idx = max(0, idx - CONTEXT_WINDOW)
        end_idx = min(len(df), idx + CONTEXT_WINDOW)
        
        context = df.iloc[start_idx:end_idx][['x', 'a', 'b', 'c', 'd', 'index_value']].copy()
        context['distance_from_signal'] = range(start_idx - idx, end_idx - idx)
        context['is_signal'] = context.index == idx
        
        sequence = {
            'signal_index': idx,
            'signal_label': signal['label'],
            'context': context,
            'index_value_at_signal': signal['index_value'],
            'timestamp': signal['x']
        }
        
        sequences.append(sequence)
    
    return sequences

def identify_trading_pairs(sequences):
    """
    识别交易对（开仓-平仓）
    """
    trading_pairs = []
    
    # 按时间顺序排列信号
    sorted_sequences = sorted(sequences, key=lambda x: x['signal_index'])
    
    i = 0
    while i < len(sorted_sequences) - 1:
        open_signal = sorted_sequences[i]
        close_signal = sorted_sequences[i + 1]
        
        # 检查是否为有效的交易对
        # 做多交易对：标签1(开仓) -> 标签2(平仓)
        # 做空交易对：标签3(开仓) -> 标签4(平仓)
        is_long_pair = (open_signal['signal_label'] == 1 and close_signal['signal_label'] == 2)
        is_short_pair = (open_signal['signal_label'] == 3 and close_signal['signal_label'] == 4)
        
        if is_long_pair or is_short_pair:
            pair = {
                'open_signal': open_signal,
                'close_signal': close_signal,
                'pair_type': 'long' if is_long_pair else 'short',
                'price_change': (close_signal['index_value_at_signal'] - open_signal['index_value_at_signal']) / open_signal['index_value_at_signal'],
                'time_diff': close_signal['signal_index'] - open_signal['signal_index']
            }
            trading_pairs.append(pair)
            i += 2  # 跳过已配对的两个信号
        else:
            i += 1
    
    return trading_pairs

def analyze_pattern_features(context_data):
    """
    分析模式特征
    """
    features = {}
    
    # 基本统计特征
    features['mean_index_value'] = context_data['index_value'].mean()
    features['std_index_value'] = context_data['index_value'].std()
    features['min_index_value'] = context_data['index_value'].min()
    features['max_index_value'] = context_data['index_value'].max()
    
    # 影响因子特征
    for factor in ['a', 'b', 'c', 'd']:
        features[f'mean_{factor}'] = context_data[factor].mean()
        features[f'std_{factor}'] = context_data[factor].std()
    
    # 趋势特征
    index_values = context_data['index_value'].values
    if len(index_values) > 1:
        features['trend_slope'] = np.polyfit(range(len(index_values)), index_values, 1)[0]
    else:
        features['trend_slope'] = 0
    
    return features

def extract_pattern_features(sequences):
    """
    提取模式特征
    """
    pattern_features = []
    
    for seq in sequences:
        # 提取信号前的上下文数据
        context_before = seq['context'][seq['context']['distance_from_signal'] < 0]
        if len(context_before) > 5:  # 确保有足够的数据
            features = analyze_pattern_features(context_before)
            features['signal_label'] = seq['signal_label']
            features['signal_index'] = seq['signal_index']
            pattern_features.append(features)
    
    return pattern_features

def calculate_pair_statistics(trading_pairs):
    """
    计算交易对统计信息
    """
    if not trading_pairs:
        return {}
    
    long_pairs = [p for p in trading_pairs if p['pair_type'] == 'long']
    short_pairs = [p for p in trading_pairs if p['pair_type'] == 'short']
    
    stats = {
        'total_pairs': len(trading_pairs),
        'long_pairs': len(long_pairs),
        'short_pairs': len(short_pairs),
        'long_win_rate': np.mean([1 if p['price_change'] > 0 else 0 for p in long_pairs]) if long_pairs else 0,
        'short_win_rate': np.mean([1 if p['price_change'] < 0 else 0 for p in short_pairs]) if short_pairs else 0,
        'avg_long_return': np.mean([p['price_change'] for p in long_pairs]) if long_pairs else 0,
        'avg_short_return': np.mean([p['price_change'] for p in short_pairs]) if short_pairs else 0,
        'long_profit_factor': (
            np.sum([p['price_change'] for p in long_pairs if p['price_change'] > 0]) /
            abs(np.sum([p['price_change'] for p in long_pairs if p['price_change'] < 0])) 
            if long_pairs and np.sum([p['price_change'] for p in long_pairs if p['price_change'] < 0]) != 0 else 0
        ),
        'short_profit_factor': (
            np.sum([p['price_change'] for p in short_pairs if p['price_change'] < 0]) /
            abs(np.sum([p['price_change'] for p in short_pairs if p['price_change'] > 0])) 
            if short_pairs and np.sum([p['price_change'] for p in short_pairs if p['price_change'] > 0]) != 0 else 0
        )
    }
    
    return stats

def visualize_trading_pairs(trading_pairs, file_name, top_k=5):
    """
    可视化交易对
    """
    if not trading_pairs:
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 按收益排序
    sorted_pairs = sorted(trading_pairs, key=lambda x: abs(x['price_change']), reverse=True)
    
    # 只显示前top_k个交易对
    display_pairs = sorted_pairs[:min(top_k, len(sorted_pairs))]
    
    fig, axes = plt.subplots(len(display_pairs), 1, figsize=(15, 4*len(display_pairs)))
    if len(display_pairs) == 1:
        axes = [axes]
    
    for i, pair in enumerate(display_pairs):
        ax = axes[i] if len(display_pairs) > 1 else axes[0]
        
        # 绘制开仓信号前后的时间序列
        open_context = pair['open_signal']['context']
        ax.plot(open_context['distance_from_signal'], open_context['index_value'], 'b-', linewidth=2, label='Price')
        
        # 标记开仓和平仓点
        open_point = open_context[open_context['is_signal']]
        close_point = pair['close_signal']['context'][pair['close_signal']['context']['is_signal']]
        
        ax.scatter(open_point['distance_from_signal'], open_point['index_value'], 
                  color='green', s=100, zorder=5, label=f"Open ({pair['open_signal']['signal_label']})")
        ax.scatter(close_point['distance_from_signal'] + (pair['close_signal']['signal_index'] - pair['open_signal']['signal_index']), 
                  close_point['index_value'], 
                  color='red', s=100, zorder=5, label=f"Close ({pair['close_signal']['signal_label']})")
        
        pair_type = "Long" if pair['pair_type'] == 'long' else "Short"
        ax.set_title(f'{file_name} - {pair_type} Pair (Return: {pair["price_change"]:.4f})')
        ax.set_xlabel('Time (relative to open signal)')
        ax.set_ylabel('Index Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{file_name}_trading_pairs.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_results(pair_stats, pattern_features):
    """
    保存分析结果
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存交易对统计信息
    stats_df = pd.DataFrame([pair_stats])
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'pair_statistics.csv'), index=False)
    
    # 保存模式特征
    if pattern_features:
        features_df = pd.DataFrame(pattern_features)
        features_df.to_csv(os.path.join(OUTPUT_DIR, 'pattern_features.csv'), index=False)
    
    # 打印统计结果
    print("\nTrading Pair Statistics:")
    print("=" * 40)
    print(f"Total Pairs: {pair_stats['total_pairs']}")
    print(f"Long Pairs: {pair_stats['long_pairs']}")
    print(f"Short Pairs: {pair_stats['short_pairs']}")
    print(f"Long Win Rate: {pair_stats['long_win_rate']:.2%}")
    print(f"Short Win Rate: {pair_stats['short_win_rate']:.2%}")
    print(f"Avg Long Return: {pair_stats['avg_long_return']:.4f}")
    print(f"Avg Short Return: {pair_stats['avg_short_return']:.4f}")

def process_single_file(file_path):
    """
    处理单个文件
    """
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    
    # 加载数据
    df = load_data(file_path)
    
    # 提取交易序列
    sequences = extract_trading_sequences(df)
    print(f"  Found {len(sequences)} trading signals")
    
    if len(sequences) == 0:
        return [], [], {}
    
    # 识别交易对
    trading_pairs = identify_trading_pairs(sequences)
    print(f"  Identified {len(trading_pairs)} trading pairs")
    
    # 提取模式特征
    pattern_features = extract_pattern_features(sequences)
    print(f"  Extracted {len(pattern_features)} pattern features")
    
    # 可视化交易对
    visualize_trading_pairs(trading_pairs, file_name.replace('.csv', ''))
    
    return trading_pairs, pattern_features, df

def main():
    """
    主函数
    """
    # 获取所有标签文件
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
    print(f"Found {len(label_files)} label files")
    
    # 存储所有结果
    all_trading_pairs = []
    all_pattern_features = []
    
    # 处理前10个文件作为示例
    for i, file_path in enumerate(label_files[:10]):
        try:
            trading_pairs, pattern_features, df = process_single_file(file_path)
            all_trading_pairs.extend(trading_pairs)
            all_pattern_features.extend(pattern_features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 计算总体统计信息
    pair_stats = calculate_pair_statistics(all_trading_pairs)
    
    # 保存分析结果
    save_analysis_results(pair_stats, all_pattern_features)
    
    print(f"\nAnalysis completed! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()