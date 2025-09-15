#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号预测分析和可视化生成器
对./data/目录下的所有CSV文件进行交易信号预测分析，并生成可视化图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# 导入交易信号预测器
from trading_signal_predictor import TradingSignalPredictor

class TradingAnalysisVisualizer:
    def __init__(self):
        self.predictor = TradingSignalPredictor()
        self.signal_colors = {
            1: 'green',    # 做多开仓 - 绿色向上箭头
            2: 'green',    # 做多平仓 - 绿色向下箭头
            3: 'red',      # 做空开仓 - 红色向下箭头
            4: 'red'       # 做空平仓 - 红色向上箭头
        }
        self.signal_names = {
            0: "等待",
            1: "做多开仓",
            2: "做多平仓", 
            3: "做空开仓",
            4: "做空平仓"
        }
        
    def create_result_directory(self):
        """创建结果目录"""
        result_dir = "./result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print(f"✓ 创建结果目录: {result_dir}")
        return result_dir
    
    def get_data_files(self):
        """获取data目录下的所有CSV文件"""
        data_dir = "./data"
        if not os.path.exists(data_dir):
            print(f"❌ 数据目录不存在: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"✓ 找到 {len(csv_files)} 个CSV文件")
        return csv_files
    
    def add_arrow_annotation(self, ax, x, y, signal_type, price_range):
        """添加箭头标注"""
        color = self.signal_colors[signal_type]
        arrow_height = price_range * 0.02  # 箭头高度为价格范围的2%
        
        if signal_type == 1:  # 做多开仓 - 绿色向上箭头
            ax.annotate('', xy=(x, y + arrow_height), xytext=(x, y - arrow_height),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
        elif signal_type == 2:  # 做多平仓 - 绿色向下箭头
            ax.annotate('', xy=(x, y - arrow_height), xytext=(x, y + arrow_height),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
        elif signal_type == 3:  # 做空开仓 - 红色向下箭头
            ax.annotate('', xy=(x, y - arrow_height), xytext=(x, y + arrow_height),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
        elif signal_type == 4:  # 做空平仓 - 红色向上箭头
            ax.annotate('', xy=(x, y + arrow_height), xytext=(x, y - arrow_height),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8))
    
    def filter_trading_signals(self, predictions, prices, confidences, max_daily_trades=3):
        """过滤交易信号，实现持仓状态管理，限制每日开仓次数，并计算盈利"""
        # 首先选择置信度最高的开仓信号
        selected_signals = self.select_best_signals(predictions, confidences, max_daily_trades)
        
        filtered_predictions = []
        position_state = 0  # 0: 空仓, 1: 多仓, -1: 空仓
        open_price = 0  # 开仓价格
        trades = []  # 存储完整的交易记录
        current_trade = None  # 当前交易
        daily_trade_count = 0  # 当日开仓次数
        
        for i, pred in enumerate(selected_signals):
            filtered_pred = pred
            current_price = prices[i]
            
            if pred == 1:  # 做多开仓信号
                if position_state == -1:  # 当前持空仓，不允许做多开仓
                    filtered_pred = 0
                elif position_state == 1:  # 当前已持多仓，忽略重复开仓
                    filtered_pred = 0
                elif daily_trade_count >= max_daily_trades:  # 超过每日开仓限制
                    filtered_pred = 0
                else:  # 空仓状态，允许做多开仓
                    position_state = 1
                    open_price = current_price
                    daily_trade_count += 1
                    current_trade = {
                        'type': 'long',
                        'open_index': i,
                        'open_price': open_price,
                        'close_index': None,
                        'close_price': None,
                        'profit': None,
                        'confidence': confidences[i]
                    }
                    
            elif pred == 2:  # 做多平仓信号
                if position_state == 1:  # 当前持多仓，允许平仓
                    position_state = 0
                    profit = current_price - open_price
                    if current_trade:
                        current_trade.update({
                            'close_index': i,
                            'close_price': current_price,
                            'profit': profit
                        })
                        trades.append(current_trade)
                        current_trade = None
                else:  # 非多仓状态，忽略平仓信号
                    filtered_pred = 0
                    
            elif pred == 3:  # 做空开仓信号
                if position_state == 1:  # 当前持多仓，不允许做空开仓
                    filtered_pred = 0
                elif position_state == -1:  # 当前已持空仓，忽略重复开仓
                    filtered_pred = 0
                elif daily_trade_count >= max_daily_trades:  # 超过每日开仓限制
                    filtered_pred = 0
                else:  # 空仓状态，允许做空开仓
                    position_state = -1
                    open_price = current_price
                    daily_trade_count += 1
                    current_trade = {
                        'type': 'short',
                        'open_index': i,
                        'open_price': open_price,
                        'close_index': None,
                        'close_price': None,
                        'profit': None,
                        'confidence': confidences[i]
                    }
                    
            elif pred == 4:  # 做空平仓信号
                if position_state == -1:  # 当前持空仓，允许平仓
                    position_state = 0
                    profit = open_price - current_price  # 做空盈利 = 开仓价 - 平仓价
                    if current_trade:
                        current_trade.update({
                            'close_index': i,
                            'close_price': current_price,
                            'profit': profit
                        })
                        trades.append(current_trade)
                        current_trade = None
                else:  # 非空仓状态，忽略平仓信号
                    filtered_pred = 0
            
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions, trades
    
    def select_best_signals(self, predictions, confidences, max_daily_trades):
        """选择置信度最高的开仓信号，限制每日开仓次数"""
        selected_predictions = predictions.copy()
        
        # 找出所有开仓信号的位置和置信度
        open_signals = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if pred in [1, 3]:  # 做多开仓或做空开仓
                open_signals.append({
                    'index': i,
                    'signal': pred,
                    'confidence': conf
                })
        
        # 按置信度降序排序
        open_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 只保留置信度最高的max_daily_trades个开仓信号
        if len(open_signals) > max_daily_trades:
            # 将超出限制的开仓信号设为0（等待）
            kept_indices = set(signal['index'] for signal in open_signals[:max_daily_trades])
            for i, pred in enumerate(selected_predictions):
                if pred in [1, 3] and i not in kept_indices:
                    selected_predictions[i] = 0
        
        return selected_predictions
    
    def analyze_and_visualize_file(self, filename, result_dir):
        """分析单个文件并生成可视化"""
        try:
            # 读取数据
            file_path = os.path.join("./data", filename)
            df = pd.read_csv(file_path)
            
            print(f"\n=== 分析文件: {filename} ===")
            print(f"数据行数: {len(df)}")
            
            # 检查必要的列
            required_columns = ['index_value']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ 缺少必要列: {missing_columns}")
                return False
            
            # 进行交易信号预测
            raw_predictions, confidences = self.predictor.batch_predict(df, min_confidence=0.6)
            
            # 过滤交易信号并计算盈利（限制每日最多3次开仓）
            predictions, trades = self.filter_trading_signals(raw_predictions, df['index_value'].values, confidences, max_daily_trades=3)
            
            # 统计原始预测结果
            raw_signal_counts = {}
            for pred in raw_predictions:
                raw_signal_counts[pred] = raw_signal_counts.get(pred, 0) + 1
            
            # 统计过滤后预测结果
            signal_counts = {}
            for pred in predictions:
                signal_counts[pred] = signal_counts.get(pred, 0) + 1
            
            print("原始预测信号统计:")
            for signal, count in sorted(raw_signal_counts.items()):
                percentage = count / len(raw_predictions) * 100
                print(f"  {self.signal_names.get(signal, f'未知({signal})')}: {count} 次 ({percentage:.2f}%)")
            
            print("\n过滤后预测信号统计:")
            for signal, count in sorted(signal_counts.items()):
                percentage = count / len(predictions) * 100
                print(f"  {self.signal_names.get(signal, f'未知({signal})')}: {count} 次 ({percentage:.2f}%)")
            
            # 统计交易盈利情况
            if trades:
                total_profit = sum(trade['profit'] for trade in trades)
                profitable_trades = [t for t in trades if t['profit'] > 0]
                losing_trades = [t for t in trades if t['profit'] < 0]
                avg_confidence = sum(trade['confidence'] for trade in trades) / len(trades)
                
                print(f"\n交易盈利统计 (限制每日最多3次开仓):")
                print(f"  总交易次数: {len(trades)}")
                print(f"  盈利交易: {len(profitable_trades)} 次")
                print(f"  亏损交易: {len(losing_trades)} 次")
                print(f"  胜率: {len(profitable_trades)/len(trades)*100:.2f}%")
                print(f"  总盈亏: {total_profit:.2f}")
                print(f"  平均每笔盈亏: {total_profit/len(trades):.2f}")
                print(f"  平均置信度: {avg_confidence:.3f}")
                
                if profitable_trades:
                    avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades)
                    avg_profit_conf = sum(t['confidence'] for t in profitable_trades) / len(profitable_trades)
                    print(f"  平均盈利: {avg_profit:.2f} (平均置信度: {avg_profit_conf:.3f})")
                if losing_trades:
                    avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades)
                    avg_loss_conf = sum(t['confidence'] for t in losing_trades) / len(losing_trades)
                    print(f"  平均亏损: {avg_loss:.2f} (平均置信度: {avg_loss_conf:.3f})")
                    
                # 显示每笔交易的详细信息
                print(f"\n详细交易记录:")
                for i, trade in enumerate(trades, 1):
                    trade_type = "做多" if trade['type'] == 'long' else "做空"
                    print(f"  交易{i}: {trade_type} | 开仓:{trade['open_price']:.2f} -> 平仓:{trade['close_price']:.2f} | 盈亏:{trade['profit']:+.2f} | 置信度:{trade['confidence']:.3f}")
            else:
                print("\n交易盈利统计: 无完整交易记录")
            
            # 创建可视化图表
            plt.figure(figsize=(15, 8))
            
            # 创建x轴（如果没有x列，使用索引）
            if 'x' in df.columns:
                x_values = df['x'].values
            else:
                x_values = np.arange(len(df))
            
            y_values = df['index_value'].values
            
            # 绘制价格曲线
            plt.plot(x_values, y_values, 'b-', linewidth=1, alpha=0.7, label='指数价格')
            
            # 计算价格范围用于箭头大小
            price_range = np.max(y_values) - np.min(y_values)
            
            # 添加交易信号箭头（使用过滤后的信号）
            trading_signals = 0
            for i, pred in enumerate(predictions):
                if pred != 0:  # 非等待信号
                    self.add_arrow_annotation(plt.gca(), x_values[i], y_values[i], pred, price_range)
                    trading_signals += 1
            
            # 添加交易盈亏标注
            for trade in trades:
                if trade['close_index'] is not None:  # 完整交易
                    open_x = x_values[trade['open_index']]
                    close_x = x_values[trade['close_index']]
                    open_y = trade['open_price']
                    close_y = trade['close_price']
                    profit = trade['profit']
                    
                    # 在交易区间中点添加盈亏标注
                    mid_x = (open_x + close_x) / 2
                    mid_y = (open_y + close_y) / 2
                    
                    # 根据盈亏设置颜色
                    profit_color = 'green' if profit > 0 else 'red'
                    profit_text = f'+{profit:.1f}' if profit > 0 else f'{profit:.1f}'
                    
                    plt.annotate(profit_text, (mid_x, mid_y), 
                               xytext=(0, 20), textcoords='offset points',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=profit_color, alpha=0.7),
                               fontsize=8, color='white', weight='bold')
            
            # 计算总盈亏用于标题显示
            total_profit = sum(trade['profit'] for trade in trades) if trades else 0
            profit_text = f" | 总盈亏: {total_profit:+.1f}" if trades else ""
            
            # 设置图表属性
            plt.title(f'交易信号分析 - {filename}\n(共{trading_signals}个交易信号, {len(trades)}笔完整交易{profit_text})', fontsize=14, pad=20)
            plt.xlabel('时间序列 (x)', fontsize=12)
            plt.ylabel('指数价值 (index_value)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加信号说明
            legend_elements = [
                plt.Line2D([0], [0], marker='^', color='green', linestyle='None', 
                          markersize=8, label='做多开仓'),
                plt.Line2D([0], [0], marker='v', color='green', linestyle='None', 
                          markersize=8, label='做多平仓'),
                plt.Line2D([0], [0], marker='v', color='red', linestyle='None', 
                          markersize=8, label='做空开仓'),
                plt.Line2D([0], [0], marker='^', color='red', linestyle='None', 
                          markersize=8, label='做空平仓')
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.95))
            
            # 保存图片
            output_filename = filename.replace('.csv', '.png')
            output_path = os.path.join(result_dir, output_filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 验证文件是否真的被创建
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✓ 图表已保存: {output_path} (大小: {file_size} bytes)")
                else:
                    print(f"❌ 图表保存失败: {output_path}")
                    return False
                    
            except Exception as save_error:
                print(f"❌ 保存图表时出错: {str(save_error)}")
                plt.close()
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {str(e)}")
            return False
    
    def run_analysis(self):
        """运行完整的分析流程"""
        print("=" * 60)
        print("交易信号预测分析和可视化生成器")
        print("=" * 60)
        
        # 创建结果目录
        result_dir = self.create_result_directory()
        
        # 获取数据文件
        csv_files = self.get_data_files()
        if not csv_files:
            print("❌ 没有找到CSV文件")
            return
        
        # 处理每个文件
        success_count = 0
        for filename in csv_files:
            if self.analyze_and_visualize_file(filename, result_dir):
                success_count += 1
        
        print(f"\n=" * 60)
        print(f"分析完成: {success_count}/{len(csv_files)} 个文件处理成功")
        print(f"结果保存在: {result_dir}")
        print("=" * 60)

def main():
    """主函数"""
    visualizer = TradingAnalysisVisualizer()
    visualizer.run_analysis()

if __name__ == "__main__":
    main()