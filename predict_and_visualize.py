# -*- coding: utf-8 -*-
"""
股指期货交易信号预测与可视化
对result目录中的CSV文件进行预测并生成可视化图表
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    import talib
    TORCH_AVAILABLE = True
except ImportError:
    print("缺少必要的库，请安装: pip install torch scikit-learn TA-Lib")
    TORCH_AVAILABLE = False

class ImprovedVolatilityNet(nn.Module):
    """
    改进的波动性预测网络（与训练时保持一致）
    """
    def __init__(self, input_dim, num_classes=4):
        super(ImprovedVolatilityNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class TradingSignalPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.signal_names = {
            1: '做多开仓',
            2: '做多平仓', 
            3: '做空开仓',
            4: '做空平仓'
        }
        self.signal_colors = {
            1: 'green',    # 做多开仓 - 绿色上三角
            2: 'green',    # 做多平仓 - 绿色下三角
            3: 'red',      # 做空开仓 - 红色下三角
            4: 'red'       # 做空平仓 - 红色上三角
        }
        self.signal_markers = {
            1: '^',        # 做多开仓 - 上三角
            2: 'v',        # 做多平仓 - 下三角
            3: 'v',        # 做空开仓 - 下三角
            4: '^'         # 做空平仓 - 上三角
        }
        
    def load_model(self):
        """
        加载训练好的模型
        """
        model_path = "./models_deep_fixed/best_model.pth"
        scaler_path = "./models_deep_fixed/scaler.pkl"
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("请先运行 fixed_deep_learning_predictor.py 训练模型")
            return False
            
        try:
            # 创建模型实例（需要知道输入维度）
            input_dim = 20  # 根据特征提取函数确定
            self.model = ImprovedVolatilityNet(input_dim).to(self.device)
            
            # 加载模型权重
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # 加载标准化器（如果存在）
            try:
                import pickle
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    print("警告: 未找到标准化器文件，使用默认标准化器")
            except:
                print("警告: 加载标准化器失败，使用默认标准化器")
            
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def extract_features(self, df):
        """
        提取特征（与训练时保持一致）
        """
        features = []
        indices = []
        
        if len(df) < 50:
            return np.array([]), []
        
        for i in range(30, len(df)):
            try:
                window_data = df.iloc[i-30:i]
                prices = window_data['index_value'].values.astype(float)
                
                if len(prices) < 30:
                    continue
                
                # 基础价格特征
                current_price = prices[-1]
                price_mean = np.mean(prices)
                price_std = np.std(prices)
                
                if price_std == 0:
                    continue
                
                # 标准化价格变化
                returns = np.diff(prices) / prices[:-1]
                
                # 技术指标
                sma_5 = talib.SMA(prices, timeperiod=5)
                sma_10 = talib.SMA(prices, timeperiod=10)
                sma_20 = talib.SMA(prices, timeperiod=20)
                
                rsi = talib.RSI(prices, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(prices)
                
                # 波动率指标
                volatility_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
                volatility_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
                volatility_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
                
                # 动量指标
                momentum_5 = (current_price - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
                momentum_10 = (current_price - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
                
                # 趋势指标
                trend_5 = 1 if sma_5[-1] > sma_5[-2] else 0 if not np.isnan(sma_5[-1]) and not np.isnan(sma_5[-2]) else 0.5
                trend_10 = 1 if sma_10[-1] > sma_10[-2] else 0 if not np.isnan(sma_10[-1]) and not np.isnan(sma_10[-2]) else 0.5
                
                # 相对位置
                price_position = (current_price - np.min(prices)) / (np.max(prices) - np.min(prices))
                
                # 组合特征
                feature_vector = [
                    (current_price - price_mean) / price_std,
                    price_position,
                    current_price / sma_5[-1] - 1 if not np.isnan(sma_5[-1]) and sma_5[-1] > 0 else 0,
                    current_price / sma_10[-1] - 1 if not np.isnan(sma_10[-1]) and sma_10[-1] > 0 else 0,
                    current_price / sma_20[-1] - 1 if not np.isnan(sma_20[-1]) and sma_20[-1] > 0 else 0,
                    (rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0,
                    macd[-1] / price_std if not np.isnan(macd[-1]) else 0,
                    macd_signal[-1] / price_std if not np.isnan(macd_signal[-1]) else 0,
                    macd_hist[-1] / price_std if not np.isnan(macd_hist[-1]) else 0,
                    volatility_5,
                    volatility_10,
                    volatility_20,
                    momentum_5,
                    momentum_10,
                    trend_5,
                    trend_10,
                    np.mean(returns[-5:]) if len(returns) >= 5 else 0,
                    np.mean(returns[-10:]) if len(returns) >= 10 else 0,
                    1 if current_price == np.max(prices[-10:]) else 0,
                    1 if current_price == np.min(prices[-10:]) else 0
                ]
                
                # 检查特征有效性
                if any(np.isnan(f) or np.isinf(f) for f in feature_vector):
                    continue
                
                features.append(feature_vector)
                indices.append(i)
                
            except Exception as e:
                continue
        
        return np.array(features), indices
    
    def filter_trading_signals(self, raw_signals, confidence_threshold=0.7, max_daily_trades=3):
        """
        过滤交易信号，应用交易逻辑约束和质量控制
        约束规则：
        1. 在一个方向开仓后，必须等该方向平仓才能出现反方向开仓信号
        2. 同一方向的重复开仓信号只保留第一个
        3. 只保留高置信度信号（置信度 >= confidence_threshold）
        4. 每日最多允许max_daily_trades笔开仓交易
        """
        if not raw_signals:
            return []
        
        # 首先按置信度过滤
        high_confidence_signals = []
        for signal in raw_signals:
            if signal['confidence'] >= confidence_threshold:
                high_confidence_signals.append(signal)
        
        print(f"置信度过滤: {len(raw_signals)} -> {len(high_confidence_signals)} (阈值: {confidence_threshold})")
        
        if not high_confidence_signals:
            return []
        
        filtered_signals = []
        position_state = 0  # 0: 无仓位, 1: 多头仓位, -1: 空头仓位
        daily_open_count = 0  # 当日开仓次数计数
        
        for signal in high_confidence_signals:
            label = signal['label']
            should_keep = False
            
            if label == 1:  # 做多开仓
                if position_state == 0 and daily_open_count < max_daily_trades:  # 无仓位且未超过每日限制
                    should_keep = True
                    position_state = 1
                    daily_open_count += 1
                # 如果已有多头仓位或空头仓位，或超过每日限制，忽略开仓信号
                
            elif label == 2:  # 做多平仓
                if position_state == 1:  # 有多头仓位时可以平仓
                    should_keep = True
                    position_state = 0
                # 如果无多头仓位，忽略平仓信号
                
            elif label == 3:  # 做空开仓
                if position_state == 0 and daily_open_count < max_daily_trades:  # 无仓位且未超过每日限制
                    should_keep = True
                    position_state = -1
                    daily_open_count += 1
                # 如果已有空头仓位或多头仓位，或超过每日限制，忽略开仓信号
                
            elif label == 4:  # 做空平仓
                if position_state == -1:  # 有空头仓位时可以平仓
                    should_keep = True
                    position_state = 0
                # 如果无空头仓位，忽略平仓信号
            
            if should_keep:
                filtered_signals.append(signal)
        
        print(f"交易逻辑过滤: {len(high_confidence_signals)} -> {len(filtered_signals)} (每日开仓限制: {max_daily_trades})")
        return filtered_signals
    
    def predict_signals(self, df):
        """
        预测交易信号（应用过滤逻辑）
        """
        features, indices = self.extract_features(df)
        
        if len(features) == 0:
            return [], [], []
        
        # 标准化特征
        try:
            features_scaled = self.scaler.transform(features)
        except:
            # 如果标准化器未正确加载，使用简单标准化
            features_scaled = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # 预测
        raw_predictions = []
        raw_confidences = []
        
        if self.model is not None:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predicted_labels = np.argmax(probabilities, axis=1)
                raw_confidences = np.max(probabilities, axis=1)
                raw_predictions = predicted_labels + 1  # 转换回1-4
        else:
            # 如果模型未加载，使用简单规则预测
            print("警告: 使用简单规则预测")
            for i, feature_vec in enumerate(features):
                # 基于价格趋势的简单规则
                momentum = feature_vec[12]  # momentum_5
                volatility = feature_vec[9]   # volatility_5
                
                if momentum > 0.01 and volatility < 0.02:
                    pred = 1  # 做多开仓
                elif momentum < -0.01 and volatility < 0.02:
                    pred = 3  # 做空开仓
                elif momentum > 0 and volatility > 0.03:
                    pred = 4  # 做空平仓
                else:
                    pred = 2  # 做多平仓
                
                raw_predictions.append(pred)
                raw_confidences.append(0.6)
        
        # 构建原始信号列表
        raw_signals = []
        for i, (pred, conf, idx) in enumerate(zip(raw_predictions, raw_confidences, indices)):
            raw_signals.append({
                'index': i,
                'label': pred,
                'confidence': conf,
                'data_index': idx
            })
        
        # 应用交易逻辑过滤（高置信度 + 每日交易限制）
        filtered_signals = self.filter_trading_signals(raw_signals, confidence_threshold=0.7, max_daily_trades=3)
        
        # 提取过滤后的结果
        predictions = [signal['label'] for signal in filtered_signals]
        confidences = [signal['confidence'] for signal in filtered_signals]
        filtered_indices = [signal['data_index'] for signal in filtered_signals]
        
        print(f"原始信号数: {len(raw_predictions)}, 过滤后信号数: {len(predictions)}")
        
        return predictions, confidences, filtered_indices
    
    def create_visualization(self, df, predictions, confidences, indices, filename):
        """
        创建可视化图表
        """
        plt.figure(figsize=(15, 8))
        
        # 绘制价格曲线
        x_values = df['x'].values
        prices = df['index_value'].values
        
        plt.plot(x_values, prices, 'b-', linewidth=1.5, label='股指价格', alpha=0.8)
        
        # 标记交易信号
        signal_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for pred, conf, idx in zip(predictions, confidences, indices):
            if idx < len(df):
                x_pos = df['x'].iloc[idx]
                y_pos = df['index_value'].iloc[idx]
                
                # 只显示高置信度的信号
                if conf > 0.5:
                    plt.scatter(x_pos, y_pos, 
                              c=self.signal_colors[pred], 
                              marker=self.signal_markers[pred], 
                              s=100, 
                              alpha=0.8,
                              edgecolors='black',
                              linewidth=0.5,
                              label=self.signal_names[pred] if signal_counts[pred] == 0 else "")
                    
                    # 添加置信度标注（可选）
                    if conf > 0.7:
                        plt.annotate(f'{conf:.2f}', 
                                   (x_pos, y_pos), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=8, 
                                   alpha=0.7)
                    
                    signal_counts[pred] += 1
        
        # 图表设置
        plt.title(f'股指期货交易信号预测 - {os.path.splitext(filename)[0]}', fontsize=16, fontweight='bold')
        plt.xlabel('时间序列', fontsize=12)
        plt.ylabel('股指价格', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=10)
        
        # 添加统计信息
        total_signals = sum(signal_counts.values())
        info_text = f'总信号数: {total_signals}\n'
        for signal_type, count in signal_counts.items():
            if count > 0:
                info_text += f'{self.signal_names[signal_type]}: {count}\n'
        
        plt.text(0.02, 0.98, info_text, 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        # 保存图片
        output_path = f"./result/{os.path.splitext(filename)[0]}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已生成: {output_path} (信号数: {total_signals})")
        return output_path, signal_counts
    
    def process_all_files(self):
        """
        处理所有CSV文件
        """
        # 确保result目录存在
        os.makedirs('./result', exist_ok=True)
        
        # 获取label目录下的所有CSV文件
        label_dir = './label'
        if not os.path.exists(label_dir):
            print(f"❌ 标签目录不存在: {label_dir}")
            return
            
        csv_files = glob.glob('./label/*.csv')
        
        if not csv_files:
            print(f"❌ 在{label_dir}目录中未找到CSV文件")
            return
        
        print(f"📁 在{label_dir}目录中找到 {len(csv_files)} 个CSV文件: {[os.path.basename(f) for f in csv_files]}")
        
        # 加载模型
        if not self.load_model():
            print("⚠️  模型加载失败，将使用简单规则预测")
        
        # 处理每个文件
        total_files = 0
        successful_files = 0
        all_signal_stats = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for csv_file in sorted(csv_files):
            try:
                filename = os.path.basename(csv_file)
                print(f"\n📊 处理文件: {filename}")
                
                # 读取数据
                df = pd.read_csv(csv_file)
                
                # 检查必要的列
                if 'x' not in df.columns or 'index_value' not in df.columns:
                    print(f"❌ 文件 {filename} 缺少必要的列 (x, index_value)")
                    continue
                
                # 预测信号
                predictions, confidences, indices = self.predict_signals(df)
                
                if len(predictions) == 0:
                    print(f"⚠️  文件 {filename} 无法生成预测信号")
                    continue
                
                # 创建可视化
                output_path, signal_counts = self.create_visualization(
                    df, predictions, confidences, indices, filename
                )
                
                # 统计
                for signal_type, count in signal_counts.items():
                    all_signal_stats[signal_type] += count
                
                successful_files += 1
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
                continue
            
            total_files += 1
        
        # 输出总结
        print(f"\n" + "=" * 60)
        print(f"🎯 处理完成！")
        print(f"   总文件数: {total_files}")
        print(f"   成功处理: {successful_files}")
        print(f"   生成图片: {successful_files} 张")
        
        print(f"\n📈 信号统计:")
        total_signals = sum(all_signal_stats.values())
        for signal_type, count in all_signal_stats.items():
            percentage = (count / total_signals * 100) if total_signals > 0 else 0
            print(f"   {self.signal_names[signal_type]}: {count} 次 ({percentage:.1f}%)")
        
        print(f"\n📁 所有图片已保存到 ./result/ 目录")

def main():
    print("=== 股指期货交易信号预测与可视化系统 ===")
    print("🚀 开始处理...")
    
    predictor = TradingSignalPredictor()
    predictor.process_all_files()
    
    print("\n✨ 全部完成！")

if __name__ == "__main__":
    main()