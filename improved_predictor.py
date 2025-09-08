# -*- coding: utf-8 -*-
"""
改进的模式预测器设计
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class ImprovedPatternPredictor:
    """
    改进的模式预测器
    """
    def __init__(self):
        self.models = {}  # 为每种信号类型创建专门的模型
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, df, index):
        """
        提取更丰富的特征
        """
        if index < 20:  # 需要至少20个数据点来计算技术指标
            return None
            
        features = []
        
        # 基础特征
        features.extend([df.iloc[index]['x'], df.iloc[index]['a'], df.iloc[index]['b'], 
                        df.iloc[index]['c'], df.iloc[index]['d'], df.iloc[index]['index_value']])
        
        # 时间序列特征
        for i in range(1, 6):  # 前5个时间点
            if index - i >= 0:
                features.extend([df.iloc[index-i]['x'], df.iloc[index-i]['index_value']])
            else:
                features.extend([0, 0])  # 填充
        
        # 滑动窗口统计特征
        window_size = min(20, index)
        if window_size > 1:
            window_data = df.iloc[index-window_size:index]['index_value']
            features.extend([
                window_data.mean(),    # 均值
                window_data.std(),     # 标准差
                window_data.max(),     # 最大值
                window_data.min(),     # 最小值
                window_data.iloc[-1] - window_data.iloc[0],  # 趋势
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 技术指标特征
        if index >= 20:
            # 计算移动平均线
            ma_window = df.iloc[index-20:index]['index_value']
            ma20 = ma_window.mean()
            current_price = df.iloc[index]['index_value']
            features.append(current_price - ma20)  # 价格偏离MA20
            
            # 计算RSI
            gains = []
            losses = []
            for i in range(1, min(15, index)):
                change = df.iloc[index-i+1]['index_value'] - df.iloc[index-i]['index_value']
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi)
            
            # 计算MACD相关指标
            if index >= 26:
                # 12日EMA
                ema12_window = df.iloc[index-12:index]['index_value']
                ema12 = ema12_window.ewm(span=12).mean().iloc[-1]
                
                # 26日EMA
                ema26_window = df.iloc[index-26:index]['index_value']
                ema26 = ema26_window.ewm(span=26).mean().iloc[-1]
                
                # MACD线
                macd = ema12 - ema26
                features.append(macd)
                
                # 信号线(9日EMA of MACD)
                macd_series = []
                for j in range(9):
                    if index-j >= 26:
                        ema12_w = df.iloc[index-j-12:index-j]['index_value']
                        ema12_val = ema12_w.ewm(span=12).mean().iloc[-1]
                        ema26_w = df.iloc[index-j-26:index-j]['index_value']
                        ema26_val = ema26_w.ewm(span=26).mean().iloc[-1]
                        macd_series.append(ema12_val - ema26_val)
                    else:
                        macd_series.append(0)
                
                signal_line = np.mean(macd_series)
                features.append(signal_line)
                
                # MACD柱状图
                macd_histogram = macd - signal_line
                features.append(macd_histogram)
            else:
                features.extend([0, 0, 0])  # MACD相关指标填充
            
            # 布林带特征
            if index >= 20:
                bb_window = df.iloc[index-20:index]['index_value']
                bb_mean = bb_window.mean()
                bb_std = bb_window.std()
                bb_upper = bb_mean + 2 * bb_std
                bb_lower = bb_mean - 2 * bb_std
                current_price = df.iloc[index]['index_value']
                
                # 价格相对于布林带的位置
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-10)
                features.append(bb_position)
                
                # 是否突破布林带
                bb_upper_break = 1 if current_price > bb_upper else 0
                bb_lower_break = 1 if current_price < bb_lower else 0
                features.extend([bb_upper_break, bb_lower_break])
            else:
                features.extend([0, 0, 0])  # 布林带特征填充
        else:
            # 如果没有足够的数据计算技术指标，填充默认值
            features.extend([0, 50])  # MA偏离和RSI默认值
            features.extend([0, 0, 0])  # MACD相关指标默认值
            features.extend([0, 0, 0])  # 布林带特征默认值
        
        return np.array(features)
    
    def prepare_training_data(self, label_files):
        """
        准备训练数据
        """
        print("Preparing training data...")
        
        X_list = []
        y_list = []
        
        for file_path in label_files[:20]:  # 使用前20个文件训练
            try:
                df = pd.read_csv(file_path)
                for i in range(len(df)):
                    label = df.iloc[i]['label']
                    # 只使用交易信号进行训练（排除无操作信号0）
                    if label in [1, 2, 3, 4]:  
                        features = self.extract_features(df, i)
                        if features is not None:
                            X_list.append(features)
                            y_list.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if len(X_list) == 0:
            print("No training data prepared!")
            return None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Prepared {len(X)} training samples")
        print(f"Label distribution: {pd.Series(y).value_counts().sort_index()}")
        
        return X, y
    
    def train(self, label_files):
        """
        训练改进的预测模型
        """
        print("Training improved pattern predictor...")
        
        # 准备训练数据（只使用交易信号）
        X, y = self.prepare_training_data(label_files)
        if X is None:
            return False
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 为每种信号类型训练专门的模型
        signal_types = [1, 2, 3, 4]  # 只为交易信号训练模型
        
        for signal in signal_types:
            print(f"Training model for signal {signal}...")
            
            # 创建二分类标签
            y_binary = (y == signal).astype(int)
            
            # 训练随机森林模型
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # 处理类别不平衡
            )
            
            model.fit(X_scaled, y_binary)
            self.models[signal] = model
        
        self.is_trained = True
        print("Training completed!")
        return True
    
    def predict(self, df, index):
        """
        预测信号
        """
        if not self.is_trained:
            return 0, 0.0
        
        # 提取特征
        features = self.extract_features(df, index)
        if features is None:
            return 0, 0.0
        
        # 标准化特征
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 获取所有模型的预测概率
        probabilities = {}
        for signal, model in self.models.items():
            prob = model.predict_proba(features_scaled)[0][1]  # 正类概率
            probabilities[signal] = prob
        
        # 选择概率最高的信号
        if probabilities:  # 如果有预测概率
            predicted_signal = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_signal]
            
            # 设置合理的置信度阈值
            if confidence < 0.3:  # 置信度阈值
                predicted_signal = 0  # 返回无操作信号
                confidence = 0.0
        else:
            predicted_signal = 0
            confidence = 0.0
        
        return predicted_signal, confidence
    
    def save_model(self, model_path):
        """
        保存模型
        """
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path):
        """
        加载模型
        """
        try:
            model_data = joblib.load(model_path)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def test_improved_predictor():
    """
    测试改进的预测器
    """
    print("Testing improved pattern predictor...")
    
    # 创建预测器
    predictor = ImprovedPatternPredictor()
    
    # 获取标签文件
    label_files = [os.path.join("label", f) for f in sorted(os.listdir("label")) if f.endswith(".csv")]
    if not label_files:
        print("No label files found!")
        return
    
    # 训练模型
    print("Training model...")
    if predictor.train(label_files):
        # 保存模型
        model_path = os.path.join("model", "balanced_model", "improved_predictor.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        predictor.save_model(model_path)
        
        # 测试预测
        test_file = label_files[0]
        print(f"Testing on {test_file}...")
        
        df = pd.read_csv(test_file)
        predictions = []
        confidences = []
        
        # 为最后50个点生成预测
        start_idx = max(0, len(df) - 50)
        for i in range(start_idx, len(df)):
            signal, confidence = predictor.predict(df, i)
            predictions.append(signal)
            confidences.append(confidence)
        
        # 分析结果
        signal_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"Prediction distribution:")
        for signal, count in signal_counts.items():
            print(f"  Signal {signal}: {count} times")
        
        print(f"Average confidence: {np.mean(confidences):.4f}")
        
    else:
        print("Failed to train model!")

if __name__ == "__main__":
    test_improved_predictor()