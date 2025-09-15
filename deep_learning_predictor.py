# -*- coding: utf-8 -*-
"""
深度学习预测器
专为股指期货短期剧烈波动设计
使用Transformer + CNN架构
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import talib
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch未安装，将使用轻量级神经网络实现")
    TORCH_AVAILABLE = False

class VolatilityDataset(Dataset):
    """
    专为波动性数据设计的数据集
    """
    def __init__(self, features, labels, sequence_length=20):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.LongTensor([y])

class VolatilityTransformer(nn.Module):
    """
    专为股指期货波动性设计的Transformer模型
    """
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, num_classes=4):
        super(VolatilityTransformer, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CNN层用于捕捉局部模式
        self.conv1d = nn.Conv1d(d_model, 128, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm1d(128)
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # CNN处理
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = torch.relu(self.conv_bn(self.conv1d(x)))
        x = x.transpose(1, 2)  # (batch, seq_len, 128)
        
        # 注意力池化
        attn_output, _ = self.attention_pool(x, x, x)
        
        # 全局平均池化
        x = torch.mean(attn_output, dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output

class LightweightNN:
    """
    轻量级神经网络实现（不依赖PyTorch）
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=4):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.weights = []
        self.biases = []
        
        # 初始化权重
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * 0.1
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            if i < len(self.weights) - 1:  # 不在最后一层使用激活函数
                x = self.relu(x)
        return self.softmax(x)
    
    def train_simple(self, X, y, epochs=100, lr=0.01):
        """
        简单的训练过程
        """
        for epoch in range(epochs):
            # 前向传播
            predictions = self.forward(X)
            
            # 计算损失（交叉熵）
            y_onehot = np.eye(self.num_classes)[y]
            loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-8), axis=1))
            
            # 简单的梯度下降（仅更新最后一层）
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            # 反向传播（简化版）
            grad = predictions - y_onehot
            self.weights[-1] -= lr * np.dot(X.T, grad) / len(X)
            self.biases[-1] -= lr * np.mean(grad, axis=0)

class DeepLearningPredictor:
    def __init__(self):
        self.models_dir = "./models_deep/"
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
    def extract_advanced_features(self, df, window_size=20):
        """
        提取高级特征用于深度学习
        """
        features = []
        labels = []
        
        for i in range(window_size, len(df) - 1):
            window_data = df.iloc[i-window_size:i]
            prices = window_data['index_value'].values.astype(float)
            
            try:
                # 基础技术指标
                sma_5 = talib.SMA(prices, timeperiod=5)
                sma_10 = talib.SMA(prices, timeperiod=10)
                sma_20 = talib.SMA(prices, timeperiod=20)
                ema_5 = talib.EMA(prices, timeperiod=5)
                ema_10 = talib.EMA(prices, timeperiod=10)
                
                # 动量指标
                rsi = talib.RSI(prices, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(prices)
                
                # 波动率指标
                atr = talib.ATR(prices, prices, prices, timeperiod=14)
                
                # 价格变化率
                roc_5 = talib.ROC(prices, timeperiod=5)
                roc_10 = talib.ROC(prices, timeperiod=10)
                
                # 布林带
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                
                # 威廉指标
                willr = talib.WILLR(prices, prices, prices, timeperiod=14)
                
                # 随机指标
                slowk, slowd = talib.STOCH(prices, prices, prices)
                
                # 价格位置特征
                current_price = prices[-1]
                price_position = (current_price - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
                
                # 波动性特征
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                skewness = self.calculate_skewness(returns)
                kurtosis = self.calculate_kurtosis(returns)
                
                # 趋势强度
                trend_strength = abs(prices[-1] - prices[0]) / (np.std(prices) + 1e-8)
                
                # 组合特征向量
                feature_vector = [
                    # 价格相关
                    current_price / np.mean(prices),
                    price_position,
                    
                    # 移动平均
                    sma_5[-1] / current_price if not np.isnan(sma_5[-1]) else 1,
                    sma_10[-1] / current_price if not np.isnan(sma_10[-1]) else 1,
                    sma_20[-1] / current_price if not np.isnan(sma_20[-1]) else 1,
                    ema_5[-1] / current_price if not np.isnan(ema_5[-1]) else 1,
                    ema_10[-1] / current_price if not np.isnan(ema_10[-1]) else 1,
                    
                    # 动量指标
                    rsi[-1] / 100 if not np.isnan(rsi[-1]) else 0.5,
                    macd[-1] if not np.isnan(macd[-1]) else 0,
                    macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                    macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                    
                    # 波动率
                    volatility,
                    atr[-1] / current_price if not np.isnan(atr[-1]) else 0,
                    
                    # 变化率
                    roc_5[-1] / 100 if not np.isnan(roc_5[-1]) else 0,
                    roc_10[-1] / 100 if not np.isnan(roc_10[-1]) else 0,
                    
                    # 布林带
                    (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-8) if not np.isnan(bb_upper[-1]) else 0.5,
                    
                    # 其他指标
                    willr[-1] / -100 if not np.isnan(willr[-1]) else 0.5,
                    slowk[-1] / 100 if not np.isnan(slowk[-1]) else 0.5,
                    slowd[-1] / 100 if not np.isnan(slowd[-1]) else 0.5,
                    
                    # 统计特征
                    skewness,
                    kurtosis,
                    trend_strength
                ]
                
                # 确保所有特征都是有效数值
                feature_vector = [f if not np.isnan(f) and not np.isinf(f) else 0 for f in feature_vector]
                
                features.append(feature_vector)
                
                # 标签
                next_label = df['label'].iloc[i + 1]
                if next_label in [1, 2, 3, 4]:
                    labels.append(next_label - 1)  # 转换为0-3
                else:
                    labels.append(0)  # 默认标签
                    
            except Exception as e:
                print(f"特征提取错误: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    def calculate_skewness(self, data):
        """计算偏度"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """计算峰度"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def prepare_training_data(self, data_files=None):
        """
        准备训练数据
        """
        if data_files is None:
            data_files = sorted(glob.glob("./label/*.csv"))[:10]  # 使用前10个文件训练
        
        all_features = []
        all_labels = []
        
        print(f"准备训练数据，使用 {len(data_files)} 个文件...")
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                features, labels = self.extract_advanced_features(df)
                
                if len(features) > 0:
                    all_features.extend(features)
                    all_labels.extend(labels)
                    print(f"文件 {os.path.basename(file_path)}: {len(features)} 个样本")
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        if len(all_features) == 0:
            raise ValueError("没有有效的训练数据")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n总训练样本: {len(X)}")
        print(f"特征维度: {X.shape[1]}")
        print(f"标签分布: {Counter(y)}")
        
        return X, y
    
    def train_model(self, X, y, epochs=50, batch_size=32):
        """
        训练深度学习模型
        """
        print("\n开始训练深度学习模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if TORCH_AVAILABLE:
            return self.train_pytorch_model(X_train, X_val, y_train, y_val, epochs, batch_size)
        else:
            return self.train_lightweight_model(X_train, y_train)
    
    def train_pytorch_model(self, X_train, X_val, y_train, y_val, epochs, batch_size):
        """
        使用PyTorch训练模型
        """
        print("使用PyTorch Transformer模型训练...")
        
        # 创建数据集
        train_dataset = VolatilityDataset(X_train, y_train)
        val_dataset = VolatilityDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_dim = X_train.shape[1]
        self.model = VolatilityTransformer(input_dim).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_y = batch_y.squeeze()
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_y = batch_y.squeeze()
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(self.models_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.models_dir, "best_model.pth"))
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
        return best_val_acc
    
    def train_lightweight_model(self, X_train, y_train):
        """
        训练轻量级模型
        """
        print("使用轻量级神经网络训练...")
        
        self.model = LightweightNN(X_train.shape[1])
        self.model.train_simple(X_train, y_train, epochs=100)
        
        # 计算训练准确率
        predictions = self.model.forward(X_train)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == y_train)
        
        print(f"\n训练完成！训练准确率: {accuracy:.4f}")
        return accuracy
    
    def predict_signals(self, test_files=None, n_files=3):
        """
        预测信号
        """
        if test_files is None:
            label_files = sorted(glob.glob("./label/*.csv"))
            test_files = label_files[-n_files:]
        
        print(f"\n开始深度学习预测，使用 {len(test_files)} 个文件...")
        
        all_predictions = []
        
        for file_path in test_files:
            try:
                df = pd.read_csv(file_path)
                features, actual_labels = self.extract_advanced_features(df)
                
                if len(features) == 0:
                    continue
                
                # 标准化特征
                features_scaled = self.scaler.transform(features)
                
                # 预测
                if TORCH_AVAILABLE and hasattr(self.model, 'eval'):
                    self.model.eval()
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                        outputs = self.model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                        predicted_labels = np.argmax(probabilities, axis=1)
                else:
                    probabilities = self.model.forward(features_scaled)
                    predicted_labels = np.argmax(probabilities, axis=1)
                
                # 记录预测结果
                for i, (pred, actual, prob) in enumerate(zip(predicted_labels, actual_labels, probabilities)):
                    confidence = np.max(prob)
                    is_correct = pred == actual
                    
                    prediction_detail = {
                        'file': os.path.basename(file_path),
                        'index': i,
                        'predicted': pred + 1,  # 转换回1-4
                        'actual': actual + 1,
                        'confidence': confidence,
                        'correct': is_correct
                    }
                    
                    all_predictions.append(prediction_detail)
                
                print(f"文件 {os.path.basename(file_path)}: {len(features)} 个预测")
                
            except Exception as e:
                print(f"预测文件 {file_path} 时出错: {e}")
                continue
        
        return all_predictions
    
    def evaluate_performance(self, predictions):
        """
        评估性能
        """
        if not predictions:
            print("没有预测结果可评估")
            return
        
        # 整体统计
        total_predictions = len(predictions)
        correct_predictions = sum(1 for p in predictions if p['correct'])
        overall_accuracy = correct_predictions / total_predictions
        
        # 信号类型统计
        signal_stats = {1: [], 2: [], 3: [], 4: []}
        for p in predictions:
            signal_stats[p['predicted']].append(p['correct'])
        
        print(f"\n=== 深度学习模型性能评估 ===")
        print(f"总预测数: {total_predictions}")
        print(f"整体准确率: {overall_accuracy:.4f}")
        
        signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
        signal_diversity = 0
        
        print("\n各信号类型表现:")
        for signal_type, results in signal_stats.items():
            if results:
                accuracy = sum(results) / len(results)
                print(f"  {signal_names[signal_type]}: {len(results)}次, 准确率{accuracy:.4f}")
                signal_diversity += 1
        
        print(f"\n信号多样性: {signal_diversity} 种")
        
        # 置信度分析
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences)
        print(f"平均置信度: {avg_confidence:.4f}")
        
        # 保存结果
        os.makedirs(self.models_dir, exist_ok=True)
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(os.path.join(self.models_dir, "deep_learning_results.csv"), index=False)
        
        return overall_accuracy, signal_diversity
    
    def run_complete_training(self):
        """
        运行完整的训练和评估流程
        """
        print("=== 深度学习预测器训练开始 ===")
        print(f"使用设备: {'GPU' if self.device and self.device.type == 'cuda' else 'CPU'}")
        
        try:
            # 准备数据
            X, y = self.prepare_training_data()
            
            # 训练模型
            train_accuracy = self.train_model(X, y)
            
            # 预测测试
            predictions = self.predict_signals()
            
            # 评估性能
            test_accuracy, diversity = self.evaluate_performance(predictions)
            
            print(f"\n=== 最终结果 ===")
            print(f"训练准确率: {train_accuracy:.4f}")
            print(f"测试准确率: {test_accuracy:.4f}")
            print(f"信号多样性: {diversity} 种")
            
            # 与之前结果对比
            print(f"\n=== 改进效果 ===")
            print(f"规则方法准确率: 33.3%")
            print(f"深度学习准确率: {test_accuracy:.1%}")
            
            if test_accuracy > 0.333:
                improvement = (test_accuracy - 0.333) / 0.333 * 100
                print(f"✅ 准确率提升: {improvement:.1f}%")
            else:
                print(f"⚠️  准确率需要进一步优化")
            
            return test_accuracy, diversity
            
        except Exception as e:
            print(f"训练过程出错: {e}")
            return 0, 0

def main():
    """
    主函数
    """
    predictor = DeepLearningPredictor()
    accuracy, diversity = predictor.run_complete_training()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 深度学习预测器总结:")
    print(f"   测试准确率: {accuracy:.4f}")
    print(f"   信号多样性: {diversity} 种")
    
    if accuracy > 0.4:
        print(f"\n✅ 深度学习方法成功提升了预测准确率！")
        print(f"   • 使用了Transformer + CNN架构")
        print(f"   • 专为股指期货波动性设计")
        print(f"   • 提取了22维高级特征")
    else:
        print(f"\n⚠️  深度学习方法需要进一步调优")
        print(f"   • 可能需要更多训练数据")
        print(f"   • 可以尝试不同的网络架构")
        print(f"   • 需要优化超参数")

if __name__ == "__main__":
    main()