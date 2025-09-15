# -*- coding: utf-8 -*-
"""
基于Transformer的交易信号预测模型
目标：每个交易日(单独的csv文件)至少有一个开仓和一个相应平仓交易信号
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging

# 添加学习率调度器导入
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加TA-Lib库用于技术指标计算
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("TA-Lib not available. Technical indicators will not be calculated.")

class TransformerPredictor(nn.Module):
    """
    基于Transformer的交易信号预测模型
    """
    def __init__(self, input_dim=6, d_model=64, nhead=8, num_layers=4, num_classes=5, dropout=0.1, max_seq_len=2000):
        """
        初始化Transformer预测模型
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 多头注意力头数
            num_layers: Transformer层数
            num_classes: 输出类别数(0,1,2,3,4)
            dropout: Dropout率
            max_seq_len: 最大序列长度
        """
        super(TransformerPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim)
            
        Returns:
            output: 预测结果 (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # 检查序列长度
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}")
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

class TradingSignalDataset(Dataset):
    """
    交易信号数据集
    """
    def __init__(self, data_files, window_size=50, stride=10):
        """
        初始化数据集
        
        Args:
            data_files: 数据文件列表
            window_size: 窗口大小
            stride: 步长
        """
        self.window_size = window_size
        self.stride = stride
        self.data = []
        self.labels = []
        self.scaler = StandardScaler()
        
        # 加载数据
        self._load_data(data_files)
        
        # 标准化特征
        self._normalize_features()
    
    def _calculate_technical_indicators(self, df):
        """
        计算技术指标
        
        Args:
            df: DataFrame，包含原始数据
            
        Returns:
            features: 包含技术指标的特征数组
        """
        # 如果TA-Lib不可用，返回原始特征
        if not TALIB_AVAILABLE:
            # 返回原始6维特征
            return df[['x', 'a', 'b', 'c', 'd', 'index_value']].values
        
        try:
            # 计算各种技术指标
            open_prices = df['index_value'].values
            high_prices = df['index_value'].values  # 简化处理，实际应用中应使用真实高低价格
            low_prices = df['index_value'].values
            close_prices = df['index_value'].values
            volume = np.ones(len(df))  # 简化处理，实际应用中应使用真实成交量
            
            # 初始化技术指标数组
            sma_5 = close_prices.copy()
            sma_10 = close_prices.copy()
            sma_20 = close_prices.copy()
            macd = close_prices.copy()
            macd_signal = close_prices.copy()
            macd_hist = close_prices.copy()
            rsi = close_prices.copy()
            upper = close_prices.copy()
            middle = close_prices.copy()
            lower = close_prices.copy()
            slowk = close_prices.copy()
            slowd = close_prices.copy()
            
            # 只有当TA-Lib可用时才计算技术指标
            if TALIB_AVAILABLE:
                try:
                    # 移动平均线
                    sma_5 = talib.SMA(close_prices, timeperiod=5)
                    sma_10 = talib.SMA(close_prices, timeperiod=10)
                    sma_20 = talib.SMA(close_prices, timeperiod=20)
                    
                    # MACD
                    macd, macd_signal, macd_hist = talib.MACD(close_prices)
                    
                    # RSI
                    rsi = talib.RSI(close_prices)
                    
                    # 布林带
                    upper, middle, lower = talib.BBANDS(close_prices)
                    
                    # 随机指标
                    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
                except Exception as e:
                    logger.warning(f"Error calculating technical indicators with TA-Lib: {e}")
            
            # 原始特征
            original_features = df[['x', 'a', 'b', 'c', 'd', 'index_value']].values
            
            # 合并所有特征（如果计算了技术指标，则特征维度会增加）
            if TALIB_AVAILABLE:
                # 合并所有技术指标
                technical_features = np.column_stack([
                    sma_5, sma_10, sma_20,
                    macd, macd_signal, macd_hist,
                    rsi,
                    upper, middle, lower,
                    slowk, slowd
                ])
                
                # 合并原始特征和技术指标
                all_features = np.concatenate([original_features, technical_features], axis=1)
            else:
                # 如果没有计算技术指标，只使用原始特征
                all_features = original_features
            
            return all_features
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}. Using original features.")
            return df[['x', 'a', 'b', 'c', 'd', 'index_value']].values
    
    def _load_data(self, data_files):
        """加载数据"""
        logger.info(f"Loading data from {len(data_files)} files...")
        
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                
                # 提取特征（包括技术指标）
                features = self._calculate_technical_indicators(df)
                labels = df['label'].values
                
                # 滑动窗口处理
                for i in range(0, len(features) - self.window_size + 1, self.stride):
                    window_features = features[i:i + self.window_size]
                    window_labels = labels[i:i + self.window_size]
                    
                    # 只使用包含交易信号的窗口
                    if any(label in [1, 2, 3, 4] for label in window_labels):
                        self.data.append(window_features)
                        self.labels.append(window_labels)
                        all_features.append(window_features)
                        all_labels.extend(window_labels)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.data)} windows with trading signals")
        if all_labels:
            label_counts = pd.Series(all_labels).value_counts().sort_index()
            logger.info(f"Label distribution: {dict(label_counts)}")
    
    def _normalize_features(self):
        """标准化特征"""
        if not self.data:
            return
            
        # 合并所有特征进行标准化
        all_features = np.vstack(self.data)
        self.scaler.fit(all_features)
        
        # 标准化每个窗口
        for i in range(len(self.data)):
            self.data[i] = self.scaler.transform(self.data[i])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        labels = torch.LongTensor(self.labels[idx])
        return features, labels

class TradingSignalTransformer:
    """
    交易信号Transformer预测器
    """
    def __init__(self, input_dim=6, d_model=64, nhead=8, num_layers=4, num_classes=5, max_seq_len=2000):
        """
        初始化预测器
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 多头注意力头数
            num_layers: Transformer层数
            num_classes: 输出类别数
            max_seq_len: 最大序列长度
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 创建模型
        self.model = TransformerPredictor(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            max_seq_len=max_seq_len
        ).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 添加学习率调度器
        self.scheduler = None
        
        # 标准化器
        self.scaler = StandardScaler()
        
        logger.info("TradingSignalTransformer initialized")
    
    def train(self, train_files, val_files=None, epochs=50, batch_size=32, use_scheduler=True):
        """
        训练模型
        
        Args:
            train_files: 训练文件列表
            val_files: 验证文件列表
            epochs: 训练轮数
            batch_size: 批次大小
            use_scheduler: 是否使用学习率调度器
        """
        logger.info("Starting training...")
        
        # 创建数据集
        train_dataset = TradingSignalDataset(train_files)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_files:
            val_dataset = TradingSignalDataset(val_files)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化学习率调度器
        if use_scheduler:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_samples = 0
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(features)
                
                # 计算损失
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs}, Current Learning Rate: {current_lr:.6f}")
            
            avg_loss = total_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # 验证
            if val_loader:
                val_loss = self._validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
        
        logger.info("Training completed")
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
        
        self.model.train()
        return total_loss / total_samples
    
    def predict(self, df):
        """
        预测交易信号
        
        Args:
            df: DataFrame，包含测试数据
            
        Returns:
            predictions: 预测结果列表
            confidences: 置信度列表
        """
        self.model.eval()
        
        # 提取特征
        features = df[['x', 'a', 'b', 'c', 'd', 'index_value']].values
        
        # 检查序列长度
        if len(features) > self.model.max_seq_len:
            # 如果序列太长，只取最后max_seq_len个点
            logger.warning(f"Sequence length {len(features)} exceeds maximum {self.model.max_seq_len}, truncating to last {self.model.max_seq_len} points")
            features = features[-self.model.max_seq_len:]
        
        # 注意：在实际预测时，我们应该使用训练时的标准化参数
        # 这里为了简化，我们重新拟合标准化器，但在实际应用中应该保存和加载训练时的参数
        try:
            features = self.scaler.transform(features)
        except:
            # 如果标准化器未拟合，则重新拟合（仅用于测试）
            features = self.scaler.fit_transform(features)
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)
            
        # 转换为numpy数组
        predictions = predictions.cpu().numpy().flatten()
        confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy().flatten()
        
        self.model.train()
        return predictions.tolist(), confidences.tolist()
    
    def save_model(self, model_path, quantize=False):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            quantize: 是否进行模型量化
        """
        # 如果需要量化模型
        if quantize:
            try:
                # 创建量化模型副本
                import torch.quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
                
                model_data = {
                    'model_state_dict': quantized_model.state_dict(),
                    'scaler': self.scaler,
                    'quantized': True
                }
                logger.info("Model quantized to 8-bit precision")
            except Exception as e:
                logger.warning(f"Error quantizing model: {e}. Saving original model.")
                model_data = {
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'quantized': False
                }
        else:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'quantized': False
            }
        
        # 保存模型
        import torch
        torch.save(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            model_data = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.scaler = model_data['scaler']
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def train_transformer_model():
    """
    训练Transformer模型
    """
    logger.info("Training Transformer model...")
    
    # 获取标签文件
    label_dir = "label"
    label_files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir)) if f.endswith(".csv")]
    
    if len(label_files) < 10:
        logger.error("Not enough label files for training!")
        return
    
    # 分割训练和验证集
    train_files = label_files[:int(0.8 * len(label_files))]
    val_files = label_files[int(0.8 * len(label_files)):]
    
    logger.info(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # 创建预测器
    predictor = TradingSignalTransformer()
    
    # 训练模型
    predictor.train(train_files, val_files, epochs=30, batch_size=16)
    
    # 保存模型
    model_path = os.path.join("model", "balanced_model", "transformer_predictor.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    predictor.save_model(model_path)
    
    logger.info("Transformer model training completed")

def test_transformer_model():
    """
    测试Transformer模型
    """
    logger.info("Testing Transformer model...")
    
    # 获取标签文件
    label_dir = "label"
    label_files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir)) if f.endswith(".csv")]
    
    if not label_files:
        logger.error("No label files found!")
        return
    
    # 创建预测器
    predictor = TradingSignalTransformer()
    
    # 加载模型
    model_path = os.path.join("model", "balanced_model", "transformer_predictor.pth")
    if not predictor.load_model(model_path):
        logger.error("Failed to load model!")
        return
    
    # 测试前几个文件
    for file_path in label_files[:3]:
        logger.info(f"Testing on {os.path.basename(file_path)}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # 进行预测
            predictions, confidences = predictor.predict(df)
            
            # 分析结果
            pred_series = pd.Series(predictions)
            pred_counts = pred_series.value_counts().sort_index()
            logger.info(f"  Prediction distribution: {dict(pred_counts)}")
            logger.info(f"  Average confidence: {np.mean(confidences):.4f}")
            
            # 检查是否有开仓和平仓信号
            long_open_signals = sum(1 for p in predictions if p == 1)
            long_close_signals = sum(1 for p in predictions if p == 2)
            short_open_signals = sum(1 for p in predictions if p == 3)
            short_close_signals = sum(1 for p in predictions if p == 4)
            
            logger.info(f"  Long open signals: {long_open_signals}")
            logger.info(f"  Long close signals: {long_close_signals}")
            logger.info(f"  Short open signals: {short_open_signals}")
            logger.info(f"  Short close signals: {short_close_signals}")
            
        except Exception as e:
            logger.error(f"Error testing {file_path}: {e}")

if __name__ == "__main__":
    # 训练模型
    train_transformer_model()
    
    # 测试模型
    test_transformer_model()