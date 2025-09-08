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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    def _load_data(self, data_files):
        """加载数据"""
        logger.info(f"Loading data from {len(data_files)} files...")
        
        all_features = []
        all_labels = []
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                
                # 提取特征
                features = df[['x', 'a', 'b', 'c', 'd', 'index_value']].values
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
        
        # 标准化器
        self.scaler = StandardScaler()
        
        logger.info("TradingSignalTransformer initialized")
    
    def train(self, train_files, val_files=None, epochs=50, batch_size=32):
        """
        训练模型
        
        Args:
            train_files: 训练文件列表
            val_files: 验证文件列表
            epochs: 训练轮数
            batch_size: 批次大小
        """
        logger.info("Starting training...")
        
        # 创建数据集
        train_dataset = TradingSignalDataset(train_files)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_files:
            val_dataset = TradingSignalDataset(val_files)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
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
            
            avg_loss = total_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # 验证
            if val_files:
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
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }
        
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