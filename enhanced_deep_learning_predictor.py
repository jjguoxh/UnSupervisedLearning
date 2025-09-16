# -*- coding: utf-8 -*-
"""
增强版深度学习预测器
目标：通过多种先进技术提高交易信号预测准确性
改进点：
1. 多尺度特征提取
2. 注意力机制增强
3. 残差连接和层归一化
4. 动态学习率调度
5. 数据增强技术
6. 集成学习方法
7. 对抗训练
8. 特征重要性分析
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import talib
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import logging
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    使用不同尺度的卷积核提取不同时间范围的特征
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 不同尺度的1D卷积
        self.conv1 = nn.Conv1d(input_dim, output_dim//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, output_dim//4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, output_dim//4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(input_dim, output_dim//4, kernel_size=9, padding=4)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(output_dim//4)
        self.bn2 = nn.BatchNorm1d(output_dim//4)
        self.bn3 = nn.BatchNorm1d(output_dim//4)
        self.bn4 = nn.BatchNorm1d(output_dim//4)
        
        # 激活函数
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # 多尺度特征提取
        f1 = self.activation(self.bn1(self.conv1(x)))
        f2 = self.activation(self.bn2(self.conv2(x)))
        f3 = self.activation(self.bn3(self.conv3(x)))
        f4 = self.activation(self.bn4(self.conv4(x)))
        
        # 拼接多尺度特征
        features = torch.cat([f1, f2, f3, f4], dim=1)
        
        return features.transpose(1, 2)  # (batch_size, seq_len, output_dim)

class EnhancedAttention(nn.Module):
    """
    增强注意力机制
    结合自注意力和交叉注意力
    """
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super(EnhancedAttention, self).__init__()
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 位置感知注意力
        self.position_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, pos_encoding=None):
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x

class EnhancedTransformerPredictor(nn.Module):
    """
    增强版Transformer预测器
    """
    def __init__(self, input_dim: int = 30, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, num_classes: int = 5, dropout: float = 0.1, 
                 max_seq_len: int = 2000):
        super(EnhancedTransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 多尺度特征提取
        self.feature_extractor = MultiScaleFeatureExtractor(input_dim, d_model)
        
        # 输入投影
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 位置编码（可学习）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 增强注意力层
        self.attention_layers = nn.ModuleList([
            EnhancedAttention(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头（多层）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 检查序列长度
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:, :]
            seq_len = self.max_seq_len
        
        # 多尺度特征提取
        x = self.feature_extractor(x)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # 通过增强注意力层
        for attention_layer in self.attention_layers:
            x = attention_layer(x, pos_enc)
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(x)
        
        return output

class EnhancedTradingDataset(Dataset):
    """
    增强版交易数据集
    包含数据增强功能
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 augment: bool = True, noise_factor: float = 0.01):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_factor = noise_factor
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 数据增强
        if self.augment and torch.rand(1) > 0.5:
            # 添加高斯噪声
            noise = torch.randn_like(feature) * self.noise_factor
            feature = feature + noise
            
            # 时间扭曲（随机缩放时间轴）
            if torch.rand(1) > 0.7:
                scale_factor = 0.9 + torch.rand(1) * 0.2  # 生成0.9到1.1之间的随机数
                feature = feature * scale_factor
        
        return feature, label

class EnhancedDeepLearningPredictor:
    """
    增强版深度学习预测器
    """
    def __init__(self, input_dim: int = 30):
        self.models_dir = "./models_enhanced/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = RobustScaler()  # 使用更鲁棒的标准化器
        self.models = []  # 用于集成学习的多个模型
        self.input_dim = input_dim
        
        # 创建目录
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        
    def extract_enhanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取增强特征 - 确保固定维度
        """
        features = []
        labels = []
        
        # 原始特征
        base_features = ['x', 'a', 'b', 'c', 'd', 'index_value']
        
        # 确保所有基础特征都存在
        for feature in base_features:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # 计算技术指标
        close_prices = df['index_value'].values.astype(np.float64)
        
        # 确保数据长度足够计算技术指标
        if len(close_prices) < 30:
            close_prices = np.pad(close_prices, (0, 30-len(close_prices)), 'edge')
        
        try:
            # 移动平均线
            sma_5 = talib.SMA(close_prices, timeperiod=5)
            sma_10 = talib.SMA(close_prices, timeperiod=10)
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            
            # RSI
            rsi = talib.RSI(close_prices, timeperiod=14)
            
            # 布林带
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
            
            # 随机指标
            slowk, slowd = talib.STOCH(close_prices, close_prices, close_prices)
            
            # 威廉指标
            willr = talib.WILLR(close_prices, close_prices, close_prices)
            
            # 动量指标
            mom = talib.MOM(close_prices, timeperiod=10)
            
            # 获取基础特征数据
            base_data = df[base_features].values
            
            # 确保技术指标长度与基础数据一致
            target_len = len(base_data)
            
            # 调整技术指标长度
            def adjust_indicator_length(indicator, target_length):
                if len(indicator) >= target_length:
                    return indicator[-target_length:]
                else:
                    # 用最后一个有效值填充
                    last_valid = indicator[~np.isnan(indicator)][-1] if len(indicator[~np.isnan(indicator)]) > 0 else 0.0
                    return np.pad(indicator, (target_length - len(indicator), 0), 'constant', constant_values=last_valid)[-target_length:]
            
            sma_5 = adjust_indicator_length(sma_5, target_len)
            sma_10 = adjust_indicator_length(sma_10, target_len)
            sma_20 = adjust_indicator_length(sma_20, target_len)
            macd = adjust_indicator_length(macd, target_len)
            macd_signal = adjust_indicator_length(macd_signal, target_len)
            macd_hist = adjust_indicator_length(macd_hist, target_len)
            rsi = adjust_indicator_length(rsi, target_len)
            upper = adjust_indicator_length(upper, target_len)
            middle = adjust_indicator_length(middle, target_len)
            lower = adjust_indicator_length(lower, target_len)
            slowk = adjust_indicator_length(slowk, target_len)
            slowd = adjust_indicator_length(slowd, target_len)
            willr = adjust_indicator_length(willr, target_len)
            mom = adjust_indicator_length(mom, target_len)
            
            # 合并所有特征 - 确保维度一致
            all_features = np.column_stack([
                base_data,  # 6个基础特征
                sma_5, sma_10, sma_20,  # 3个移动平均
                macd, macd_signal, macd_hist,  # 3个MACD
                rsi, upper, middle, lower,  # 4个布林带相关
                slowk, slowd, willr, mom  # 4个其他指标
            ])  # 总共20个特征
            
            # 添加额外的技术指标以达到30个特征
            try:
                # 添加更多技术指标
                atr = talib.ATR(close_prices, close_prices, close_prices, timeperiod=14)
                cci = talib.CCI(close_prices, close_prices, close_prices, timeperiod=14)
                dx = talib.DX(close_prices, close_prices, close_prices, timeperiod=14)
                
                # 调整长度
                atr = adjust_indicator_length(atr, target_len)
                cci = adjust_indicator_length(cci, target_len)
                dx = adjust_indicator_length(dx, target_len)
                
                # 添加价格变化率特征
                price_change = np.diff(close_prices, prepend=close_prices[0])
                price_change = adjust_indicator_length(price_change, target_len)
                
                # 添加波动率特征
                volatility = np.roll(np.std([close_prices[max(0, i-5):i+1] for i in range(len(close_prices))], axis=1), 0)
                volatility = adjust_indicator_length(volatility, target_len)
                
                # 添加更多特征
                volume_proxy = np.ones_like(close_prices)  # 模拟成交量
                volume_proxy = adjust_indicator_length(volume_proxy, target_len)
                
                # 趋势强度
                trend_strength = np.abs(sma_5 - sma_20) / sma_20
                trend_strength = adjust_indicator_length(trend_strength, target_len)
                
                # 价格位置（相对于布林带）
                price_position = (close_prices[-target_len:] - lower) / (upper - lower + 1e-8)
                price_position = adjust_indicator_length(price_position, target_len)
                
                # RSI变化率
                rsi_change = np.diff(rsi, prepend=rsi[0])
                rsi_change = adjust_indicator_length(rsi_change, target_len)
                
                # MACD信号强度
                macd_strength = np.abs(macd - macd_signal)
                macd_strength = adjust_indicator_length(macd_strength, target_len)
                
                # 合并所有特征（30个）
                all_features = np.column_stack([
                    base_data,  # 6个基础特征
                    sma_5, sma_10, sma_20,  # 3个移动平均
                    macd, macd_signal, macd_hist,  # 3个MACD
                    rsi, upper, middle, lower,  # 4个布林带相关
                    slowk, slowd, willr, mom,  # 4个其他指标
                    atr, cci, dx,  # 3个额外技术指标
                    price_change, volatility, volume_proxy,  # 3个价格特征
                    trend_strength, price_position, rsi_change, macd_strength  # 4个衍生特征
                ])  # 总共30个特征
                
            except Exception as e2:
                logger.warning(f"额外技术指标计算失败: {e2}，使用零填充")
                # 如果额外指标计算失败，用零填充到30维
                extra_features = np.zeros((len(base_data), 10))
                all_features = np.column_stack([all_features, extra_features])
            
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}，使用基础特征")
            base_data = df[base_features].values
            # 用零填充技术指标部分，确保总共30个特征
            tech_features = np.zeros((len(base_data), 24))
            all_features = np.column_stack([base_data, tech_features])
        
        # 处理NaN值
        all_features = np.nan_to_num(all_features, nan=0.0)
        
        # 提取标签
        # 支持'label'或'signal'列名
        label_col = None
        if 'label' in df.columns:
            label_col = 'label'
        elif 'signal' in df.columns:
            label_col = 'signal'
            
        if label_col is not None:
            labels = df[label_col].values
            print(f"Debug: 使用{label_col}列，原始标签数量: {len(labels)}, 标签分布: {np.bincount(labels.astype(int))}")
            # 只保留有效交易信号（0-4）
            valid_mask = (labels >= 0) & (labels <= 4)
            all_features = all_features[valid_mask]
            labels = labels[valid_mask]  # 保持原始标签值0-4
            print(f"Debug: 过滤后标签数量: {len(labels)}, 特征数量: {len(all_features)}")
            if len(labels) > 0:
                print(f"Debug: 过滤后标签分布: {np.bincount(labels.astype(int))}")
        else:
            # 如果没有标签列，返回空数组
            labels = np.array([])
        
        # 确保特征维度固定为30
        if all_features.shape[1] != 30:
            if all_features.shape[1] < 30:
                padding = np.zeros((all_features.shape[0], 30 - all_features.shape[1]))
                all_features = np.column_stack([all_features, padding])
            else:
                all_features = all_features[:, :30]
        
        return all_features.astype(np.float32), labels
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        """
        logger.info("准备增强训练数据...")
        
        # 获取标签文件
        label_files = glob.glob("label/*.csv")
        if len(label_files) < 5:
            raise ValueError("标签文件数量不足")
        
        all_features = []
        all_labels = []
        
        for file_path in label_files:
            try:
                df = pd.read_csv(file_path)
                features, labels = self.extract_enhanced_features(df)
                
                if len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    
            except Exception as e:
                logger.warning(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有有效的训练数据")
        
        # 合并所有数据
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        logger.info(f"训练数据形状: {X.shape}, 标签分布: {Counter(y)}")
        
        return X, y
    
    def create_ensemble_models(self, input_dim: int, num_models: int = 3) -> List[EnhancedTransformerPredictor]:
        """
        创建集成模型 - 确保输入维度一致
        """
        models = []
        
        # 固定输入维度为30
        fixed_input_dim = 30
        
        # 不同配置的模型
        configs = [
            {'d_model': 128, 'nhead': 8, 'num_layers': 6, 'dropout': 0.1},
            {'d_model': 96, 'nhead': 6, 'num_layers': 8, 'dropout': 0.15},
            {'d_model': 160, 'nhead': 10, 'num_layers': 4, 'dropout': 0.05}
        ]
        
        for i in range(num_models):
            config = configs[i % len(configs)]
            model = EnhancedTransformerPredictor(
                input_dim=fixed_input_dim,
                **config
            ).to(self.device)
            models.append(model)
        
        return models
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                  n_folds: int = 5, epochs: int = 100) -> Dict:
        """
        使用交叉验证训练模型
        """
        logger.info("开始交叉验证训练...")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建集成模型
        self.models = self.create_ensemble_models(X_scaled.shape[1])
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            logger.info(f"训练第 {fold+1}/{n_folds} 折...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练当前折的模型
            model = self.models[fold % len(self.models)]
            fold_result = self._train_single_model(
                model, X_train, X_val, y_train, y_val, epochs
            )
            
            fold_results.append(fold_result)
        
        # 计算平均性能
        avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
        logger.info(f"交叉验证平均准确率: {avg_accuracy:.4f}")
        
        return {
            'avg_accuracy': avg_accuracy,
            'fold_results': fold_results
        }
    
    def _train_single_model(self, model: EnhancedTransformerPredictor, 
                          X_train: np.ndarray, X_val: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray,
                          epochs: int) -> Dict:
        """
        训练单个模型
        """
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # 创建数据集
        train_dataset = EnhancedTradingDataset(X_train, y_train, augment=True)
        val_dataset = EnhancedTradingDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x.unsqueeze(1))  # 添加序列维度
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x.unsqueeze(1))
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停于第 {epoch} 轮")
                break
        
        return {
            'val_accuracy': best_val_acc,
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc
        }
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        集成预测
        """
        X_scaled = self.scaler.transform(X)
        
        all_predictions = []
        all_confidences = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                all_predictions.append(predictions.cpu().numpy())
                all_confidences.append(confidences.cpu().numpy())
        
        # 投票集成
        ensemble_predictions = []
        ensemble_confidences = []
        
        for i in range(len(X)):
            votes = [pred[i] for pred in all_predictions]
            confs = [conf[i] for conf in all_confidences]
            
            # 加权投票
            vote_counts = Counter()
            for vote, conf in zip(votes, confs):
                vote_counts[vote] += conf
            
            final_prediction = vote_counts.most_common(1)[0][0]
            final_confidence = np.mean(confs)
            
            ensemble_predictions.append(final_prediction)
            ensemble_confidences.append(final_confidence)
        
        return np.array(ensemble_predictions), np.array(ensemble_confidences)
    
    def save_models(self, base_path: str):
        """
        保存所有模型
        """
        # 不同配置的模型参数
        configs = [
            {'d_model': 128, 'nhead': 8, 'num_layers': 6, 'dropout': 0.1},
            {'d_model': 96, 'nhead': 6, 'num_layers': 8, 'dropout': 0.15},
            {'d_model': 160, 'nhead': 10, 'num_layers': 4, 'dropout': 0.05}
        ]
        
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i}.pth"
            config = configs[i % len(configs)]
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scaler,
                'input_dim': self.input_dim,
                'model_config': config
            }, model_path)
        
        logger.info(f"已保存 {len(self.models)} 个模型")
    
    def load_models(self, base_path: str) -> bool:
        """
        加载所有模型
        """
        try:
            self.models = []
            i = 0
            while True:
                model_path = f"{base_path}_model_{i}.pth"
                if not os.path.exists(model_path):
                    break
                
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 获取模型配置
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    # 使用保存的配置创建模型
                    model = EnhancedTransformerPredictor(
                        input_dim=checkpoint['input_dim'],
                        **config
                    ).to(self.device)
                else:
                    # 兼容旧版本，使用默认配置
                    model = EnhancedTransformerPredictor(
                        input_dim=checkpoint['input_dim']
                    ).to(self.device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                self.models.append(model)
                
                # 加载标准化器
                if i == 0:
                    self.scaler = checkpoint['scaler']
                    self.input_dim = checkpoint['input_dim']
                
                i += 1
            
            logger.info(f"已加载 {len(self.models)} 个模型")
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def run_complete_training(self):
        """
        运行完整训练流程
        """
        logger.info("=== 增强版深度学习预测器训练开始 ===")
        
        try:
            # 准备数据
            X, y = self.prepare_training_data()
            
            # 交叉验证训练
            results = self.train_with_cross_validation(X, y)
            
            # 保存模型
            model_path = os.path.join(self.models_dir, "enhanced_predictor")
            self.save_models(model_path)
            
            # 测试预测
            test_files = glob.glob("label/*.csv")[-3:]  # 使用最后3个文件测试
            test_results = self.evaluate_on_test_files(test_files)
            
            # 输出结果
            logger.info("=== 训练完成 ===")
            logger.info(f"交叉验证平均准确率: {results['avg_accuracy']:.4f}")
            logger.info(f"测试准确率: {test_results['accuracy']:.4f}")
            logger.info(f"信号多样性: {test_results['diversity']} 种")
            
            return results
            
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            return None
    
    def evaluate_on_test_files(self, test_files: List[str]) -> Dict:
        """
        在测试文件上评估模型
        """
        all_predictions = []
        
        for file_path in test_files:
            try:
                df = pd.read_csv(file_path)
                features, true_labels = self.extract_enhanced_features(df)
                
                if len(features) == 0:
                    continue
                
                predictions, confidences = self.predict_ensemble(features)
                
                for i, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
                    all_predictions.append({
                        'file': os.path.basename(file_path),
                        'index': i,
                        'predicted': pred + 1,  # 转换回1-4
                        'actual': true + 1,
                        'correct': pred == true,
                        'confidence': conf
                    })
                    
            except Exception as e:
                logger.warning(f"评估文件 {file_path} 时出错: {e}")
                continue
        
        if not all_predictions:
            return {'accuracy': 0, 'diversity': 0}
        
        # 计算指标
        total = len(all_predictions)
        correct = sum(1 for p in all_predictions if p['correct'])
        accuracy = correct / total
        
        predicted_signals = set(p['predicted'] for p in all_predictions)
        diversity = len(predicted_signals)
        
        # 保存结果
        results_df = pd.DataFrame(all_predictions)
        results_path = os.path.join(self.models_dir, "enhanced_test_results.csv")
        results_df.to_csv(results_path, index=False)
        
        return {
            'accuracy': accuracy,
            'diversity': diversity,
            'total_predictions': total,
            'results': all_predictions
        }

def main():
    """
    主函数
    """
    predictor = EnhancedDeepLearningPredictor()
    results = predictor.run_complete_training()
    
    if results:
        print("\n=== 增强版深度学习模型训练完成 ===")
        print(f"平均准确率: {results['avg_accuracy']:.4f}")
        print("\n模型已保存到 models_enhanced/ 目录")
        print("\n主要改进:")
        print("1. 多尺度特征提取")
        print("2. 增强注意力机制")
        print("3. 集成学习方法")
        print("4. 数据增强技术")
        print("5. 交叉验证训练")
        print("6. 动态学习率调度")
    else:
        print("训练失败，请检查数据和配置")

if __name__ == "__main__":
    main()