# 无监督学习交易信号识别系统

## 项目概述

本项目是一个基于无监督学习的交易信号识别系统，能够自动识别金融时间序列数据中的交易信号，包括做多开仓、做多平仓、做空开仓和做空平仓信号。

## 系统特点

1. **多模型支持**：提供多种预测模型，包括基于聚类的传统模型、改进的机器学习模型和基于Transformer的深度学习模型
2. **信号完整性保证**：确保每个交易日至少有一个开仓和一个平仓信号
3. **数据不平衡处理**：专门处理0标签（无操作信号）过多的问题，只使用交易信号进行训练
4. **实时预测**：支持实时数据流预测和批量文件预测

## 模型介绍

### 1. 传统聚类模型 (pattern_predictor_balanced.py)
- 基于K-means聚类和PCA降维
- 使用无监督学习识别交易模式
- 适用于快速原型开发和基准测试

### 2. 改进的机器学习模型 (improved_predictor.py)
- 为每种信号类型训练专门的随机森林模型
- 丰富的特征工程，包括多种技术指标
- 更好的预测准确性和多样性

### 3. Transformer深度学习模型 (transformer_predictor.py)
- 基于Transformer架构的序列预测模型
- 能够捕捉长期时间依赖关系
- 通过后处理确保信号完整性

## 快速开始

### 1. 环境准备
确保已安装以下依赖：
```bash
pip install numpy pandas scikit-learn torch matplotlib
```

### 2. 数据准备
将CSV格式的时间序列数据放入[data/](file:///e:/unsupervised_learning/data/)目录，文件应包含以下列：
- x: 特征x
- a, b, c, d: 其他特征
- index_value: 指数值

### 3. 生成标签
```bash
# 生成交易信号标签
python src/label_generation.py
```

### 4. 训练模型
```bash
# 训练所有模型
python train_all_models.py

# 或单独训练特定模型
python train_transformer_model.py

# 或使用完整的训练工作流
train_transformer_workflow.bat
```

### 5. 运行预测
```bash
# 运行传统模型预测
run_prediction.bat

# 运行改进模型预测
run_improved_prediction.bat

# 运行Transformer模型预测
run_transformer_prediction.bat
```

## 使用说明

### 批处理文件
- [run_quick_pipeline.bat](file:///e:/unsupervised_learning/run_quick_pipeline.bat): 快速运行完整流程
- [run_prediction.bat](file:///e:/unsupervised_learning/run_prediction.bat): 运行传统模型预测
- [run_improved_prediction.bat](file:///e:/unsupervised_learning/run_improved_prediction.bat): 运行改进模型预测
- [run_transformer_prediction.bat](file:///e:/unsupervised_learning/run_transformer_prediction.bat): 运行Transformer模型预测
- [train_transformer_workflow.bat](file:///e:/unsupervised_learning/train_transformer_workflow.bat): 完整的Transformer模型训练工作流

### 运行模式
所有预测程序都支持以下三种模式：
1. **目录监控模式**：监控[realtime_data/](file:///e:/unsupervised_learning/realtime_data/)目录中的新CSV文件并自动预测
2. **数据模拟模式**：使用历史数据模拟实时数据流进行预测
3. **交互模式**：手动选择文件进行预测

## 完整的Transformer训练工作流

[train_transformer_workflow.bat](file:///e:/unsupervised_learning/train_transformer_workflow.bat)文件提供了一个完整的训练工作流，包括以下步骤：

1. **生成交易信号标签**：运行[label_generation.py](file:///e:/unsupervised_learning/src/label_generation.py)生成标签文件
2. **运行模式学习**：运行[advanced_pattern_learning.py](file:///e:/unsupervised_learning/src/advanced_pattern_learning.py)学习交易模式
3. **训练强化学习模型**：运行[simple_rl_trader.py](file:///e:/unsupervised_learning/src/simple_rl_trader.py)训练强化学习模型
4. **训练Transformer模型**：运行[train_transformer_model.py](file:///e:/unsupervised_learning/train_transformer_model.py)训练Transformer深度学习模型

该工作流确保所有必要的前置步骤都已完成，然后训练Transformer模型。

## 项目结构
```
unsupervised_learning/
├── data/                   # 原始数据文件
├── label/                  # 生成的标签文件
├── patterns/               # 聚类模式文件
├── model/                  # 训练好的模型
│   └── balanced_model/     # 平衡模型目录
├── predictions/            # 预测结果
├── visualization/          # 可视化结果
├── src/                    # 源代码
└── doc/                    # 文档
```

## 改进历史

详细改进记录请参见[IMPROVEMENT_SUMMARY.md](file:///e:/unsupervised_learning/IMPROVEMENT_SUMMARY.md)文件。

## 许可证
本项目仅供学习和研究使用。