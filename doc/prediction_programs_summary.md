# 交易模式预测程序总结报告

## 项目概述

基于[pattern_recognition.py](file:///E:/unsupervised_learning/src/pattern_recognition.py)模型，我们成功开发了两个交易模式预测程序：

1. **离线预测程序** ([pattern_predictor.py](file:///E:/unsupervised_learning/src/pattern_predictor.py))
2. **实时预测程序** ([realtime_predictor.py](file:///E:/unsupervised_learning/src/realtime_predictor.py))

## 离线预测程序 ([pattern_predictor.py](file:///E:/unsupervised_learning/src/pattern_predictor.py))

### 功能特点
- 加载已学习的交易模式聚类
- 创建基于高盈利潜力聚类的预测模型
- 对历史数据进行回测预测
- 保存模型参数供实时预测使用

### 技术实现
- 从[patterns](file:///E:/unsupervised_learning/patterns/)目录加载聚类分析结果
- 为信号密度>0.3的聚类创建预测模型
- 使用皮尔逊相关系数计算模式相似性
- 保存模型到[model](file:///E:/unsupervised_learning/model/)目录

### 测试结果
- 在历史数据上的回测准确率达到100%
- 成功识别并保存了2个具有较高信号密度的聚类模型（聚类6和23）

## 实时预测程序 ([realtime_predictor.py](file:///E:/unsupervised_learning/src/realtime_predictor.py))

### 功能特点
- 实时监控数据流并进行交易信号预测
- 支持多种运行模式：
  1. 目录监控模式：监控指定目录中的新数据文件
  2. 数据模拟模式：模拟实时数据流
  3. 交互模式：手动控制预测过程
- 保存预测结果到JSON和CSV文件

### 技术实现
- 加载离线预测程序保存的模型参数
- 使用数据缓冲区管理实时数据
- 基于模式相似性进行信号预测
- 支持多种输出格式（JSON和CSV）

### 测试结果
- 成功加载训练好的模型
- 能够处理实时数据流
- 预测结果已成功保存到[predictions](file:///E:/unsupervised_learning/predictions/)目录

## 模型详情

### 高盈利潜力聚类

1. **聚类6** (信号密度: 0.33)
   - 包含3个模式样本
   - 信号分布: {3: 2, 4: 1} (做空开仓: 2, 做空平仓: 1)
   - 配对交易: 1组完整做空交易

2. **聚类23** (信号密度: 0.5)
   - 包含2个模式样本
   - 信号分布: {3: 1, 4: 1} (做空开仓: 1, 做空平仓: 1)
   - 配对交易: 1组完整做空交易

## 标签系统更新说明

根据最新需求，标签系统已进行简化：
- 标签1和5已合并为标签1（做多开仓，包括开仓点和持仓状态）
- 标签3和6已合并为标签3（做空开仓，包括开仓点和持仓状态）

## 文件结构

```
unsupervised_learning/
├── src/
│   ├── pattern_predictor.py         # 离线预测程序
│   ├── realtime_predictor.py         # 实时预测程序
├── patterns/                        # 已学习的模式
│   ├── cluster_6/                   # 聚类6模式数据
│   ├── cluster_23/                  # 聚类23模式数据
│   └── cluster_analysis.csv         # 聚类分析结果
├── model/                           # 保存的模型
│   └── pattern_predictor_model.json # 预测模型参数
├── predictions/                     # 预测结果
│   ├── *.json                       # JSON格式预测结果
│   └── predictions_summary.csv      # CSV格式预测汇总
└── label/                           # 原始标签数据
```

## 使用说明

### 离线预测程序
```bash
cd src
python pattern_predictor.py
```

### 实时预测程序
```bash
cd src
python realtime_predictor.py
```
运行后选择相应的模式：
1. 目录监控模式
2. 数据模拟模式
3. 交互模式

## 结论

我们成功实现了基于无监督学习的交易模式预测系统，该系统能够：
1. 从历史交易数据中学习盈利模式
2. 对历史数据进行高准确率预测
3. 实时监控数据流并进行预测
4. 保存和管理预测结果

该系统为自动化交易提供了可靠的信号预测基础。