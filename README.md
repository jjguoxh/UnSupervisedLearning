# UnSupervisedLearning

## 项目概述

本项目是一个基于无监督学习的金融交易信号识别系统。通过聚类分析历史价格模式，识别出具有预测价值的交易信号模式。

## 标签系统说明

### 标签定义
- **0**: 无操作状态
- **1**: 做多开仓
- **2**: 做多平仓
- **3**: 做空开仓
- **4**: 做空平仓

### 标签生成规则
1. 使用动态回撤规则识别趋势段
2. 在趋势开始点标记开仓信号
3. 在趋势结束点标记平仓信号
4. 趋势段之间的所有点标记为0（表示无操作状态）
5. 根据最新需求，不再将开仓和平仓之间的点标记为开仓状态，而是保持它们为0（无操作）

## 项目结构

- `src/` - 源代码目录
  - `label_generation.py` - 标签生成脚本
  - `pattern_recognition.py` - 模式识别脚本
  - `trading_pattern_learning.py` - 交易模式学习脚本
  - `pattern_predictor.py` - 原始模式预测脚本
  - `pattern_predictor_balanced.py` - 平衡模式预测脚本（改进版，集成强化学习优化）
  - `realtime_predictor.py` - 实时预测脚本
  - `simple_rl_trader.py` - 简单强化学习交易器
  - `rl_optimized_realtime_predictor.py` - 强化学习优化的实时预测脚本
- `data/` - 原始数据目录
- `label/` - 标签数据目录
- `patterns/` - 模式数据目录
- `model/` - 模型保存目录
- `predictions/` - 预测结果目录
- `visualization/` - 可视化结果目录
- `realtime_data/` - 实时数据目录（用于目录监控模式）

## 使用方法

### 批处理脚本方式（推荐）

项目提供了几个批处理脚本来简化操作流程：

1. **完整流程控制脚本**:
   ```bash
   run_full_pipeline.bat
   ```
   该脚本允许您选择性地执行各个步骤：
   - 标签生成
   - 模式识别
   - 模式训练
   - 信号预测（使用平衡模式预测器，集成强化学习优化）

2. **快速流程脚本**:
   ```bash
   run_quick_pipeline.bat
   ```
   该脚本会自动依次执行所有步骤，无需用户干预，使用平衡模式预测器进行预测。

3. **实时预测脚本**:
   ```bash
   run_realtime_prediction.bat
   ```
   该脚本启动实时预测程序，支持三种运行模式：
   - 目录监控模式：监控指定目录中的新数据文件
   - 数据模拟模式：模拟实时数据流
   - 交互模式：手动控制预测过程

4. **平衡模式预测脚本**:
   ```bash
   run_balanced_prediction.bat
   ```
   该脚本专门用于运行改进的平衡模式预测器，对所有标签文件进行预测并生成可视化结果。

5. **强化学习优化实时预测脚本**:
   ```bash
   run_rl_optimized_prediction.bat
   ```
   该脚本启动强化学习优化的实时预测程序，使用强化学习模型进一步优化预测结果。

### 手动执行方式

1. 准备数据：将CSV格式的价格数据放入`data/`目录
2. 生成标签：运行`src/label_generation.py`
3. 模式识别：运行`src/pattern_recognition.py`
4. 模式训练：运行`src/trading_pattern_learning.py`
5. 训练强化学习模型：运行`src/simple_rl_trader.py`
6. 信号预测：
   - 原始预测器：运行`src/pattern_predictor.py`
   - 平衡预测器（推荐）：运行`src/pattern_predictor_balanced.py`
   - 实时预测器：运行`src/realtime_predictor.py`
   - 强化学习优化预测器：运行`src/rl_optimized_realtime_predictor.py`

### 实时预测模式说明

实时预测器支持三种不同的运行模式：

1. **目录监控模式**：
   - 程序会持续监控`realtime_data/`目录
   - 当有新的CSV文件放入该目录时，自动进行预测
   - 预测结果保存在`predictions/`和`visualization/`目录中

2. **数据模拟模式**：
   - 使用历史数据模拟实时数据流
   - 逐点进行预测，模拟真实的实时预测场景

3. **交互模式**：
   - 提供菜单式交互界面
   - 用户可以选择不同的操作，包括加载文件预测、启动监控模式等

### 强化学习优化功能

本项目新增了强化学习优化功能，通过以下方式提升预测性能：

1. **训练强化学习模型**：
   - 运行`src/simple_rl_trader.py`训练强化学习模型
   - 模型将根据预测信号和实际信号的匹配情况学习最优决策策略

2. **集成强化学习优化**：
   - 平衡模式预测器(`pattern_predictor_balanced.py`)已集成强化学习优化功能
   - 强化学习模型会根据预测信号的置信度和其他特征决定是否执行该信号

3. **强化学习优化的实时预测**：
   - 使用`rl_optimized_realtime_predictor.py`可以获得强化学习优化的实时预测结果

## 注意事项

- 根据最新需求，不再将开仓和平仓之间的点标记为开仓状态，而是保持它们为0（无操作）
- 在评估预测准确性时，标签1表示做多开仓状态，标签3表示做空开仓状态
- 平衡模式预测器使用严格平衡后的数据，提供更好的预测性能
- 可视化结果保存在`visualization/`目录中，每个标签文件都有对应的预测可视化图像
- 实时预测不需要标签数据，可以直接对原始价格数据进行预测
- 强化学习模型会自动保存到`model/balanced_model/rl_trader_model.json`，可重复使用