# -*- coding: utf-8 -*-
"""
训练增强版深度学习模型
使用多种先进技术提高预测准确性
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import logging
import warnings
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# 导入增强版预测器
from enhanced_deep_learning_predictor import EnhancedDeepLearningPredictor

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(CURRENT_DIR, "E:/unsupervised_learning/label/")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed_data/")
MODELS_DIR = os.path.join(CURRENT_DIR, "models_enhanced/")
TRAINING_RESULTS_DIR = os.path.join(CURRENT_DIR, "training_results/")

def load_training_data():
    """
    加载训练数据
    """
    logger.info("开始加载训练数据...")
    
    # 首先尝试使用预处理的数据
    processed_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    
    if processed_files:
        logger.info(f"使用预处理数据，找到 {len(processed_files)} 个文件")
        data_files = processed_files
    else:
        # 如果没有预处理数据，尝试原始标签文件
        label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv")))
        
        if not label_files:
            logger.error(f"在 {LABEL_DIR} 和 {PROCESSED_DIR} 目录中都没有找到数据文件")
            logger.info("请先运行: python prepare_training_data.py")
            return None, None
        
        logger.info(f"使用原始标签数据，找到 {len(label_files)} 个文件")
        data_files = label_files
    
    all_features = []
    all_labels = []
    
    # 创建临时预测器用于特征提取
    temp_predictor = EnhancedDeepLearningPredictor()
    
    for i, file_path in enumerate(data_files):
        try:
            # 加载数据
            df = pd.read_csv(file_path)
            
            # 检查必要的列
            required_columns = ['x', 'a', 'b', 'c', 'd', 'index_value']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"文件 {file_path} 缺少必要的列，跳过")
                continue
            
            # 数据清洗
            df = df.dropna(subset=required_columns)
            
            if len(df) < 50:  # 至少需要50个数据点
                logger.warning(f"文件 {file_path} 数据点不足，跳过")
                continue
            
            # 提取增强特征
            features, labels = temp_predictor.extract_enhanced_features(df)
            
            if len(features) > 0:
                all_features.extend(features)
                all_labels.extend(labels)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(data_files)} 个文件")
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    if not all_features:
        logger.error("没有成功提取到任何特征")
        return None, None
    
    # 转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.info(f"成功加载数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    # 确保标签是整数类型
    y = y.astype(int)
    logger.info(f"标签分布: {np.bincount(y)}")
    
    return X, y

def evaluate_model_performance(predictor, X_test, y_test):
    """
    评估模型性能
    """
    logger.info("开始模型性能评估...")
    
    try:
        # 进行预测
        predictions, confidences = predictor.predict_ensemble(X_test)
        
        # 计算准确率
        accuracy = np.mean(predictions == y_test)
        
        # 生成分类报告
        class_names = ['做多开仓', '做多平仓', '做空开仓', '做空平仓']
        report = classification_report(y_test, predictions, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, predictions)
        
        # 输出结果
        logger.info(f"\n=== 模型性能评估结果 ===")
        logger.info(f"总体准确率: {accuracy:.4f}")
        logger.info(f"平均置信度: {np.mean(confidences):.4f}")
        
        # 各类别性能
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                logger.info(f"{class_name}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
        
        # 保存评估结果
        evaluation_results = {
            'accuracy': float(accuracy),
            'avg_confidence': float(np.mean(confidences)),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'test_samples': len(y_test),
            'evaluation_time': datetime.now().isoformat()
        }
        
        # 保存到文件
        results_path = os.path.join(TRAINING_RESULTS_DIR, "enhanced_model_evaluation.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 创建可视化
        create_evaluation_plots(cm, class_names, confidences, accuracy)
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        return None

def create_evaluation_plots(cm, class_names, confidences, accuracy):
    """
    创建评估可视化图表
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
        axes[0,0].set_title('混淆矩阵')
        axes[0,0].set_xlabel('预测标签')
        axes[0,0].set_ylabel('真实标签')
        
        # 2. 置信度分布
        axes[0,1].hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('预测置信度分布')
        axes[0,1].set_xlabel('置信度')
        axes[0,1].set_ylabel('频次')
        axes[0,1].axvline(np.mean(confidences), color='red', linestyle='--', 
                         label=f'平均值: {np.mean(confidences):.3f}')
        axes[0,1].legend()
        
        # 3. 类别准确率
        class_accuracies = []
        for i in range(len(class_names)):
            class_mask = (np.arange(len(cm)) == i)
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
            else:
                class_acc = 0
            class_accuracies.append(class_acc)
        
        bars = axes[1,0].bar(class_names, class_accuracies, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        axes[1,0].set_title('各类别准确率')
        axes[1,0].set_ylabel('准确率')
        axes[1,0].set_ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, acc in zip(bars, class_accuracies):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. 模型性能总结
        axes[1,1].text(0.1, 0.8, f'总体准确率: {accuracy:.4f}', fontsize=14, fontweight='bold')
        axes[1,1].text(0.1, 0.7, f'平均置信度: {np.mean(confidences):.4f}', fontsize=12)
        axes[1,1].text(0.1, 0.6, f'测试样本数: {len(confidences)}', fontsize=12)
        axes[1,1].text(0.1, 0.5, f'训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12)
        
        # 添加性能等级
        if accuracy >= 0.8:
            performance_level = "优秀"
            color = 'green'
        elif accuracy >= 0.7:
            performance_level = "良好"
            color = 'blue'
        elif accuracy >= 0.6:
            performance_level = "一般"
            color = 'orange'
        else:
            performance_level = "需要改进"
            color = 'red'
        
        axes[1,1].text(0.1, 0.4, f'性能等级: {performance_level}', fontsize=12, color=color, fontweight='bold')
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        
        plt.suptitle('增强版深度学习模型 - 性能评估报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(TRAINING_RESULTS_DIR, "enhanced_model_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"评估图表已保存: {plot_path}")
        
    except Exception as e:
        logger.error(f"创建评估图表失败: {e}")

def create_training_summary(training_history, evaluation_results):
    """
    创建训练总结报告
    """
    try:
        summary = {
            'training_info': {
                'start_time': training_history.get('start_time'),
                'end_time': datetime.now().isoformat(),
                'total_epochs': training_history.get('total_epochs', 0),
                'models_trained': training_history.get('models_trained', 0)
            },
            'data_info': {
                'total_samples': training_history.get('total_samples', 0),
                'training_samples': training_history.get('training_samples', 0),
                'test_samples': training_history.get('test_samples', 0),
                'feature_dimension': training_history.get('feature_dimension', 0)
            },
            'performance': evaluation_results if evaluation_results else {},
            'model_architecture': {
                'ensemble_size': 5,
                'features': [
                    '多尺度特征提取',
                    '增强注意力机制',
                    '残差连接',
                    '批量归一化',
                    '集成学习',
                    '自适应学习率'
                ]
            }
        }
        
        # 保存总结
        summary_path = os.path.join(TRAINING_RESULTS_DIR, "enhanced_training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 创建Markdown报告
        md_path = os.path.join(TRAINING_RESULTS_DIR, "ENHANCED_MODEL_REPORT.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 增强版深度学习模型训练报告\n\n")
            
            f.write("## 模型概述\n")
            f.write("本模型采用了多种先进的深度学习技术，包括：\n")
            f.write("- 多尺度特征提取\n")
            f.write("- 增强注意力机制\n")
            f.write("- 残差连接\n")
            f.write("- 批量归一化\n")
            f.write("- 集成学习\n")
            f.write("- 自适应学习率\n\n")
            
            f.write("## 训练信息\n")
            f.write(f"- 训练开始时间: {training_history.get('start_time', 'N/A')}\n")
            f.write(f"- 训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 总训练轮数: {training_history.get('total_epochs', 0)}\n")
            f.write(f"- 模型数量: {training_history.get('models_trained', 0)}\n\n")
            
            f.write("## 数据信息\n")
            f.write(f"- 总样本数: {training_history.get('total_samples', 0)}\n")
            f.write(f"- 训练样本数: {training_history.get('training_samples', 0)}\n")
            f.write(f"- 测试样本数: {training_history.get('test_samples', 0)}\n")
            f.write(f"- 特征维度: {training_history.get('feature_dimension', 0)}\n\n")
            
            if evaluation_results:
                f.write("## 性能评估\n")
                f.write(f"- 总体准确率: {evaluation_results['accuracy']:.4f}\n")
                f.write(f"- 平均置信度: {evaluation_results['avg_confidence']:.4f}\n")
                f.write(f"- 测试样本数: {evaluation_results['test_samples']}\n\n")
                
                f.write("### 各类别性能\n")
                class_names = ['做多开仓', '做多平仓', '做空开仓', '做空平仓']
                report = evaluation_results['classification_report']
                
                for i, class_name in enumerate(class_names):
                    if str(i) in report:
                        precision = report[str(i)]['precision']
                        recall = report[str(i)]['recall']
                        f1 = report[str(i)]['f1-score']
                        f.write(f"- {class_name}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}\n")
            
            f.write("\n## 使用方法\n")
            f.write("```bash\n")
            f.write("# 实时预测\n")
            f.write("python enhanced_realtime_predictor.py --mode interactive\n\n")
            f.write("# 目录监控\n")
            f.write("python enhanced_realtime_predictor.py --mode monitor\n\n")
            f.write("# 数据模拟\n")
            f.write("python enhanced_realtime_predictor.py --mode simulate\n")
            f.write("```\n")
        
        logger.info(f"训练总结已保存: {summary_path}")
        logger.info(f"Markdown报告已保存: {md_path}")
        
    except Exception as e:
        logger.error(f"创建训练总结失败: {e}")

def main():
    """
    主训练函数
    """
    logger.info("开始训练增强版深度学习模型...")
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    
    # 记录训练开始时间
    training_start_time = datetime.now()
    training_history = {
        'start_time': training_start_time.isoformat()
    }
    
    try:
        # 1. 加载训练数据
        X, y = load_training_data()
        if X is None or y is None:
            logger.error("无法加载训练数据")
            return
        
        # 记录数据信息
        training_history.update({
            'total_samples': len(X),
            'feature_dimension': X.shape[1]
        })
        
        # 2. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        training_history.update({
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"测试集大小: {len(X_test)}")
        
        # 3. 创建并训练模型
        predictor = EnhancedDeepLearningPredictor()
        
        # 训练模型
        results = predictor.train_with_cross_validation(X_train, y_train, epochs=100)
        
        if not results:
            logger.error("模型训练失败")
            return
        
        training_history.update({
            'total_epochs': 100,
            'models_trained': len(predictor.models),
            'avg_accuracy': results['avg_accuracy']
        })
        
        # 4. 保存模型
        model_path = os.path.join(MODELS_DIR, "enhanced_predictor")
        if predictor.save_models(model_path):
            logger.info(f"模型已保存到: {model_path}")
        else:
            logger.error("模型保存失败")
        
        # 5. 评估模型性能
        evaluation_results = evaluate_model_performance(predictor, X_test, y_test)
        
        # 6. 创建训练总结
        create_training_summary(training_history, evaluation_results)
        
        # 7. 输出最终结果
        logger.info("\n=== 训练完成 ===")
        if evaluation_results:
            logger.info(f"最终准确率: {evaluation_results['accuracy']:.4f}")
            logger.info(f"平均置信度: {evaluation_results['avg_confidence']:.4f}")
        
        logger.info(f"模型文件: {model_path}")
        logger.info(f"结果目录: {TRAINING_RESULTS_DIR}")
        logger.info("\n可以使用以下命令进行预测:")
        logger.info("python enhanced_realtime_predictor.py --mode interactive")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()