#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无监督学习交易信号预测系统 - 专注改进工具
基于交易信号稀疏性的合理性，专注提高交易信号预测准确性
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from collections import Counter
import pickle
from datetime import datetime

class FocusedImprovement:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.label_dir = os.path.join(base_dir, "label")
        self.model_dir = os.path.join(base_dir, "model")
        
    def analyze_trading_signal_quality(self):
        """分析交易信号质量，专注于1-4信号"""
        print("\n=== 交易信号质量分析 ===")
        
        all_labels = []
        trading_signals = []
        signal_contexts = []  # 存储信号前后的市场环境
        
        # 收集所有标签数据
        for file in os.listdir(self.label_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.label_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    labels = df['label'].values
                    all_labels.extend(labels)
                    
                    # 分析交易信号的上下文
                    for i, label in enumerate(labels):
                        if label in [1, 2, 3, 4]:  # 交易信号
                            trading_signals.append(label)
                            
                            # 提取信号前后的市场环境特征
                            context = {
                                'signal': label,
                                'position': i,
                                'total_length': len(labels)
                            }
                            
                            # 添加价格变化信息（如果有index_value列）
                            if 'index_value' in df.columns and i > 0:
                                context['price_change'] = df.iloc[i]['index_value'] - df.iloc[i-1]['index_value']
                                
                                # 计算信号后的价格变化（用于验证信号有效性）
                                if i < len(df) - 10:  # 确保有足够的后续数据
                                    future_prices = df.iloc[i:i+10]['index_value'].values
                                    context['future_return_5'] = (future_prices[4] - future_prices[0]) / future_prices[0] if len(future_prices) > 4 else 0
                                    context['future_return_10'] = (future_prices[-1] - future_prices[0]) / future_prices[0]
                            
                            signal_contexts.append(context)
                            
                except Exception as e:
                    print(f"警告: 无法读取 {file_path}: {e}")
        
        # 分析结果
        total_points = len(all_labels)
        trading_signal_count = len(trading_signals)
        
        print(f"总数据点: {total_points}")
        print(f"交易信号数量: {trading_signal_count}")
        print(f"交易信号比例: {trading_signal_count/total_points*100:.2f}%")
        
        signal_distribution = Counter(trading_signals)
        print("\n交易信号分布:")
        signal_names = {1: "做多开仓", 2: "做多平仓", 3: "做空开仓", 4: "做空平仓"}
        for signal, count in sorted(signal_distribution.items()):
            print(f"  {signal_names[signal]}({signal}): {count} 次")
        
        # 分析信号有效性（基于未来收益）
        if signal_contexts:
            print("\n信号有效性分析:")
            for signal_type in [1, 2, 3, 4]:
                signal_data = [ctx for ctx in signal_contexts if ctx['signal'] == signal_type and 'future_return_5' in ctx]
                if signal_data:
                    returns_5 = [ctx['future_return_5'] for ctx in signal_data]
                    returns_10 = [ctx['future_return_10'] for ctx in signal_data]
                    
                    avg_return_5 = np.mean(returns_5) * 100
                    avg_return_10 = np.mean(returns_10) * 100
                    
                    print(f"  {signal_names[signal_type]}: 5期平均收益 {avg_return_5:.2f}%, 10期平均收益 {avg_return_10:.2f}%")
        
        return signal_contexts
    
    def create_trading_focused_features(self):
        """创建专注于交易信号的特征"""
        print("\n=== 创建交易专用特征 ===")
        
        enhanced_features = []
        enhanced_labels = []
        
        for file in os.listdir(self.label_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.label_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    for i in range(20, len(df)-20):  # 确保有足够的前后数据
                        label = df.iloc[i]['label']
                        
                        # 只处理交易信号和部分0信号（作为负样本）
                        if label in [1, 2, 3, 4] or (label == 0 and np.random.random() < 0.1):
                            features = self.extract_enhanced_features(df, i)
                            if features is not None:
                                enhanced_features.append(features)
                                enhanced_labels.append(label)
                                
                except Exception as e:
                    print(f"警告: 处理文件 {file_path} 失败: {e}")
        
        print(f"✓ 创建了 {len(enhanced_features)} 个增强特征样本")
        print(f"标签分布: {dict(Counter(enhanced_labels))}")
        
        return np.array(enhanced_features), np.array(enhanced_labels)
    
    def extract_enhanced_features(self, df, index):
        """提取增强的交易特征"""
        try:
            features = []
            
            # 基础特征
            if 'a' in df.columns:
                features.extend([df.iloc[index]['a'], df.iloc[index]['b'], 
                               df.iloc[index]['c'], df.iloc[index]['d']])
            
            # 价格相关特征
            if 'index_value' in df.columns:
                current_price = df.iloc[index]['index_value']
                
                # 短期价格变化
                price_changes = []
                for lookback in [1, 3, 5, 10]:
                    if index >= lookback:
                        past_price = df.iloc[index-lookback]['index_value']
                        change = (current_price - past_price) / past_price
                        price_changes.append(change)
                    else:
                        price_changes.append(0)
                
                features.extend(price_changes)
                
                # 价格波动率
                if index >= 10:
                    recent_prices = df.iloc[index-10:index]['index_value'].values
                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                    features.append(volatility)
                else:
                    features.append(0)
                
                # 趋势强度
                if index >= 20:
                    long_prices = df.iloc[index-20:index]['index_value'].values
                    trend_strength = (long_prices[-1] - long_prices[0]) / long_prices[0]
                    features.append(trend_strength)
                else:
                    features.append(0)
            
            # 技术指标特征
            if 'a' in df.columns and 'b' in df.columns:
                # 简单移动平均
                if index >= 5:
                    ma_a = np.mean(df.iloc[index-5:index]['a'].values)
                    ma_b = np.mean(df.iloc[index-5:index]['b'].values)
                    features.extend([ma_a, ma_b])
                    
                    # 当前值与移动平均的偏离
                    features.extend([
                        df.iloc[index]['a'] - ma_a,
                        df.iloc[index]['b'] - ma_b
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            return None
    
    def train_specialized_models(self, X, y):
        """训练专门的交易信号预测模型"""
        print("\n=== 训练专用交易信号模型 ===")
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分离交易信号和非交易信号
        trading_mask = y != 0
        X_trading = X_scaled[trading_mask]
        y_trading = y[trading_mask]
        
        print(f"交易信号样本数: {len(X_trading)}")
        print(f"交易信号分布: {dict(Counter(y_trading))}")
        
        if len(X_trading) < 10:
            print("警告: 交易信号样本太少，无法训练有效模型")
            return None
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_trading, y_trading, test_size=0.3, random_state=42, stratify=y_trading
        )
        
        # 训练多个模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\n训练 {name} 模型...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
            print(f"交叉验证 F1 得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 测试预测
            y_pred = model.predict(X_test)
            
            # 计算详细指标
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            print(f"测试集性能:")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1得分: {f1:.4f}")
            
            # 各类别详细报告
            print(f"\n各信号类别性能:")
            report = classification_report(y_test, y_pred, output_dict=True)
            signal_names = {1: "做多开仓", 2: "做多平仓", 3: "做空开仓", 4: "做空平仓"}
            
            for signal in [1, 2, 3, 4]:
                if str(signal) in report:
                    metrics = report[str(signal)]
                    print(f"  {signal_names[signal]}: 精确率={metrics['precision']:.3f}, 召回率={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
            # 保存结果
            results[name] = {
                'cv_f1_score': cv_scores.mean(),
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'classification_report': report
            }
            
            # 选择最佳模型
            if f1 > best_score:
                best_score = f1
                best_model = (name, model, scaler)
        
        # 保存最佳模型
        if best_model:
            model_name, model, scaler = best_model
            
            specialized_dir = os.path.join(self.model_dir, "specialized")
            os.makedirs(specialized_dir, exist_ok=True)
            
            # 保存模型
            model_file = os.path.join(specialized_dir, f"best_trading_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存标准化器
            scaler_file = os.path.join(specialized_dir, f"trading_scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"\n✓ 最佳模型 ({model_name}) 已保存，F1得分: {best_score:.4f}")
        
        # 保存训练结果
        results_file = os.path.join(specialized_dir, "training_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def create_trading_predictor(self):
        """创建专用的交易信号预测器"""
        print("\n=== 创建交易信号预测器 ===")
        
        predictor_code = '''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用交易信号预测器
专注于预测1-4交易信号，忽略0信号的平衡问题
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class TradingSignalPredictor:
    def __init__(self, model_dir="model/specialized"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            model_file = os.path.join(self.model_dir, "best_trading_model.pkl")
            scaler_file = os.path.join(self.model_dir, "trading_scaler.pkl")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✓ 模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def extract_features(self, df, index):
        """提取预测特征（与训练时保持一致）"""
        try:
            features = []
            
            # 基础特征
            if 'a' in df.columns:
                features.extend([df.iloc[index]['a'], df.iloc[index]['b'], 
                               df.iloc[index]['c'], df.iloc[index]['d']])
            
            # 价格相关特征
            if 'index_value' in df.columns:
                current_price = df.iloc[index]['index_value']
                
                # 短期价格变化
                price_changes = []
                for lookback in [1, 3, 5, 10]:
                    if index >= lookback:
                        past_price = df.iloc[index-lookback]['index_value']
                        change = (current_price - past_price) / past_price
                        price_changes.append(change)
                    else:
                        price_changes.append(0)
                
                features.extend(price_changes)
                
                # 价格波动率
                if index >= 10:
                    recent_prices = df.iloc[index-10:index]['index_value'].values
                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                    features.append(volatility)
                else:
                    features.append(0)
                
                # 趋势强度
                if index >= 20:
                    long_prices = df.iloc[index-20:index]['index_value'].values
                    trend_strength = (long_prices[-1] - long_prices[0]) / long_prices[0]
                    features.append(trend_strength)
                else:
                    features.append(0)
            
            # 技术指标特征
            if 'a' in df.columns and 'b' in df.columns:
                # 简单移动平均
                if index >= 5:
                    ma_a = np.mean(df.iloc[index-5:index]['a'].values)
                    ma_b = np.mean(df.iloc[index-5:index]['b'].values)
                    features.extend([ma_a, ma_b])
                    
                    # 当前值与移动平均的偏离
                    features.extend([
                        df.iloc[index]['a'] - ma_a,
                        df.iloc[index]['b'] - ma_b
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            return np.array(features).reshape(1, -1) if len(features) > 0 else None
            
        except Exception as e:
            return None
    
    def predict_signal(self, df, index):
        """预测交易信号"""
        if self.model is None or self.scaler is None:
            return 0, 0.0
        
        features = self.extract_features(df, index)
        if features is None:
            return 0, 0.0
        
        try:
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            
            # 获取预测概率（置信度）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.5
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            return 0, 0.0
    
    def batch_predict(self, df, min_confidence=0.6):
        """批量预测整个数据集"""
        predictions = []
        confidences = []
        
        for i in range(len(df)):
            pred, conf = self.predict_signal(df, i)
            
            # 只保留高置信度的交易信号
            if pred != 0 and conf < min_confidence:
                pred = 0
            
            predictions.append(pred)
            confidences.append(conf)
        
        return predictions, confidences
    
    def evaluate_predictions(self, df, predictions, actual_labels=None):
        """评估预测结果"""
        signal_names = {0: "等待", 1: "做多开仓", 2: "做多平仓", 3: "做空开仓", 4: "做空平仓"}
        
        # 统计预测信号
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        print("\n预测信号统计:")
        for signal, count in sorted(pred_counts.items()):
            percentage = count / len(predictions) * 100
            print(f"  {signal_names.get(signal, f'未知({signal})')}: {count} 次 ({percentage:.2f}%)")
        
        # 如果有实际标签，计算准确率
        if actual_labels is not None:
            trading_signals_pred = [p for p in predictions if p != 0]
            trading_signals_actual = [a for a in actual_labels if a != 0]
            
            if len(trading_signals_pred) > 0 and len(trading_signals_actual) > 0:
                # 只评估交易信号的准确性
                correct_trading = sum(1 for p, a in zip(predictions, actual_labels) 
                                    if p != 0 and p == a)
                total_trading_pred = sum(1 for p in predictions if p != 0)
                
                if total_trading_pred > 0:
                    trading_precision = correct_trading / total_trading_pred
                    print(f"\n交易信号精确率: {trading_precision:.2%}")
                    print(f"正确交易信号: {correct_trading}/{total_trading_pred}")

def test_predictor():
    """测试预测器"""
    predictor = TradingSignalPredictor()
    
    # 测试数据文件
    test_files = [f for f in os.listdir("label") if f.endswith(".csv")][:3]
    
    for file_name in test_files:
        print(f"\n=== 测试文件: {file_name} ===")
        
        file_path = os.path.join("label", file_name)
        df = pd.read_csv(file_path)
        
        # 批量预测
        predictions, confidences = predictor.batch_predict(df, min_confidence=0.7)
        
        # 评估结果
        actual_labels = df['label'].tolist() if 'label' in df.columns else None
        predictor.evaluate_predictions(df, predictions, actual_labels)

if __name__ == "__main__":
    test_predictor()
'''
        
        predictor_file = os.path.join(self.base_dir, "trading_signal_predictor.py")
        with open(predictor_file, 'w', encoding='utf-8') as f:
            f.write(predictor_code)
        
        print(f"✓ 交易信号预测器已创建: {predictor_file}")
        return True
    
    def run_focused_improvement(self):
        """运行专注的改进流程"""
        print("\n" + "="*60)
        print("交易信号预测系统 - 专注改进流程")
        print("专注于提高1-4交易信号的预测准确性")
        print("="*60)
        
        success_count = 0
        total_steps = 5
        
        # 1. 分析交易信号质量
        try:
            signal_contexts = self.analyze_trading_signal_quality()
            success_count += 1
        except Exception as e:
            print(f"信号质量分析失败: {e}")
        
        # 2. 创建交易专用特征
        try:
            X, y = self.create_trading_focused_features()
            if len(X) > 0:
                success_count += 1
            else:
                print("特征创建失败: 没有足够的数据")
                return False
        except Exception as e:
            print(f"特征创建失败: {e}")
            return False
        
        # 3. 训练专用模型
        try:
            results = self.train_specialized_models(X, y)
            if results:
                success_count += 1
        except Exception as e:
            print(f"模型训练失败: {e}")
        
        # 4. 创建预测器
        try:
            if self.create_trading_predictor():
                success_count += 1
        except Exception as e:
            print(f"预测器创建失败: {e}")
        
        # 5. 生成改进报告
        try:
            if self.generate_focused_report():
                success_count += 1
        except Exception as e:
            print(f"报告生成失败: {e}")
        
        print(f"\n" + "="*60)
        print(f"专注改进完成: {success_count}/{total_steps} 步骤成功")
        print("="*60)
        
        if success_count >= 4:
            print("\n🎉 交易信号预测系统改进成功！")
            print("\n关键改进:")
            print("1. ✓ 正确理解0信号的含义（等待状态）")
            print("2. ✓ 专注于提高1-4交易信号的预测准确性")
            print("3. ✓ 创建了增强的交易特征")
            print("4. ✓ 训练了专门的交易信号模型")
            print("5. ✓ 提供了专用的预测器")
            
            print("\n下一步建议:")
            print("1. 运行: python trading_signal_predictor.py")
            print("2. 测试新模型的交易信号预测效果")
            print("3. 监控实际交易中的信号质量")
            print("4. 根据实际效果进一步调优参数")
        else:
            print(f"\n⚠️  改进过程中有问题，请检查错误信息")
        
        return success_count >= 4
    
    def generate_focused_report(self):
        """生成专注改进报告"""
        print(f"\n=== 生成专注改进报告 ===")
        
        report = {
            'improvement_date': datetime.now().isoformat(),
            'improvement_philosophy': {
                'understanding': '0信号代表等待状态，这是正常的交易行为',
                'focus': '专注于提高1-4交易信号的预测准确性',
                'goal': '实现每天1-2次高质量的开仓和平仓信号'
            },
            'improvements_applied': [
                '重新定义问题：从信号平衡转向信号质量',
                '创建交易专用特征工程',
                '训练专门的交易信号分类模型',
                '实现高置信度信号过滤',
                '建立交易信号质量评估体系'
            ],
            'technical_enhancements': [
                '增强价格变化特征（多时间窗口）',
                '添加波动率和趋势强度指标',
                '实现技术指标特征（移动平均等）',
                '使用类别平衡的随机森林模型',
                '集成梯度提升模型提高准确性'
            ],
            'expected_benefits': [
                '提高交易信号的预测精确率',
                '减少假信号，提高信号可靠性',
                '更好地捕捉市场转折点',
                '适应股指期货交易的实际需求'
            ],
            'evaluation_metrics': [
                '交易信号精确率（Precision）',
                '交易信号召回率（Recall）',
                'F1得分（平衡精确率和召回率）',
                '各类信号的独立性能评估'
            ],
            'next_steps': [
                '测试新预测器的实际效果',
                '收集更多历史数据进行验证',
                '根据实际交易结果调优模型',
                '建立在线学习机制持续改进'
            ]
        }
        
        report_file = os.path.join(self.base_dir, "focused_improvement_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 专注改进报告已生成: {report_file}")
        return True

if __name__ == "__main__":
    improver = FocusedImprovement()
    improver.run_focused_improvement()