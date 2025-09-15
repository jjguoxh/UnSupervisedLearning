
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的训练管道
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json

def load_balanced_data():
    """加载平衡后的数据"""
    balanced_dir = "balanced_data"
    
    with open(os.path.join(balanced_dir, "balanced_features.pkl"), 'rb') as f:
        X = pickle.load(f)
    
    with open(os.path.join(balanced_dir, "balanced_labels.pkl"), 'rb') as f:
        y = pickle.load(f)
    
    return X, y

def train_improved_models():
    """训练改进的模型"""
    print("加载平衡数据...")
    X, y = load_balanced_data()
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"
训练 {name} 模型...")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"交叉验证得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 测试预测
        y_pred = model.predict(X_test)
        
        # 评估
        print(f"
{name} 分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        model_dir = "model/improved"
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, f"{name.lower()}_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        scaler_file = os.path.join(model_dir, f"{name.lower()}_scaler.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        results[name] = {
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    # 保存结果
    results_file = "model/improved/training_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"
✓ 训练完成，结果保存到 {results_file}")

if __name__ == "__main__":
    train_improved_models()
