# -*- coding: utf-8 -*-
"""
实时预测程序
实时监控数据流并进行交易信号预测
"""

import pandas as pd
import numpy as np
import os
import time
import json
import threading
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========= 配置参数 =========
MODEL_DIR = "../model/"  # 模型保存目录
REALTIME_DATA_DIR = "../realtime_data/"  # 实时数据目录
PREDICTIONS_DIR = "../predictions/"  # 预测结果目录
PATTERN_LENGTH = 10  # 模式长度
CHECK_INTERVAL = 5  # 检查新数据的间隔（秒）

class RealtimePredictor:
    def __init__(self):
        self.model = None
        self.data_buffer = deque(maxlen=1000)  # 保存最近1000个数据点
        self.last_processed_file = None
        self.load_model()
        self.setup_directories()
        
    def setup_directories(self):
        """
        设置必要的目录
        """
        os.makedirs(REALTIME_DATA_DIR, exist_ok=True)
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        
    def load_model(self):
        """
        加载训练好的模型
        """
        model_path = os.path.join(MODEL_DIR, "pattern_predictor_model.json")
        if not os.path.exists(model_path):
            print("Error: Model file not found! Please run pattern_predictor.py first.")
            return False
            
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.model = model_data
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_realtime_data(self, file_path):
        """
        加载实时数据
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading data file {file_path}: {e}")
            return None
    
    def update_data_buffer(self, df):
        """
        更新数据缓冲区
        """
        for _, row in df.iterrows():
            data_point = {
                'x': row['x'],
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'd': row['d'],
                'index_value': row['index_value'],
                'label': row.get('label', 0)  # 如果没有标签列，则默认为0
            }
            self.data_buffer.append(data_point)
    
    def extract_recent_pattern(self, buffer_data, pattern_length=PATTERN_LENGTH):
        """
        从缓冲区数据中提取最近的模式
        """
        if len(buffer_data) < pattern_length:
            return None
            
        # 取最近的pattern_length个数据点
        recent_data = list(buffer_data)[-pattern_length:]
        
        return {
            'index_value': np.array([point['index_value'] for point in recent_data]),
            'a': np.array([point['a'] for point in recent_data]),
            'b': np.array([point['b'] for point in recent_data]),
            'c': np.array([point['c'] for point in recent_data]),
            'd': np.array([point['d'] for point in recent_data]),
            'x': np.array([point['x'] for point in recent_data])
        }
    
    def calculate_pattern_similarity(self, pattern1, pattern2):
        """
        计算两个模式的相似性
        """
        if len(pattern1) != len(pattern2):
            return 0
            
        # 使用皮尔逊相关系数计算相似性
        try:
            correlation = np.corrcoef(pattern1, pattern2)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0
    
    def predict_signal(self):
        """
        基于当前数据缓冲区进行信号预测
        """
        if not self.model or len(self.data_buffer) < PATTERN_LENGTH:
            return 0, 0.0
        
        # 提取最近的模式
        recent_pattern = self.extract_recent_pattern(self.data_buffer)
        if recent_pattern is None:
            return 0, 0.0
        
        # 计算与各聚类模式的相似性
        best_cluster = None
        best_similarity = -1
        best_signal = 0
        best_confidence = 0
        
        # 遍历所有聚类模型
        for cluster_id, model_info in self.model.get('cluster_models', {}).items():
            try:
                # 获取该聚类的平均模式
                avg_pattern = np.array(model_info['avg_pattern'])
                
                # 计算与该聚类平均模式的相似性
                similarity = self.calculate_pattern_similarity(
                    recent_pattern['index_value'], 
                    avg_pattern
                )
                
                # 如果相似性更高，更新最佳匹配
                if similarity > best_similarity and similarity > 0.7:  # 相似性阈值
                    best_similarity = similarity
                    best_cluster = cluster_id
                    
                    # 根据聚类中最常见的信号类型进行预测
                    patterns_info = self.model.get('patterns_info', {})
                    if str(cluster_id) in patterns_info:
                        cluster_info = patterns_info[str(cluster_id)]
                        signal_counts = cluster_info.get('signal_counts', {})
                        
                        # 预测信号类型（选择最常见的信号）
                        if signal_counts:
                            predicted_signal = max(signal_counts, key=signal_counts.get)
                            best_signal = int(predicted_signal)
                            best_confidence = similarity * cluster_info.get('signal_density', 0)
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
                continue
        
        return best_signal, best_confidence
    
    def get_signal_description(self, signal):
        """
        获取信号描述
        """
        signal_names = {
            0: "无操作",
            1: "做多开仓",  # 包括开仓点和持仓状态
            2: "做多平仓",
            3: "做空开仓",  # 包括开仓点和持仓状态
            4: "做空平仓"
        }
        return signal_names.get(signal, "未知信号")
    
    def save_prediction(self, signal, confidence):
        """
        保存预测结果
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prediction_data = {
            'timestamp': timestamp,
            'predicted_signal': signal,
            'signal_description': self.get_signal_description(signal),
            'confidence': confidence,
            'data_points_in_buffer': len(self.data_buffer)
        }
        
        # 保存为JSON文件
        filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(PREDICTIONS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为CSV格式的汇总文件
        self.update_predictions_csv(prediction_data)
        
        return filepath
    
    def update_predictions_csv(self, prediction_data):
        """
        更新预测结果的CSV汇总文件
        """
        csv_path = os.path.join(PREDICTIONS_DIR, "predictions_summary.csv")
        
        # 创建DataFrame
        df = pd.DataFrame([prediction_data])
        
        # 如果文件存在，追加数据；否则创建新文件
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(csv_path, index=False)
    
    def process_data_file(self, file_path):
        """
        处理单个数据文件
        """
        print(f"Processing data file: {file_path}")
        
        # 加载数据
        df = self.load_realtime_data(file_path)
        if df is None:
            return
        
        # 更新数据缓冲区
        self.update_data_buffer(df)
        
        # 进行预测
        signal, confidence = self.predict_signal()
        
        # 保存预测结果
        filepath = self.save_prediction(signal, confidence)
        
        # 打印预测结果
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: "
              f"{self.get_signal_description(signal)} ({signal}), "
              f"Confidence: {confidence:.3f}")
        print(f"Prediction saved to: {filepath}")
        
        # 更新最后处理的文件
        self.last_processed_file = file_path
    
    def monitor_data_directory(self):
        """
        监控数据目录中的新文件
        """
        print(f"Monitoring {REALTIME_DATA_DIR} for new data files...")
        
        processed_files = set()
        
        while True:
            try:
                # 获取目录中的所有CSV文件
                csv_files = glob.glob(os.path.join(REALTIME_DATA_DIR, "*.csv"))
                
                # 处理新文件
                for file_path in csv_files:
                    if file_path not in processed_files:
                        self.process_data_file(file_path)
                        processed_files.add(file_path)
                
                # 等待一段时间后再次检查
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(CHECK_INTERVAL)
    
    def simulate_realtime_data(self, base_file_path, interval=2):
        """
        模拟实时数据流（用于演示）
        """
        print("Simulating real-time data stream...")
        
        # 加载基础数据
        try:
            base_df = pd.read_csv(base_file_path)
        except Exception as e:
            print(f"Error loading base data: {e}")
            return
        
        # 逐行添加数据到缓冲区并进行预测
        for i in range(0, len(base_df), 5):  # 每次处理5行数据
            # 获取一小段数据
            chunk = base_df.iloc[i:min(i+5, len(base_df))]
            
            # 更新数据缓冲区
            self.update_data_buffer(chunk)
            
            # 进行预测
            signal, confidence = self.predict_signal()
            
            # 保存预测结果
            filepath = self.save_prediction(signal, confidence)
            
            # 打印预测结果
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Prediction: "
                  f"{self.get_signal_description(signal)} ({signal}), "
                  f"Confidence: {confidence:.3f}")
            
            # 等待
            time.sleep(interval)
    
    def run_interactive_mode(self):
        """
        运行交互模式
        """
        print("Real-time Pattern Predictor - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  'predict' - Make a prediction based on current buffer")
        print("  'buffer' - Show buffer status")
        print("  'last' - Show last prediction")
        print("  'quit' - Exit the program")
        print("=" * 50)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    print("Exiting...")
                    break
                elif command == 'predict':
                    signal, confidence = self.predict_signal()
                    filepath = self.save_prediction(signal, confidence)
                    print(f"Prediction: {self.get_signal_description(signal)} ({signal}), "
                          f"Confidence: {confidence:.3f}")
                    print(f"Saved to: {filepath}")
                elif command == 'buffer':
                    print(f"Data points in buffer: {len(self.data_buffer)}")
                    if self.data_buffer:
                        print("Recent data points:")
                        for i, point in enumerate(list(self.data_buffer)[-5:]):
                            print(f"  {len(self.data_buffer)-5+i}: x={point['x']}, "
                                  f"index_value={point['index_value']}, label={point['label']}")
                elif command == 'last':
                    # 显示最后一次预测
                    csv_path = os.path.join(PREDICTIONS_DIR, "predictions_summary.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            last_pred = df.iloc[-1]
                            print(f"Last prediction: {last_pred['timestamp']}")
                            print(f"  Signal: {last_pred['signal_description']} ({last_pred['predicted_signal']})")
                            print(f"  Confidence: {last_pred['confidence']:.3f}")
                        else:
                            print("No predictions yet.")
                    else:
                        print("No predictions yet.")
                else:
                    print("Unknown command. Available commands: predict, buffer, last, quit")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """
    主函数
    """
    # 创建实时预测器
    predictor = RealtimePredictor()
    
    if not predictor.model:
        print("Failed to load model. Exiting.")
        return
    
    print("Real-time Pattern Predictor")
    print("=" * 30)
    print("Select mode:")
    print("1. Monitor directory for new files")
    print("2. Simulate real-time data stream")
    print("3. Interactive mode")
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            predictor.monitor_data_directory()
        elif choice == '2':
            # 获取一个示例数据文件用于模拟
            label_files = glob.glob("../label/*.csv")
            if label_files:
                predictor.simulate_realtime_data(label_files[0])
            else:
                print("No sample data files found for simulation.")
        elif choice == '3':
            predictor.run_interactive_mode()
        else:
            print("Invalid choice. Running in interactive mode...")
            predictor.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import glob
    main()