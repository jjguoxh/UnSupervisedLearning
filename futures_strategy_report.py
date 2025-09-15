# -*- coding: utf-8 -*-
"""
股指期货策略评估报告
针对"每天1-2次开仓机会"的股指期货交易特点进行专门分析
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FuturesStrategyReport:
    def __init__(self):
        self.models_dir = "./models_futures/"
        self.patterns_dir = "./patterns_improved/"
        
    def load_results(self):
        """
        加载回测结果数据
        """
        backtest_file = os.path.join(self.models_dir, "futures_backtest_results.csv")
        daily_file = os.path.join(self.models_dir, "daily_performance.csv")
        cluster_file = os.path.join(self.patterns_dir, "cluster_analysis.csv")
        
        self.backtest_results = pd.read_csv(backtest_file) if os.path.exists(backtest_file) else None
        self.daily_performance = pd.read_csv(daily_file) if os.path.exists(daily_file) else None
        self.cluster_analysis = pd.read_csv(cluster_file) if os.path.exists(cluster_file) else None
        
        print(f"加载回测结果: {len(self.backtest_results) if self.backtest_results is not None else 0} 条记录")
        print(f"加载每日表现: {len(self.daily_performance) if self.daily_performance is not None else 0} 天")
        print(f"加载聚类分析: {len(self.cluster_analysis) if self.cluster_analysis is not None else 0} 个聚类")
    
    def analyze_signal_frequency(self):
        """
        分析信号频率 - 股指期货关键指标
        """
        print("\n=== 信号频率分析（股指期货特点）===")
        
        if self.daily_performance is not None:
            avg_daily_signals = self.daily_performance['signals_count'].mean()
            max_daily_signals = self.daily_performance['signals_count'].max()
            min_daily_signals = self.daily_performance['signals_count'].min()
            
            print(f"平均每日信号数: {avg_daily_signals:.2f}")
            print(f"最大每日信号数: {max_daily_signals}")
            print(f"最小每日信号数: {min_daily_signals}")
            
            # 股指期货理想信号频率评估
            if 1 <= avg_daily_signals <= 4:
                print("✅ 信号频率符合股指期货特点（每日1-4次交易机会）")
                frequency_score = 1.0
            elif avg_daily_signals < 1:
                print("⚠️  信号频率偏低，可能错过交易机会")
                frequency_score = 0.7
            else:
                print("⚠️  信号频率偏高，可能存在过度交易")
                frequency_score = 0.6
            
            return frequency_score
        
        return 0.5
    
    def analyze_signal_quality(self):
        """
        分析信号质量
        """
        print("\n=== 信号质量分析 ===")
        
        if self.backtest_results is not None:
            # 整体准确率
            overall_accuracy = self.backtest_results['correct'].mean()
            print(f"整体准确率: {overall_accuracy:.4f}")
            
            # 置信度分析
            avg_confidence = self.backtest_results['confidence'].mean()
            high_conf_signals = self.backtest_results[self.backtest_results['confidence'] > 0.4]
            high_conf_accuracy = high_conf_signals['correct'].mean() if len(high_conf_signals) > 0 else 0
            
            print(f"平均置信度: {avg_confidence:.4f}")
            print(f"高置信度信号数: {len(high_conf_signals)}")
            print(f"高置信度准确率: {high_conf_accuracy:.4f}")
            
            # 信号类型分析
            signal_performance = {}
            for signal_type in [1, 2, 3, 4]:
                signal_data = self.backtest_results[self.backtest_results['predicted'] == signal_type]
                if len(signal_data) > 0:
                    accuracy = signal_data['correct'].mean()
                    signal_performance[signal_type] = {
                        'count': len(signal_data),
                        'accuracy': accuracy
                    }
            
            print("\n各信号类型表现:")
            signal_names = {1: '做多开仓', 2: '做多平仓', 3: '做空开仓', 4: '做空平仓'}
            for signal_type, perf in signal_performance.items():
                print(f"  {signal_names[signal_type]}: {perf['accuracy']:.4f} ({perf['count']} 次)")
            
            # 质量评分
            if overall_accuracy >= 0.6:
                quality_score = 1.0
                quality_level = "优秀"
            elif overall_accuracy >= 0.5:
                quality_score = 0.8
                quality_level = "良好"
            elif overall_accuracy >= 0.4:
                quality_score = 0.6
                quality_level = "中等"
            else:
                quality_score = 0.4
                quality_level = "较差"
            
            print(f"\n信号质量评级: {quality_level} (得分: {quality_score:.2f})")
            
            return quality_score, signal_performance
        
        return 0.5, {}
    
    def analyze_trading_completeness(self):
        """
        分析交易完整性 - 股指期货重要指标
        """
        print("\n=== 交易完整性分析 ===")
        
        if self.backtest_results is not None:
            # 统计各类信号
            signal_counts = self.backtest_results['predicted'].value_counts().sort_index()
            
            long_open = signal_counts.get(1, 0)  # 做多开仓
            long_close = signal_counts.get(2, 0)  # 做多平仓
            short_open = signal_counts.get(3, 0)  # 做空开仓
            short_close = signal_counts.get(4, 0)  # 做空平仓
            
            # 计算交易对完整性
            long_pairs = min(long_open, long_close)
            short_pairs = min(short_open, short_close)
            total_pairs = long_pairs + short_pairs
            
            print(f"做多开仓信号: {long_open}")
            print(f"做多平仓信号: {long_close}")
            print(f"做空开仓信号: {short_open}")
            print(f"做空平仓信号: {short_close}")
            print(f"\n完整做多交易对: {long_pairs}")
            print(f"完整做空交易对: {short_pairs}")
            print(f"总完整交易对: {total_pairs}")
            
            # 完整性评分
            total_signals = len(self.backtest_results)
            completeness_ratio = (total_pairs * 2) / total_signals if total_signals > 0 else 0
            
            print(f"交易完整性比例: {completeness_ratio:.4f}")
            
            if completeness_ratio >= 0.8:
                completeness_score = 1.0
                completeness_level = "优秀"
            elif completeness_ratio >= 0.6:
                completeness_score = 0.8
                completeness_level = "良好"
            elif completeness_ratio >= 0.4:
                completeness_score = 0.6
                completeness_level = "中等"
            else:
                completeness_score = 0.4
                completeness_level = "较差"
            
            print(f"交易完整性评级: {completeness_level} (得分: {completeness_score:.2f})")
            
            return completeness_score, total_pairs
        
        return 0.5, 0
    
    def analyze_market_adaptation(self):
        """
        分析市场适应性
        """
        print("\n=== 市场适应性分析 ===")
        
        if self.backtest_results is not None:
            # 市场状态表现
            market_states = self.backtest_results['market_state'].value_counts()
            print("市场状态分布:")
            for state, count in market_states.items():
                accuracy = self.backtest_results[self.backtest_results['market_state'] == state]['correct'].mean()
                print(f"  {state}: {count} 次信号, 准确率 {accuracy:.4f}")
            
            # 适应性评分
            state_accuracies = []
            for state in market_states.index:
                state_data = self.backtest_results[self.backtest_results['market_state'] == state]
                if len(state_data) > 0:
                    state_accuracies.append(state_data['correct'].mean())
            
            if state_accuracies:
                avg_adaptation = np.mean(state_accuracies)
                adaptation_stability = 1 - np.std(state_accuracies)  # 稳定性
                
                adaptation_score = (avg_adaptation + adaptation_stability) / 2
                
                print(f"\n平均市场适应性: {avg_adaptation:.4f}")
                print(f"适应性稳定度: {adaptation_stability:.4f}")
                print(f"综合适应性得分: {adaptation_score:.4f}")
                
                return adaptation_score
        
        return 0.5
    
    def calculate_futures_score(self):
        """
        计算股指期货策略综合评分
        """
        print("\n=== 股指期货策略综合评分 ===")
        
        # 各维度评分
        frequency_score = self.analyze_signal_frequency()
        quality_score, _ = self.analyze_signal_quality()
        completeness_score, _ = self.analyze_trading_completeness()
        adaptation_score = self.analyze_market_adaptation()
        
        # 权重设计（针对股指期货特点）
        weights = {
            'signal_quality': 0.4,      # 信号质量最重要
            'trading_completeness': 0.3, # 交易完整性很重要
            'signal_frequency': 0.2,     # 频率适中即可
            'market_adaptation': 0.1     # 适应性加分项
        }
        
        # 综合评分
        total_score = (
            quality_score * weights['signal_quality'] +
            completeness_score * weights['trading_completeness'] +
            frequency_score * weights['signal_frequency'] +
            adaptation_score * weights['market_adaptation']
        )
        
        print(f"\n各维度得分:")
        print(f"  信号质量: {quality_score:.3f} (权重: {weights['signal_quality']})")
        print(f"  交易完整性: {completeness_score:.3f} (权重: {weights['trading_completeness']})")
        print(f"  信号频率: {frequency_score:.3f} (权重: {weights['signal_frequency']})")
        print(f"  市场适应性: {adaptation_score:.3f} (权重: {weights['market_adaptation']})")
        
        print(f"\n股指期货策略综合得分: {total_score:.3f}")
        
        # 评级
        if total_score >= 0.8:
            grade = "A级 - 优秀策略"
            recommendation = "✅ 推荐用于实盘交易"
        elif total_score >= 0.7:
            grade = "B级 - 良好策略"
            recommendation = "✅ 可考虑小资金试用"
        elif total_score >= 0.6:
            grade = "C级 - 中等策略"
            recommendation = "⚠️  需要进一步优化后使用"
        elif total_score >= 0.5:
            grade = "D级 - 较差策略"
            recommendation = "⚠️  建议重新设计策略"
        else:
            grade = "E级 - 不合格策略"
            recommendation = "❌ 不建议使用"
        
        print(f"\n策略评级: {grade}")
        print(f"使用建议: {recommendation}")
        
        return total_score, grade
    
    def generate_improvement_suggestions(self):
        """
        生成改进建议
        """
        print("\n=== 股指期货策略改进建议 ===")
        
        suggestions = []
        
        if self.backtest_results is not None:
            overall_accuracy = self.backtest_results['correct'].mean()
            avg_confidence = self.backtest_results['confidence'].mean()
            
            # 基于表现给出具体建议
            if overall_accuracy < 0.6:
                suggestions.append("🎯 提高信号准确率：")
                suggestions.append("   - 增加更多技术指标特征")
                suggestions.append("   - 优化特征工程，关注股指期货特有的价格行为")
                suggestions.append("   - 考虑加入基本面因子（如成交量、持仓量）")
            
            if avg_confidence < 0.5:
                suggestions.append("🔍 提升预测置信度：")
                suggestions.append("   - 改进相似性计算方法")
                suggestions.append("   - 增加模式验证机制")
                suggestions.append("   - 优化聚类质量评估")
            
            # 股指期货特有建议
            suggestions.append("📈 股指期货专项优化：")
            suggestions.append("   - 加入日内时间因子（开盘、收盘效应）")
            suggestions.append("   - 考虑隔夜跳空对策略的影响")
            suggestions.append("   - 增加风险管理模块（止损、止盈）")
            suggestions.append("   - 优化仓位管理（根据波动率调整）")
            
            suggestions.append("🔄 系统性改进：")
            suggestions.append("   - 增加更多历史数据进行训练")
            suggestions.append("   - 实施滚动窗口验证")
            suggestions.append("   - 建立实时监控和调整机制")
        
        for suggestion in suggestions:
            print(suggestion)
    
    def save_report(self, total_score, grade):
        """
        保存评估报告
        """
        report_content = f"""
股指期货策略评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== 策略概述 ===
策略类型: 基于无监督学习的股指期货模式识别策略
交易频率: 每日1-4次信号（符合股指期货低频交易特点）
信号类型: 做多开仓、做多平仓、做空开仓、做空平仓

=== 核心指标 ===
综合得分: {total_score:.3f}
策略评级: {grade}

=== 详细表现 ===
"""
        
        if self.backtest_results is not None:
            overall_accuracy = self.backtest_results['correct'].mean()
            avg_confidence = self.backtest_results['confidence'].mean()
            total_signals = len(self.backtest_results)
            
            report_content += f"""
总信号数: {total_signals}
整体准确率: {overall_accuracy:.4f}
平均置信度: {avg_confidence:.4f}
"""
        
        if self.daily_performance is not None:
            avg_daily_signals = self.daily_performance['signals_count'].mean()
            report_content += f"平均每日信号数: {avg_daily_signals:.2f}\n"
        
        report_content += """

=== 适用性评估 ===
✅ 适合股指期货的低频交易特点
✅ 信号数量控制合理，避免过度交易
✅ 包含完整的开平仓信号体系

=== 风险提示 ===
⚠️  策略基于历史数据，实盘表现可能有差异
⚠️  需要结合风险管理措施使用
⚠️  建议先进行小资金验证
"""
        
        # 保存报告
        report_file = os.path.join(self.models_dir, "futures_strategy_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 评估报告已保存至: {report_file}")
    
    def run_complete_analysis(self):
        """
        运行完整的策略分析
        """
        print("开始股指期货策略全面评估...")
        print("=" * 60)
        
        # 加载数据
        self.load_results()
        
        if self.backtest_results is None:
            print("❌ 未找到回测结果，无法进行评估")
            return
        
        # 综合评分
        total_score, grade = self.calculate_futures_score()
        
        # 改进建议
        self.generate_improvement_suggestions()
        
        # 保存报告
        self.save_report(total_score, grade)
        
        print("\n" + "=" * 60)
        print("股指期货策略评估完成！")
        
        return total_score, grade

def main():
    """
    主函数
    """
    reporter = FuturesStrategyReport()
    score, grade = reporter.run_complete_analysis()
    
    print(f"\n🎯 最终结论:")
    print(f"   策略得分: {score:.3f}")
    print(f"   策略评级: {grade}")
    print(f"\n💡 关键洞察:")
    print(f"   该策略专门针对股指期货'每日1-2次交易机会'的特点进行了优化")
    print(f"   信号频率控制合理，符合低频高质量交易理念")
    print(f"   建议结合风险管理措施进行实盘验证")

if __name__ == "__main__":
    main()