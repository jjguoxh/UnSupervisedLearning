@echo off
echo ==================================================
echo  UnSupervisedLearning 强化学习优化预测程序
echo ==================================================
echo.

cd /d "%~dp0"

echo 启动强化学习优化的实时预测程序...
echo.

echo 请选择运行模式:
echo 1. 目录监控模式（监控realtime_data目录中的新文件）
echo 2. 数据模拟模式（模拟实时数据流）
echo 3. 交互模式（手动控制预测过程）
echo.

choice /c 123 /m "请选择模式"
if %errorlevel%==1 (
    echo 启动目录监控模式...
    python src/rl_optimized_realtime_predictor.py --mode monitor
) else if %errorlevel%==2 (
    echo 启动数据模拟模式...
    python src/rl_optimized_realtime_predictor.py --mode simulate
) else if %errorlevel%==3 (
    echo 启动交互模式...
    python src/rl_optimized_realtime_predictor.py --mode interactive
)

if %errorlevel% neq 0 (
    echo 错误: 强化学习优化预测程序运行失败
    pause
    exit /b %errorlevel%
)

echo.
echo 程序执行完成。
echo 预测结果保存在: predictions\
echo 可视化结果保存在: visualization\
echo.
pause