@echo off
title 无监督学习交易信号识别系统 - 实时预测

echo ===============================================================================
echo 无监督学习交易信号识别系统 - 实时预测
echo ===============================================================================

echo.
echo 当前时间: %date% %time%
echo.

:: 设置工作目录
cd /d "e:\unsupervised_learning"

:: 检查必要目录
if not exist "src" (
    echo 错误: 未找到源代码目录 'src'
    pause
    exit /b 1
)

if not exist "model" (
    echo 警告: 未找到模型目录 'model'，请确保已运行训练流程
)

echo ✓ 环境检查通过
echo.

echo 启动增强版实时预测程序...
echo 请选择运行模式:
echo 1. 目录监控模式（监控realtime_data目录中的新文件）
echo 2. 数据模拟模式（模拟实时数据流）
echo 3. 交互模式（手动控制预测过程）
echo 4. 单文件预测模式（预测指定文件）
echo.

choice /c 1234 /m "请选择模式"
if %errorlevel%==1 (
    echo 启动目录监控模式...
    python enhanced_realtime_predictor.py monitor
) else if %errorlevel%==2 (
    echo 启动数据模拟模式...
    python enhanced_realtime_predictor.py simulate
) else if %errorlevel%==3 (
    echo 启动交互模式...
    python enhanced_realtime_predictor.py interactive
) else if %errorlevel%==4 (
    echo 启动单文件预测模式...
    set /p filepath="请输入文件路径（如 predict/240110.csv）: "
    python enhanced_realtime_predictor.py !filepath!
)

if %errorlevel% neq 0 (
    echo 错误: 实时预测程序运行失败
    pause
    exit /b %errorlevel%
)

echo.
echo ===============================================================================
echo 实时预测程序已退出
echo ===============================================================================
echo 预测结果保存在: predictions\
echo 可视化结果保存在: visualization\
echo.

pause