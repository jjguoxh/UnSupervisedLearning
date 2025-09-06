@echo off
title 无监督学习交易信号识别系统 - 快速流程

echo ===============================================================================
echo 无监督学习交易信号识别系统 - 快速流程
echo ===============================================================================

echo.
echo 当前时间: %date% %time%
echo.

:: 设置工作目录
cd /d "e:\unsupervised_learning"

:: 检查必要目录
if not exist "data" (
    echo 错误: 未找到数据目录 'data'
    pause
    exit /b 1
)

if not exist "src" (
    echo 错误: 未找到源代码目录 'src'
    pause
    exit /b 1
)

echo ✓ 环境检查通过
echo.

echo [1/4] 正在生成交易信号标签...
echo ===============================================================================
python src\label_generation.py
if %errorlevel% neq 0 (
    echo 错误: 标签生成失败
    pause
    exit /b %errorlevel%
)
echo ✓ 标签生成完成
echo.

echo [2/4] 正在运行模式识别...
echo ===============================================================================
python src\pattern_recognition.py
if %errorlevel% neq 0 (
    echo 错误: 模式识别失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式识别完成
echo.

echo [3/4] 正在运行模式训练...
echo ===============================================================================
python src\trading_pattern_learning.py
if %errorlevel% neq 0 (
    echo 错误: 模式训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式训练完成
echo.

echo [4/4] 正在运行信号预测...
echo ===============================================================================
python src\pattern_predictor.py
if %errorlevel% neq 0 (
    echo 错误: 信号预测失败
    pause
    exit /b %errorlevel%
)
echo ✓ 信号预测完成
echo.

echo ===============================================================================
echo 无监督学习交易信号识别系统快速流程已完成!
echo ===============================================================================
echo 结果文件位置:
echo   - 标签文件: label\*.csv
echo   - 模式文件: patterns\
echo   - 训练模型: patterns\
echo   - 预测结果: predictions\
echo.
echo 完成时间: %date% %time%
echo ===============================================================================

pause