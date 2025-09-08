@echo off
title 无监督学习交易信号识别系统 - 修改后工作流

echo ===============================================================================
echo 无监督学习交易信号识别系统 - 修改后工作流
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

echo [1/5] 正在生成交易信号标签（修改后版本）...
echo ===============================================================================
python src\label_generation.py
if %errorlevel% neq 0 (
    echo 错误: 标签生成失败
    pause
    exit /b %errorlevel%
)
echo ✓ 标签生成完成
echo.

echo [2/5] 正在运行模式识别...
echo ===============================================================================
python src\pattern_recognition.py
if %errorlevel% neq 0 (
    echo 错误: 模式识别失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式识别完成
echo.

echo [3/5] 正在运行模式训练...
echo ===============================================================================
python src\trading_pattern_learning.py
if %errorlevel% neq 0 (
    echo 错误: 模式训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式训练完成
echo.

echo [4/5] 正在训练强化学习模型...
echo ===============================================================================
python src\simple_rl_trader.py
if %errorlevel% neq 0 (
    echo 错误: 强化学习模型训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ 强化学习模型训练完成
echo.

echo [5/5] 正在运行平衡模式预测（集成强化学习优化）...
echo ===============================================================================
python src\pattern_predictor_balanced.py
if %errorlevel% neq 0 (
    echo 错误: 信号预测失败
    pause
    exit /b %errorlevel%
)
echo ✓ 信号预测完成
echo.

echo ===============================================================================
echo 无监督学习交易信号识别系统修改后工作流已完成!
echo ===============================================================================
echo 执行摘要:
echo   - 标签生成: 已完成（使用修改后的标签生成逻辑）
echo   - 模式识别: 已完成
echo   - 模式训练: 已完成
echo   - 强化学习训练: 已完成
echo   - 信号预测: 已完成
echo.
echo 结果文件位置:
echo   - 标签文件: label\*.csv
echo   - 模式文件: patterns\
echo   - 训练模型: model\balanced_model\
echo   - 预测结果: predictions\
echo   - 可视化结果: visualization\
echo.
echo 完成时间: %date% %time%
echo ===============================================================================

:: 询问是否要查看预测结果
echo 是否要查看最新的预测结果？(y/n)
set /p VIEW_RESULTS=
echo.

if /i "%VIEW_RESULTS%"=="y" (
    echo 最新的预测结果:
    echo ===============================================================================
    if exist "predictions\predictions_summary.csv" (
        type "predictions\predictions_summary.csv"
    ) else (
        echo 未找到预测结果文件
    )
    echo.
)

:: 询问是否要打开可视化目录
echo 是否要打开可视化结果目录？(y/n)
set /p OPEN_VISUALIZATION=
echo.

if /i "%OPEN_VISUALIZATION%"=="y" (
    echo 打开可视化结果目录...
    start "" "visualization"
)

pause