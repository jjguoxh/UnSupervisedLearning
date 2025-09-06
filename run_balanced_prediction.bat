@echo off
title 无监督学习交易信号识别系统 - 平衡模式预测

echo ===============================================================================
echo 无监督学习交易信号识别系统 - 平衡模式预测
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

if not exist "label" (
    echo 错误: 未找到标签目录 'label'
    pause
    exit /b 1
)

echo ✓ 环境检查通过
echo.

echo 正在运行平衡模式预测...
echo ===============================================================================
python src\pattern_predictor_balanced.py
if %errorlevel% neq 0 (
    echo 错误: 平衡模式预测运行失败
    pause
    exit /b %errorlevel%
)
echo ✓ 平衡模式预测完成
echo.

echo ===============================================================================
echo 平衡模式预测已完成!
echo ===============================================================================
echo 结果文件位置:
echo   - 预测结果: predictions\
echo   - 可视化结果: visualization\
echo   - 模型文件: model\balanced_model\
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