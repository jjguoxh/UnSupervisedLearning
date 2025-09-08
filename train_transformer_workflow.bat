@echo off
title 无监督学习交易信号识别系统 - Transformer模型训练工作流

echo ===============================================================================
echo 无监督学习交易信号识别系统 - Transformer模型训练工作流
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

if not exist "data" (
    echo 警告: 未找到数据目录 'data'
    echo 请确保数据文件已放置在data目录中
)

echo ✓ 环境检查通过
echo.

:: 步骤1: 生成交易信号标签
echo [步骤1/4] 正在生成交易信号标签...
echo.
python src\label_generation.py
if %errorlevel% neq 0 (
    echo 错误: 标签生成失败
    pause
    exit /b %errorlevel%
)
echo ✓ 标签生成完成
echo.

:: 步骤2: 运行模式学习
echo [步骤2/4] 正在运行模式学习...
echo.
python src\advanced_pattern_learning.py
if %errorlevel% neq 0 (
    echo 错误: 模式学习失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式学习完成
echo.

:: 步骤3: 训练强化学习模型
echo [步骤3/4] 正在训练强化学习模型...
echo.
python src\simple_rl_trader.py
if %errorlevel% neq 0 (
    echo 警告: 强化学习模型训练失败，将继续执行后续步骤
) else (
    echo ✓ 强化学习模型训练完成
)
echo.

:: 步骤4: 训练Transformer模型
echo [步骤4/4] 正在训练Transformer深度学习模型...
echo.
python train_transformer_model.py
if %errorlevel% neq 0 (
    echo 错误: Transformer模型训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ Transformer模型训练完成
echo.

echo ===============================================================================
echo Transformer模型训练工作流执行完成
echo ===============================================================================
echo 模型文件保存在: model\balanced_model\
echo 标签文件保存在: label\
echo 模式文件保存在: patterns\
echo.

echo 是否要运行Transformer模型的测试? (y/n)
set /p TEST_CHOICE=
if /i "%TEST_CHOICE%"=="y" (
    echo 正在运行Transformer模型测试...
    python -c "from transformer_predictor import test_transformer_model; test_transformer_model()"
    if %errorlevel% neq 0 (
        echo 警告: 模型测试过程中出现错误
    ) else (
        echo ✓ 模型测试完成
    )
)

echo.
echo 是否要启动Transformer模型的交互式预测程序? (y/n)
set /p RUN_CHOICE=
if /i "%RUN_CHOICE%"=="y" (
    echo 启动Transformer模型交互式预测程序...
    python src\transformer_realtime_predictor.py --mode interactive
)

echo.
echo 所有步骤已完成!
echo.
pause