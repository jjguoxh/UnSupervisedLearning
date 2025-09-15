@echo off
title 无监督学习交易信号识别系统 - 完整工作流程
chcp 65001 >nul

echo ===============================================================================
echo 无监督学习交易信号识别系统 - 完整工作流程
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

:: 显示菜单
echo 请选择要执行的工作流程:
echo.
echo 1. 完整工作流程 (推荐) - 执行所有步骤
echo 2. 模式识别流程 - 仅执行模式识别和聚类
echo 3. 预测训练流程 - 基于已有模式训练预测器
echo 4. 信号预测和可视化 - 生成交易信号图表
echo 5. 实时预测模式 - 启动实时预测程序
echo 6. 训练增强版模型 - 训练深度学习增强模型
echo 7. 增强版实时预测 - 使用增强模型进行预测
echo 8. 系统诊断 - 检查系统状态和问题
echo 0. 退出
echo.
set /p CHOICE=请输入选择 (0-8): 

if "%CHOICE%"=="0" goto :EOF
if "%CHOICE%"=="1" goto :FULL_WORKFLOW
if "%CHOICE%"=="2" goto :PATTERN_RECOGNITION
if "%CHOICE%"=="3" goto :PREDICTION_TRAINING
if "%CHOICE%"=="4" goto :SIGNAL_PREDICTION
if "%CHOICE%"=="5" goto :REALTIME_PREDICTION
if "%CHOICE%"=="6" goto :TRAIN_ENHANCED
if "%CHOICE%"=="7" goto :ENHANCED_PREDICTION
if "%CHOICE%"=="8" goto :SYSTEM_DIAGNOSIS

echo 无效选择，请重新运行程序
pause
goto :EOF

:FULL_WORKFLOW
echo ===============================================================================
echo 执行完整工作流程
echo ===============================================================================
echo.

:: 步骤1: 改进的模式识别
echo [步骤1/4] 正在执行改进的模式识别...
echo.
python improved_pattern_recognition.py
if %errorlevel% neq 0 (
    echo 错误: 改进模式识别失败
    pause
    exit /b %errorlevel%
)
echo ✓ 改进模式识别完成
echo.

:: 步骤2: 训练改进的预测器
echo [步骤2/4] 正在训练改进的预测器...
echo.
python improved_pattern_predictor.py
if %errorlevel% neq 0 (
    echo 错误: 改进预测器训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ 改进预测器训练完成
echo.

:: 步骤3: 生成交易信号和可视化
echo [步骤3/4] 正在生成交易信号和可视化...
echo.
python predict_and_visualize.py
if %errorlevel% neq 0 (
    echo 错误: 信号预测和可视化失败
    pause
    exit /b %errorlevel%
)
echo ✓ 信号预测和可视化完成
echo.

:: 步骤4: 生成交易分析报告
echo [步骤4/4] 正在生成交易分析报告...
echo.
if exist "src\generate_trading_analysis.py" (
    python src\generate_trading_analysis.py
    if %errorlevel% neq 0 (
        echo 警告: 交易分析报告生成失败，但继续执行
    ) else (
        echo ✓ 交易分析报告生成完成
    )
) else (
    echo 跳过: 未找到交易分析脚本
)
echo.

goto :WORKFLOW_COMPLETE

:PATTERN_RECOGNITION
echo ===============================================================================
echo 执行模式识别流程
echo ===============================================================================
echo.

echo 正在执行改进的模式识别...
python improved_pattern_recognition.py
if %errorlevel% neq 0 (
    echo 错误: 模式识别失败
    pause
    exit /b %errorlevel%
)
echo ✓ 模式识别完成
echo.

goto :WORKFLOW_COMPLETE

:PREDICTION_TRAINING
echo ===============================================================================
echo 执行预测训练流程
echo ===============================================================================
echo.

echo 正在训练改进的预测器...
python improved_pattern_predictor.py
if %errorlevel% neq 0 (
    echo 错误: 预测器训练失败
    pause
    exit /b %errorlevel%
)
echo ✓ 预测器训练完成
echo.

goto :WORKFLOW_COMPLETE

:SIGNAL_PREDICTION
echo ===============================================================================
echo 执行信号预测和可视化
echo ===============================================================================
echo.

echo 正在生成交易信号和可视化...
python predict_and_visualize.py
if %errorlevel% neq 0 (
    echo 错误: 信号预测失败
    pause
    exit /b %errorlevel%
)
echo ✓ 信号预测和可视化完成
echo.

goto :WORKFLOW_COMPLETE

:REALTIME_PREDICTION
echo ===============================================================================
echo 启动实时预测模式
echo ===============================================================================
echo.

echo 请选择实时预测模式:
echo 1. 改进模型实时预测
echo 2. 强化学习优化预测
echo 3. Transformer模型预测
echo.
set /p RT_CHOICE=请选择 (1-3): 

if "%RT_CHOICE%"=="1" (
    echo 启动改进模型实时预测...
    python src\improved_realtime_predictor.py --mode interactive
) else if "%RT_CHOICE%"=="2" (
    echo 启动强化学习优化预测...
    python src\rl_optimized_realtime_predictor.py --mode interactive
) else if "%RT_CHOICE%"=="3" (
    echo 启动Transformer模型预测...
    python src\transformer_realtime_predictor.py --mode interactive
) else (
    echo 无效选择，启动默认改进模型...
    python src\improved_realtime_predictor.py --mode interactive
)

goto :WORKFLOW_COMPLETE

:TRAIN_ENHANCED
echo ===============================================================================
echo 训练增强版深度学习模型
echo ===============================================================================
echo.

echo 正在训练增强版模型，这可能需要较长时间...
python train_enhanced_model.py
if %errorlevel% neq 0 (
    echo 错误: 增强版模型训练失败
    pause
    exit /b %errorlevel%
) else (
    echo ✓ 增强版模型训练完成！
    echo 模型已保存到 models_enhanced/ 目录
    echo 训练报告已保存到 training_results/ 目录
)
echo.

goto :WORKFLOW_COMPLETE

:ENHANCED_PREDICTION
echo ===============================================================================
echo 增强版实时预测
echo ===============================================================================
echo.

echo 选择预测模式:
echo 1. 交互式预测
echo 2. 目录监控模式
echo 3. 数据模拟模式
echo.
set /p PRED_MODE=请选择模式 (1-3): 

if "%PRED_MODE%"=="1" (
    echo 启动交互式预测模式...
    python enhanced_realtime_predictor.py --mode interactive
) else if "%PRED_MODE%"=="2" (
    echo 启动目录监控模式...
    echo 请将数据文件放入 realtime_data/ 目录
    python enhanced_realtime_predictor.py --mode monitor
) else if "%PRED_MODE%"=="3" (
    echo 启动数据模拟模式...
    python enhanced_realtime_predictor.py --mode simulate
) else (
    echo 无效选择，启动默认交互式模式...
    python enhanced_realtime_predictor.py --mode interactive
)

if %errorlevel% neq 0 (
    echo 错误: 增强版预测失败
    pause
    exit /b %errorlevel%
) else (
    echo ✓ 预测完成！结果已保存到 predictions/ 目录
)
echo.

goto :WORKFLOW_COMPLETE

:SYSTEM_DIAGNOSIS
echo ===============================================================================
echo 执行系统诊断
echo ===============================================================================
echo.

echo 正在执行系统诊断...
if exist "diagnosis_and_improvement.py" (
    python diagnosis_and_improvement.py
) else (
    echo 未找到诊断脚本，执行基本检查...
    echo.
    echo 检查目录结构:
    if exist "data" echo ✓ data目录存在
    if exist "label" echo ✓ label目录存在
    if exist "patterns" echo ✓ patterns目录存在
    if exist "patterns_improved" echo ✓ patterns_improved目录存在
    if exist "model" echo ✓ model目录存在
    if exist "result" echo ✓ result目录存在
    echo.
)

goto :WORKFLOW_COMPLETE

:WORKFLOW_COMPLETE
echo ===============================================================================
echo 工作流程执行完成
echo ===============================================================================
echo.
echo 结果文件位置:
echo - 模式文件: patterns_improved\目录
echo - 模型文件: models_improved\目录
echo - 预测结果: result\目录
echo - 可视化图表: result\目录
echo.
echo 是否要打开结果目录? (y/n)
set /p OPEN_RESULT=
if /i "%OPEN_RESULT%"=="y" (
    start explorer result
)

echo.
echo 感谢使用无监督学习交易信号识别系统!
echo.
pause