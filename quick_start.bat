@echo off
title 交易信号预测系统 - 快速启动
chcp 65001 >nul

echo ===============================================================================
echo 🚀 交易信号预测系统 - 快速启动
echo ===============================================================================
echo.

:: 设置工作目录
cd /d "e:\unsupervised_learning"

echo 请选择快速操作:
echo.
echo 1. 📊 生成交易信号图表 (推荐日常使用)
echo 2. 🔄 重新训练模型 (数据更新后使用)
echo 3. 🎯 实时预测模式
echo 4. 📈 查看历史结果
echo 5. 🚀 训练增强版模型
echo 6. 🔮 增强版预测
echo 0. 退出
echo.
set /p CHOICE=请选择 (0-6): 

if "%CHOICE%"=="0" goto :EOF
if "%CHOICE%"=="1" goto :GENERATE_SIGNALS
if "%CHOICE%"=="2" goto :RETRAIN_MODEL
if "%CHOICE%"=="3" goto :REALTIME_MODE
if "%CHOICE%"=="4" goto :VIEW_RESULTS
if "%CHOICE%"=="5" goto :TRAIN_ENHANCED
if "%CHOICE%"=="6" goto :ENHANCED_PREDICTION

echo 无效选择
pause
goto :EOF

:GENERATE_SIGNALS
echo.
echo 🔄 正在生成交易信号图表...
echo 这将处理data目录中的所有数据文件
echo.
python predict_and_visualize.py
if %errorlevel% neq 0 (
    echo ❌ 信号生成失败
    pause
    exit /b %errorlevel%
)
echo.
echo ✅ 信号图表生成完成！
echo 📁 结果已保存到 result\ 目录
echo.
echo 是否要打开结果目录查看图表? (y/n)
set /p OPEN_RESULT=
if /i "%OPEN_RESULT%"=="y" (
    start explorer result
)
goto :END

:RETRAIN_MODEL
echo.
echo 🔄 正在重新训练模型...
echo 这可能需要几分钟时间
echo.
echo [1/2] 执行模式识别...
python improved_pattern_recognition.py
if %errorlevel% neq 0 (
    echo ❌ 模式识别失败
    pause
    exit /b %errorlevel%
)
echo ✅ 模式识别完成
echo.
echo [2/2] 训练预测器...
python improved_pattern_predictor.py
if %errorlevel% neq 0 (
    echo ❌ 预测器训练失败
    pause
    exit /b %errorlevel%
)
echo.
echo ✅ 模型训练完成！
echo 💡 现在可以使用选项1生成新的交易信号
goto :END

:REALTIME_MODE
echo.
echo 🎯 启动实时预测模式...
echo.
python src\improved_realtime_predictor.py --mode interactive
goto :END

:VIEW_RESULTS
echo.
echo 📈 打开历史结果目录...
if exist "result" (
    start explorer result
    echo ✅ 结果目录已打开
) else (
    echo ❌ 未找到结果目录，请先生成交易信号
)
goto :END

:TRAIN_ENHANCED
echo.
echo 🚀 正在训练增强版深度学习模型...
echo 这可能需要较长时间
echo.
python train_enhanced_model.py
if %errorlevel% neq 0 (
    echo ❌ 增强版模型训练失败
    pause
    exit /b %errorlevel%
)
echo.
echo ✅ 增强版模型训练完成！
goto :END

:ENHANCED_PREDICTION
echo.
echo 🔮 启动增强版预测...
echo 这将使用训练好的增强版深度学习模型进行预测
echo.
echo 请选择预测模式:
echo 1. 交互模式（手动选择文件）
echo 2. 批量预测（处理predict目录所有文件）
echo 3. 单文件预测
echo.
set /p pred_choice=请选择 (1-3): 

if "%pred_choice%"=="1" (
    python enhanced_realtime_predictor.py interactive
) else if "%pred_choice%"=="2" (
    for %%f in (predict\*.csv) do (
        echo 正在预测: %%f
        python enhanced_realtime_predictor.py "%%f"
    )
) else if "%pred_choice%"=="3" (
    set /p single_file=请输入文件路径: 
    python enhanced_realtime_predictor.py "!single_file!"
) else (
    echo 无效选择
    pause
    goto :EOF
)

if %errorlevel% neq 0 (
    echo ❌ 增强版预测失败
    pause
    exit /b %errorlevel%
)
echo.
echo ✅ 增强版预测完成！
echo 📁 结果已保存到 predictions\ 目录
echo 📊 可视化图表已保存到 visualization\ 目录
start explorer visualization
pause
goto :EOF

:END
echo.
echo 操作完成！
echo.
pause