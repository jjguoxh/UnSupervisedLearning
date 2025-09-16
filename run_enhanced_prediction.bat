@echo off
title 增强版深度学习预测系统
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===============================================================================
echo 🔮 增强版深度学习预测系统
echo ===============================================================================
echo.
echo 本系统使用增强版Transformer模型进行交易信号预测
echo 支持多种预测模式，已修复JSON序列化和图像尺寸问题
echo.

:: 设置工作目录
cd /d "%~dp0"

:: 检查必要文件
if not exist "enhanced_realtime_predictor.py" (
    echo ❌ 错误: 未找到增强版预测器文件
    echo 请确保 enhanced_realtime_predictor.py 存在
    pause
    exit /b 1
)

if not exist "models_enhanced" (
    echo ❌ 错误: 未找到增强版模型目录
    echo 请先运行训练流程: python train_enhanced_model.py
    pause
    exit /b 1
)

:: 创建必要目录
if not exist "predictions" mkdir "predictions"
if not exist "visualization" mkdir "visualization"
if not exist "realtime_data" mkdir "realtime_data"

echo ✅ 环境检查通过
echo.

echo 请选择预测模式:
echo.
echo 1. 📁 单文件预测 - 预测指定的CSV文件
echo 2. 📂 批量预测 - 预测predict目录下所有文件
echo 3. 🔄 目录监控 - 实时监控realtime_data目录
echo 4. 🎮 交互模式 - 手动控制预测过程
echo 5. 🧪 模拟模式 - 模拟实时数据流
echo 0. 🚪 退出
echo.
set /p choice=请选择模式 (0-5): 

if "%choice%"=="0" goto :EOF
if "%choice%"=="1" goto :SINGLE_FILE
if "%choice%"=="2" goto :BATCH_PREDICT
if "%choice%"=="3" goto :MONITOR_MODE
if "%choice%"=="4" goto :INTERACTIVE_MODE
if "%choice%"=="5" goto :SIMULATE_MODE

echo ❌ 无效选择
pause
goto :EOF

:SINGLE_FILE
echo.
echo 📁 单文件预测模式
echo.
set /p filepath=请输入文件路径（如 predict/240110.csv）: 
if "%filepath%"=="" (
    echo ❌ 文件路径不能为空
    pause
    goto :EOF
)

echo.
echo 🔄 正在预测文件: %filepath%
echo.
python enhanced_realtime_predictor.py "%filepath%"
goto :SHOW_RESULTS

:BATCH_PREDICT
echo.
echo 📂 批量预测模式
echo 正在处理predict目录下的所有CSV文件...
echo.

set count=0
for %%f in (predict\*.csv) do (
    set /a count+=1
    echo [!count!] 正在预测: %%f
    python enhanced_realtime_predictor.py "%%f"
    if !errorlevel! neq 0 (
        echo ❌ 预测失败: %%f
    ) else (
        echo ✅ 预测完成: %%f
    )
    echo.
)

if %count%==0 (
    echo ❌ 未找到CSV文件在predict目录
    echo 请确保predict目录存在且包含CSV文件
    pause
    goto :EOF
)

echo ✅ 批量预测完成，共处理 %count% 个文件
goto :SHOW_RESULTS

:MONITOR_MODE
echo.
echo 🔄 目录监控模式
echo 正在监控realtime_data目录，等待新文件...
echo 按Ctrl+C停止监控
echo.
python enhanced_realtime_predictor.py monitor
goto :SHOW_RESULTS

:INTERACTIVE_MODE
echo.
echo 🎮 交互模式
echo 启动交互式预测界面...
echo.
python enhanced_realtime_predictor.py interactive
goto :SHOW_RESULTS

:SIMULATE_MODE
echo.
echo 🧪 模拟模式
echo 启动实时数据流模拟...
echo.
python enhanced_realtime_predictor.py simulate
goto :SHOW_RESULTS

:SHOW_RESULTS
if %errorlevel% neq 0 (
    echo.
    echo ❌ 预测过程中出现错误
    echo 错误代码: %errorlevel%
    echo.
    echo 💡 常见问题解决方案:
    echo   - 检查文件路径是否正确
    echo   - 确保CSV文件格式正确
    echo   - 验证模型文件是否存在
    echo   - 查看错误日志获取详细信息
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo ===============================================================================
echo ✅ 预测完成！
echo ===============================================================================
echo.
echo 📁 结果文件位置:
echo   - JSON结果: predictions\ 目录
echo   - 可视化图表: visualization\ 目录
echo.
echo 📊 查看结果:
set /p open_results=是否打开结果目录？(y/n): 
if /i "%open_results%"=="y" (
    if exist "predictions" (
        echo 📂 打开预测结果目录...
        start "" "predictions"
    )
    if exist "visualization" (
        echo 📊 打开可视化目录...
        start "" "visualization"
    )
)

echo.
echo 🎉 感谢使用增强版深度学习预测系统！
echo 💡 提示: 可以重复运行此脚本进行更多预测
echo.
pause