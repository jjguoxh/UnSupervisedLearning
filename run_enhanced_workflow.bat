@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 增强版深度学习模型完整工作流程
REM 包含训练、评估和预测的完整流程

echo ===============================================================================
echo                    增强版深度学习模型 - 完整工作流程
echo ===============================================================================
echo.
echo 本脚本将执行以下步骤：
echo 1. 检查环境和依赖
echo 2. 训练增强版深度学习模型
echo 3. 模型性能评估
echo 4. 启动实时预测系统
echo.
echo 预计总耗时：10-30分钟（取决于数据量和硬件配置）
echo.
set /p confirm="是否继续执行完整工作流程？(y/n): "
if /i not "%confirm%"=="y" (
    echo 操作已取消
    pause
    exit /b 0
)

REM 设置工作目录
cd /d "%~dp0"

REM 创建必要的目录
echo.
echo [1/4] 检查环境和创建目录...
if not exist "models_enhanced" mkdir "models_enhanced"
if not exist "training_results" mkdir "training_results"
if not exist "predictions" mkdir "predictions"
if not exist "visualization" mkdir "visualization"
if not exist "realtime_data" mkdir "realtime_data"

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python环境
    echo 请确保已安装Python并添加到PATH环境变量
    pause
    exit /b 1
)

REM 检查必要的文件
if not exist "enhanced_deep_learning_predictor.py" (
    echo ❌ 错误: 未找到增强版预测器文件
    pause
    exit /b 1
)

if not exist "train_enhanced_model.py" (
    echo ❌ 错误: 未找到训练脚本文件
    pause
    exit /b 1
)

if not exist "enhanced_realtime_predictor.py" (
    echo ❌ 错误: 未找到实时预测脚本文件
    pause
    exit /b 1
)

REM 检查标签数据
if not exist "label\*.csv" (
    echo ❌ 错误: 未找到训练数据文件
    echo 请确保 label/ 目录中包含CSV格式的标签数据
    pause
    exit /b 1
)

echo ✅ 环境检查完成

REM 步骤2: 训练增强版模型
echo.
echo [2/4] 训练增强版深度学习模型...
echo 这可能需要较长时间，请耐心等待...
echo.

python train_enhanced_model.py
if %errorlevel% neq 0 (
    echo ❌ 错误: 增强版模型训练失败
    echo 请检查训练日志和数据格式
    pause
    exit /b %errorlevel%
)

echo ✅ 增强版模型训练完成

REM 步骤3: 检查训练结果
echo.
echo [3/4] 检查训练结果...

if exist "training_results\enhanced_model_evaluation.json" (
    echo ✅ 找到模型评估结果
) else (
    echo ⚠️  警告: 未找到模型评估结果
)

if exist "training_results\ENHANCED_MODEL_REPORT.md" (
    echo ✅ 找到训练报告
) else (
    echo ⚠️  警告: 未找到训练报告
)

if exist "models_enhanced" (
    echo ✅ 模型文件已保存
) else (
    echo ❌ 错误: 模型文件保存失败
    pause
    exit /b 1
)

REM 步骤4: 选择预测模式
echo.
echo [4/4] 启动预测系统...
echo.
echo 请选择预测模式：
echo 1. 交互式预测 - 手动选择文件进行预测
echo 2. 数据模拟 - 使用历史数据进行模拟预测
echo 3. 目录监控 - 监控realtime_data目录的新文件
echo 4. 跳过预测，查看结果
echo.
set /p pred_choice="请选择 (1-4): "

if "%pred_choice%"=="1" (
    echo 启动交互式预测模式...
    python enhanced_realtime_predictor.py interactive
) else if "%pred_choice%"=="2" (
    echo 启动数据模拟模式...
    python enhanced_realtime_predictor.py simulate
) else if "%pred_choice%"=="3" (
    echo 启动目录监控模式...
    echo 请将数据文件放入 realtime_data/ 目录
    echo 按 Ctrl+C 停止监控
    python enhanced_realtime_predictor.py monitor
) else if "%pred_choice%"=="4" (
    echo 跳过预测阶段
) else (
    echo 无效选择，启动默认交互式模式...
    python enhanced_realtime_predictor.py --mode interactive
)

REM 显示结果总结
echo.
echo ===============================================================================
echo                              工作流程完成
echo ===============================================================================
echo.
echo 📁 结果文件位置：
echo   - 模型文件: models_enhanced/
echo   - 训练报告: training_results/
echo   - 预测结果: predictions/
echo   - 可视化图表: visualization/
echo.
echo 📊 查看结果：
if exist "training_results\ENHANCED_MODEL_REPORT.md" (
    echo   - 训练报告: training_results\ENHANCED_MODEL_REPORT.md
)
if exist "training_results\enhanced_model_evaluation.png" (
    echo   - 性能图表: training_results\enhanced_model_evaluation.png
)
echo.
echo 🚀 后续使用：
echo   - 交互预测: python enhanced_realtime_predictor.py interactive
echo   - 单文件预测: python enhanced_realtime_predictor.py [文件路径]
echo   - 目录监控: python enhanced_realtime_predictor.py monitor
echo   - 重新训练: python train_enhanced_model.py
echo   - 完整流程: run_enhanced_workflow.bat
echo.

REM 询问是否打开结果目录
set /p open_results="是否打开结果目录？(y/n): "
if /i "%open_results%"=="y" (
    if exist "training_results" (
        start "" "training_results"
    )
    if exist "visualization" (
        start "" "visualization"
    )
)

echo.
echo 感谢使用增强版深度学习预测系统！
echo 如有问题，请查看日志文件或联系技术支持。
echo.
pause