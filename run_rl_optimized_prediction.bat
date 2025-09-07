@echo off
echo ==================================================
echo  UnSupervisedLearning 强化学习优化预测程序
echo ==================================================
echo.

cd /d "%~dp0"

echo 启动强化学习优化的实时预测程序...
echo.

python src/rl_optimized_realtime_predictor.py --mode interactive

echo.
echo 程序执行完成。
pause