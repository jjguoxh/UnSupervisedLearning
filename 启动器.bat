@echo off
title 交易信号识别系统 - 启动器
chcp 65001 >nul

echo.
echo ===============================================================================
echo 🎯 无监督学习交易信号识别系统
echo ===============================================================================
echo.
echo 欢迎使用交易信号识别系统！
echo.
echo 请选择启动方式:
echo.
echo 1. 🚀 快速启动 (推荐) - 日常使用，快速生成交易信号
echo 2. 🔧 完整工作流程 - 高级用户，包含所有功能选项
echo 3. 📖 查看使用指南
echo 4. 📁 打开项目目录
echo 0. 退出
echo.
set /p CHOICE=请选择 (0-4): 

if "%CHOICE%"=="0" goto :EOF
if "%CHOICE%"=="1" goto :QUICK_START
if "%CHOICE%"=="2" goto :COMPLETE_WORKFLOW
if "%CHOICE%"=="3" goto :VIEW_GUIDE
if "%CHOICE%"=="4" goto :OPEN_DIRECTORY

echo 无效选择，请重新运行
pause
goto :EOF

:QUICK_START
echo.
echo 🚀 启动快速模式...
echo.
call quick_start.bat
goto :EOF

:COMPLETE_WORKFLOW
echo.
echo 🔧 启动完整工作流程...
echo.
call complete_workflow.bat
goto :EOF

:VIEW_GUIDE
echo.
echo 📖 打开使用指南...
if exist "WORKFLOW_GUIDE.md" (
    start notepad WORKFLOW_GUIDE.md
) else (
    echo 未找到使用指南文件
)
echo.
echo 按任意键返回主菜单...
pause >nul
cls
goto :START

:OPEN_DIRECTORY
echo.
echo 📁 打开项目目录...
start explorer .
echo.
echo 按任意键返回主菜单...
pause >nul
cls
goto :START

:START
echo.
echo ===============================================================================
echo 🎯 无监督学习交易信号识别系统
echo ===============================================================================
echo.
echo 欢迎使用交易信号识别系统！
echo.
echo 请选择启动方式:
echo.
echo 1. 🚀 快速启动 (推荐) - 日常使用，快速生成交易信号
echo 2. 🔧 完整工作流程 - 高级用户，包含所有功能选项
echo 3. 📖 查看使用指南
echo 4. 📁 打开项目目录
echo 0. 退出
echo.
set /p CHOICE=请选择 (0-4): 

if "%CHOICE%"=="0" goto :EOF
if "%CHOICE%"=="1" goto :QUICK_START
if "%CHOICE%"=="2" goto :COMPLETE_WORKFLOW
if "%CHOICE%"=="3" goto :VIEW_GUIDE
if "%CHOICE%"=="4" goto :OPEN_DIRECTORY

echo 无效选择，请重新运行
pause
goto :EOF