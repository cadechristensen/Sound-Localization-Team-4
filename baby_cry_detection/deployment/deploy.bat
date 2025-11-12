@echo off
REM Baby Cry Detection - Raspberry Pi Deployment (Windows Batch)
REM This batch file wraps the PowerShell deployment script for easy Windows use
REM
REM Usage:
REM   deploy.bat <pi_host> <model_path>
REM
REM Examples:
REM   deploy.bat pi@raspberrypi.local results\train_2025-10-21_02-36-55\model_quantized.pth
REM   deploy.bat pi@192.168.1.100 results\train_2025-10-21_02-36-55\model_quantized.pth
REM

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Parse arguments
set PI_HOST=%~1
set MODEL_PATH=%~2

REM Default values
if "!PI_HOST!"=="" (
    echo [ERROR] Pi host is required
    echo Usage: deploy.bat pi_host model_path
    echo Example: deploy.bat pi@raspberrypi.local results\train_2025-10-21_02-36-55\model_quantized.pth
    exit /b 1
)

if "!MODEL_PATH!"=="" (
    echo [ERROR] Model path is required
    echo Usage: deploy.bat pi_host model_path
    echo Example: deploy.bat pi@raspberrypi.local results\train_2025-10-21_02-36-55\model_quantized.pth
    exit /b 1
)

echo.
echo ============================================
echo Baby Cry Detection - Raspberry Pi Deployment
echo ============================================
echo Target: !PI_HOST!
echo Model:  !MODEL_PATH!
echo.
echo Starting deployment...
echo.

REM Run PowerShell script
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '!SCRIPT_DIR!deploy_to_pi.ps1' -PiHost '!PI_HOST!' -ModelPath '!MODEL_PATH!'"

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Deployment failed
    exit /b 1
)

echo.
echo [SUCCESS] Deployment completed!
echo.
pause
