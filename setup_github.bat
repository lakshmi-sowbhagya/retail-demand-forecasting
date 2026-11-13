@echo off
cd /d "%~dp0"
title Push Retail Demand Forecasting Project to GitHub
echo =====================================================================
echo GitHub Repository Setup and Push Helper
echo =====================================================================
echo.

set GIT_EXE=C:\Users\hp\AppData\Local\Programs\Git\cmd\git.exe

if exist "%GIT_EXE%" (
    set "GIT_CMD=%GIT_EXE%"
) else (
    where git >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Git is not installed or not in your system PATH.
        pause
        exit /b 1
    )
    set "GIT_CMD=git"
)

echo Initializing local Git repository...
"%GIT_CMD%" init
"%GIT_CMD%" config user.name "Retail Demand Planner"
"%GIT_CMD%" config user.email "planner@example.com"
"%GIT_CMD%" add .
"%GIT_CMD%" commit -m "Retail Demand Forecasting under Uncertainty - Prophet/ARIMA Hybrid Models"
"%GIT_CMD%" branch -M main

echo.
echo =====================================================================
echo Please enter your GitHub Repository URL
echo Example: https://github.com/myusername/my-retail-project.git
echo (Do not include angle brackets ^< or ^>)
echo =====================================================================
set /p REPO_URL="Enter GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo.
    echo No URL provided. Git repository is initialized and committed locally.
    pause
    exit /b 0
)

"%GIT_CMD%" remote remove origin >nul 2>nul
"%GIT_CMD%" remote add origin %REPO_URL%
echo.
echo Pushing project to GitHub main branch...
"%GIT_CMD%" push -u origin main

echo.
echo =====================================================================
echo Done! Project pushed to GitHub.
echo =====================================================================
pause
