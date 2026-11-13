@echo off
title Push Retail Demand Forecasting Project to GitHub
echo =====================================================================
echo GitHub Repository Setup & Push Automation
echo =====================================================================
echo.

where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in your system PATH.
    echo.
    echo Please install Git using either method below:
    echo 1. Download official installer from: https://git-scm.com/download/win
    echo 2. OR run in PowerShell outside sandbox: winget install --id Git.Git -e --source winget
    echo 3. OR upload this project folder directly to GitHub in your web browser!
    echo.
    pause
    exit /b 1
)

echo Initializing local Git repository...
git init
git add .
git commit -m "Initial commit: Demand Forecasting under Uncertainty with Prophet-ARIMA Hybrid Models"
git branch -M main

echo.
echo Please copy your GitHub repository URL (e.g., https://github.com/your-username/retail-demand-forecasting.git)
set /p REPO_URL="Enter GitHub Repository URL: "

if "%REPO_URL%"=="" (
    echo No URL provided. Git repository is initialized locally with your first commit.
    echo When you are ready, run:
    echo   git remote add origin ^<your-url^>
    echo   git push -u origin main
    pause
    exit /b 0
)

git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%
echo.
echo Pushing project to GitHub main branch...
git push -u origin main

echo.
echo =====================================================================
echo Project successfully attached and pushed to GitHub!
echo =====================================================================
pause
