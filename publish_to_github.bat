@echo off
cd /d "%~dp0"
title Publish to GitHub - lakshmi-sowbhagya
echo =====================================================================
echo Automated GitHub Publisher for lakshmi-sowbhagya
echo =====================================================================
echo.

set PATH=%PATH%;C:\Users\hp\AppData\Local\Programs\Git\cmd

echo Checking GitHub Login...
gh auth status >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Please log in to your GitHub account:
    echo 1. It will display a one-time code.
    echo 2. Press Enter to open the authorization page.
    echo 3. Paste the code and click Authorize.
    echo.
    gh auth login --web --git-protocol https
)

echo.
echo Creating repository 'retail-demand-forecasting' and pushing project...
gh repo create retail-demand-forecasting --public --source=. --remote=origin --push

echo.
echo =====================================================================
echo SUCCESS! Your project is published at:
echo https://github.com/lakshmi-sowbhagya/retail-demand-forecasting
echo =====================================================================
pause
