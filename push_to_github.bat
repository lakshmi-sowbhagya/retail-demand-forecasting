@echo off
cd /d "%~dp0"
title Push to GitHub - lakshmi-sowbhagya
echo =====================================================================
echo Pushing Project to:
echo https://github.com/lakshmi-sowbhagya/retail-demand-forecasting
echo =====================================================================
echo.

set GIT_EXE=C:\Users\hp\AppData\Local\Programs\Git\cmd\git.exe

"%GIT_EXE%" push -u origin main

echo.
echo =====================================================================
echo Done!
echo =====================================================================
pause
