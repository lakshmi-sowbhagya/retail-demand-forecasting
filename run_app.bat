@echo off
cd /d "%~dp0"
title Retail Demand Forecasting and Inventory Optimization App
echo =====================================================================
echo Launching Retail Demand Forecasting and Inventory Optimization App
echo Domain: Predictive Modeling and Business Analytics
echo Model : Prophet / ARIMA Hybrid with Uncertainty Quantification
echo =====================================================================
echo.

set PYTHON_EXE=C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PYTHON_EXE%" (
    echo Using Python 3.11 from: %PYTHON_EXE%
    "%PYTHON_EXE%" -m streamlit run app.py
) else (
    echo Checking default python in PATH...
    python -m streamlit run app.py
)

pause
