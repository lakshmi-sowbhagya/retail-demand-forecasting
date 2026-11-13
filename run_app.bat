@echo off
title Retail Demand Forecasting & Inventory Optimization App
echo =====================================================================
echo Launching Retail Demand Forecasting & Inventory Optimization App
echo Domain: Predictive Modeling & Business Analytics
echo Model : Prophet / ARIMA Hybrid with Uncertainty Quantification
echo =====================================================================
echo.

set PYTHON_CMD="C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"

if exist %PYTHON_CMD% (
    echo Using Python 3.11 from: %PYTHON_CMD%
    %PYTHON_CMD% -m streamlit run app.py
) else (
    echo Checking default python in PATH...
    python -m streamlit run app.py
)

pause
