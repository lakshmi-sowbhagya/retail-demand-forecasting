# PowerShell Launcher for Retail Demand Forecasting App
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Launching Retail Demand Forecasting & Inventory Optimization App" -ForegroundColor Green
Write-Host "Domain: Predictive Modeling & Business Analytics" -ForegroundColor White
Write-Host "Model : Prophet / ARIMA Hybrid with Uncertainty Quantification" -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Cyan

$pythonExe = "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"

if (Test-Path $pythonExe) {
    Write-Host "Found Python 3.11 at: $pythonExe" -ForegroundColor Yellow
    & $pythonExe -m streamlit run app.py
} else {
    Write-Host "Using default Python in PATH..." -ForegroundColor Yellow
    python -m streamlit run app.py
}
