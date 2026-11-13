# PowerShell Script: Initialize & Push to GitHub
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup & Push Helper" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

$gitInstalled = Get-Command git -ErrorAction SilentlyContinue

if (-not $gitInstalled) {
    Write-Host "[NOTICE] Git is not installed on this system yet." -ForegroundColor Yellow
    Write-Host "Quick options to get Git:" -ForegroundColor White
    Write-Host "  1. Download official Git installer: https://git-scm.com/download/win"
    Write-Host "  2. Run outside sandbox: winget install --id Git.Git -e --source winget"
    Write-Host "  3. Or create repository on GitHub.com and use 'Upload files' in browser."
    Write-Host ""
    Read-Host "Press Enter to exit"
    Exit
}

Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init
git add .
git commit -m "Initial commit: Demand Forecasting under Uncertainty with Prophet-ARIMA Hybrid Models"
git branch -M main

$repoUrl = Read-Host "Enter your GitHub Repository URL (e.g., https://github.com/username/repo.git)"
if ($repoUrl) {
    git remote remove origin 2>$null
    git remote add origin $repoUrl
    Write-Host "Pushing to GitHub..." -ForegroundColor Green
    git push -u origin main
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "Local commit created. You can push anytime with:" -ForegroundColor White
    Write-Host "  git remote add origin <repo-url>"
    Write-Host "  git push -u origin main"
}
