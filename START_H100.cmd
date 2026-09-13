@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_h100.ps1"
echo.
echo Inspect logs if the launcher stopped.
pause
