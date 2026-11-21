@echo off
title MachineVision - Start
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0server.ps1" -Action start
if errorlevel 1 (
    echo.
    pause
)
