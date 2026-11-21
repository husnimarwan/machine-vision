@echo off
title MachineVision - Stop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0server.ps1" -Action stop
rem Beri waktu 2 detik supaya pesan di atas sempat dibaca sebelum jendela menutup
ping -n 3 127.0.0.1 >nul 2>&1
