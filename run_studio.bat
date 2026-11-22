@echo off
chcp 65001 >nul
title Image-to-3D Mesh Reconstruction Studio
cd /d "%~dp0"
echo ========================================================================
echo   🚀 STARTING IMAGE-TO-3D MESH RECONSTRUCTION STUDIO
echo   • Directory  : %~dp0
echo   • Web UI URL : http://localhost:8000
echo ========================================================================
start "" "http://localhost:8000"
python app.py
pause
