@echo off
chcp 65001 >nul
title Image-to-3D Mesh Reconstruction Studio
echo ========================================================================
echo   🚀 STARTING IMAGE-TO-3D MESH RECONSTRUCTION STUDIO
echo   • Repository : D:\project\AI-repo\image-to-3D
echo   • Web UI URL : http://localhost:8000
echo ========================================================================
start "" "http://localhost:8000"
python app.py
pause
