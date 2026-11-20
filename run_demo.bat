@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ==================================================
echo   CNN MNIST  single-image prediction demo
echo   project: %~dp0
echo ==================================================
echo.
"%~dp0.venv\Scripts\python.exe" -m src.predict --checkpoint "outputs\main_seed42_v1\checkpoints\cnn_seed42_last.pt" --image "demo\digit7_test0.png"
echo.
echo --------------------------------------------------
echo   Try another image: change digit7_test0 to digit5_test8 etc.
echo   demo\ has digit0_test3 ... digit9_test7 (10 images)
echo --------------------------------------------------
pause
