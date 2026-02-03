@echo off
REM Start TensorBoard to compare multiple training runs
REM Usage: start_tensorboard_compare.bat

SET ROOT_DIR=outputs\train

REM Check if root directory exists
IF NOT EXIST "%ROOT_DIR%" (
    echo Error: Training output directory '%ROOT_DIR%' does not exist
    echo Please ensure you have training runs in the outputs/train directory
    exit /b 1
)

echo ================================================
echo Starting TensorBoard for Multiple Runs Comparison
echo ================================================
echo.
echo Root log directory: %ROOT_DIR%
echo.
echo TensorBoard Features:
echo   1. Compare loss curves across different runs
echo   2. Toggle runs on/off in the left panel
echo   3. Use smoothing slider to reduce noise
echo   4. Download data as CSV for further analysis
echo.
echo Opening in browser: http://localhost:6006
echo Press Ctrl+C to stop TensorBoard
echo ================================================
echo.

REM Start TensorBoard pointing to the root directory
REM TensorBoard will automatically detect all subdirectories as separate runs
tensorboard --logdir="%ROOT_DIR%" --port=6006 --bind_all
