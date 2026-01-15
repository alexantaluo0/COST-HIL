@echo off
set "PY_DIR=C:\Users\luoguozhong\.sys_data\py310\python-3.10.11-embed-amd64"
set "PATH=%PY_DIR%;%PY_DIR%\Scripts;%PATH%"
set "PROMPT=(sys_env) $P$G"

echo ===================================================
echo   潜行环境已启动 (Python 3.10)
echo   可以使用 python 和 pip 命令
echo ===================================================
echo.
python --version
echo.
cmd /k
