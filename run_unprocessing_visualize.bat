@echo off
setlocal
source activate clip
rem Run from repo root
@REM cd /d "%~dp0"

cd D:\02-Study\local_git\DeData_learn
@REM .\run_unprocessing_visualize.bat

rem One-click visualization run
python unprocessing.py ^
  --input D://04-dataset//test_isp//shevy_and_son.jpg ^
  --output raw_output.png ^
  --iso 6400 ^
  --visualize ^
  --vis-dir visualization_shevy

echo.
echo Done. Check the "visualization" folder for outputs.
pause

