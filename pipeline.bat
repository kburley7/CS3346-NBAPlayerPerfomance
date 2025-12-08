@echo off
rem Run the full pipeline end-to-end (data cleaning → features → modeling → evaluation)
setlocal

python -m src.data_loader || goto :error
python -m src.feature_engineering || goto :error
python -m src.models || goto :error
python -m src.evaluation || goto :error

echo.
echo Pipeline completed successfully.
goto :eof

:error
echo.
echo Pipeline aborted due to error in the previous step.
exit /b 1
