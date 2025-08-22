@echo off
echo Starting Autonomous AI Agent...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the API server
cd api
python -m uvicorn index:app --host 0.0.0.0 --port 8000

pause
