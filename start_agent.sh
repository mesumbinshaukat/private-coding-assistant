#!/bin/bash
echo "Starting Autonomous AI Agent..."

# Activate virtual environment
source venv/bin/activate

# Start the API server
cd api
python -m uvicorn index:app --host 0.0.0.0 --port 8000
