#!/bin/bash
# run_system.sh - Unix/Mac/Linux
# Automated startup script for the forecasting system

echo "=============================================="
echo "Fresh Produce Forecasting System - Startup"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt

# Setup project structure
echo -e "${GREEN}Setting up project structure...${NC}"
python main.py

# Process data
echo -e "${GREEN}Processing data...${NC}"
python data_processor.py

# Generate forecast
echo -e "${GREEN}Generating forecast...${NC}"
python forecasting_engine.py

echo ""
echo "=============================================="
echo "Data processing complete!"
echo "=============================================="
echo ""
echo "Starting services..."
echo ""

# Start API in background
echo -e "${GREEN}Starting API server on port 8000...${NC}"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit dashboard
echo -e "${GREEN}Starting dashboard on port 8501...${NC}"
streamlit run ui/dashboard.py &
DASH_PID=$!

echo ""
echo "=============================================="
echo "System is running!"
echo "=============================================="
echo ""
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Dashboard: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap "kill $API_PID $DASH_PID; exit" INT
wait
