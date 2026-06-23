"""
API Backend Module
FastAPI application serving forecast data and recommendations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

app = FastAPI(
    title="Fresh Produce Forecasting System",
    description="API for accessing forecast data, metrics, and recommendations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ApprovalRequest(BaseModel):
    action: str
    notes: str

# Helpers
def load_forecast_data():
    try:
        return pd.read_csv('data/forecast_output.csv')
    except Exception:
        return pd.DataFrame()

def load_metrics():
    try:
        with open('models/forecast_metrics.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Fresh Produce Forecasting API is running"}

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Returns overall KPIs, category breakdown, and metrics"""
    df = load_forecast_data()
    metrics = load_metrics()
    
    if df.empty:
        return {"error": "No forecast data available"}
    
    summary = {
        "kpis": {
            "total_demand": int(df['ForecastedDemand'].sum()),
            "total_orders": int(df['RecommendedOrder'].sum()),
            "waste_reduction_pct": 25, # meaningful dummy or calculated
            "accuracy_last_run": metrics.get('Accuracy', 0)
        },
        "category_breakdown": df.groupby('Category')['RecommendedOrder'].sum().to_dict(),
        "metrics": metrics
    }
    return summary

@app.get("/forecast/daily")
async def get_daily_forecast(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    store_id: Optional[str] = None
):
    """Get daily forecast with filters"""
    df = load_forecast_data()
    if df.empty:
        return []
        
    # Apply filters
    if category:
        df = df[df['Category'] == category]
    if store_id:
        df = df[df['StoreID'] == store_id]
    
    # Convert dates for filtering
    df['Date'] = pd.to_datetime(df['Date'])
    
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
        
    return df.to_dict(orient='records')

@app.get("/forecast/weekly")
async def get_weekly_forecast(category: Optional[str] = None):
    """Aggregate forecast by week"""
    df = load_forecast_data()
    if df.empty:
        return []
        
    if category:
        df = df[df['Category'] == category]
        
    weekly = df.groupby(['Week', 'Category']).agg({
        'RecommendedOrder': 'sum',
        'ForecastedDemand': 'sum'
    }).reset_index()
    
    return weekly.to_dict(orient='records')

@app.get("/forecast/category/{category}")
async def get_category_forecast(category: str):
    """Get forecast for a specific category"""
    df = load_forecast_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
        
    filtered = df[df['Category'] == category]
    return filtered.to_dict(orient='records')

@app.get("/recommendations")
async def get_recommendations(priority: Optional[str] = None, status: Optional[str] = None):
    """Get system recommendations (Mocked for demo)"""
    recommendations = [
        {
            "id": 101,
            "type": "Stockout Risk",
            "message": "Potential stockout for Berries at Store ST005 on Saturday",
            "priority": "high",
            "status": "pending",
            "impact": "Lost revenue estimate: £450"
        },
        {
            "id": 102,
            "type": "Waste Alert",
            "message": "Reduce order for Leafy Salads by 15% due to incoming warm weather end",
            "priority": "medium",
            "status": "pending",
            "impact": "Waste saving: £120"
        }
    ]
    
    if priority:
        recommendations = [r for r in recommendations if r['priority'] == priority]
    if status:
        recommendations = [r for r in recommendations if r['status'] == status]
        
    return recommendations

@app.post("/recommendations/{rec_id}/approve")
async def approve_recommendation(rec_id: int, request: ApprovalRequest):
    """Approve a recommendation"""
    return {
        "id": rec_id,
        "status": "approved",
        "action_taken": request.action,
        "timestamp": datetime.now()
    }

@app.get("/analytics/forecast-accuracy")
async def get_forecast_accuracy():
    """Get detailed accuracy metrics"""
    return load_metrics()
