"""
Forecasting Engine Module
Handles ML model training, prediction, and business logic application
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pykalman import KalmanFilter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class ForecastingEngine:
    def __init__(self):
        self.model = None
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 20
        }
        self.feature_cols = [
            'DayOfWeek', 'IsWeekend', 'Month', 'Week', 'DayOfMonth',
            'Lag7_Sales', 'RollingMean7',
            'HasPromotion', 'DiscountPct',
            'Temperature', 'Rainfall', 'IsUnseasonablyWarm',
            'CompetitorPriceIndex', 'PromoIntensity',
        ]
        self.kalman_models = {} # Store Kalman models per product

    def train_kalman(self, df):
        """Train Kalman Filter models per product"""
        print("Training Kalman Filter models...")
        products = df['ProductID'].unique()
        
        for pid in products:
            product_data = df[df['ProductID'] == pid].sort_values('Date')
            
            # Use Local Linear Trend Model
            # transition_matrix = [[1, 1], [0, 1]] -> Level + Trend
            # observation_matrix = [[1, 0]] -> We observe the Level
            
            kf = KalmanFilter(
                transition_matrices = [[1, 1], [0, 1]],
                observation_matrices = [[1, 0]],
                initial_state_mean = [product_data['QuantitySold'].iloc[0], 0],
                initial_state_covariance = [[1, 0], [0, 1]],
                observation_covariance = 1,
                transition_covariance = [[0.1, 0], [0, 0.1]]
            )
            
            # Simple training: just fit to get smoothed states
            # In a more advanced setup, we'd use em() to estimate parameters
            # but for robustness/speed we stick to reasonable defaults or light fitting
            
            # state_means, _ = kf.filter(product_data['QuantitySold'].values)
            
            self.kalman_models[pid] = {
                'model': kf,
                'last_state_mean': [product_data['QuantitySold'].iloc[-1], 0], # Approximate last state
                'last_state_cov': [[1, 0], [0, 1]] 
                # Ideally we run kf.filter() on history to get true last state
            }
            
            # To do it properly:
            state_means, state_covs = kf.filter(product_data['QuantitySold'].values)
            self.kalman_models[pid]['last_state_mean'] = state_means[-1]
            self.kalman_models[pid]['last_state_cov'] = state_covs[-1]
            
        print(f"✓ Trained Kalman models for {len(products)} products")

    def train_model(self, data_path='data/processed_features.csv'):
        """Train the XGBoost model"""
        print("Training forecasting model...")
        
        try:
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Keep only relevant columns to avoid dropping rows due to unused features (e.g. Lag14)
            cols_to_keep = self.feature_cols + ['Date', 'QuantitySold', 'Category', 'StoreID']
            # Only keep columns that actually exist
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[cols_to_keep]
            
            # Drop NaN values created by lag features
            df = df.dropna()
            
            # Split data (Last 7 days as test to ensure training data exists with short history)
            max_date = df['Date'].max()
            split_date = max_date - timedelta(days=7)
            
            train_df = df[df['Date'] <= split_date]
            test_df = df[df['Date'] > split_date]
            
            # Check if we have enough data
            if len(train_df) == 0:
                print("⚠ Not enough data for training with 14-day lags. Reducing lags or test size.")
                # Fallback: Use all data for training if test split leaves nothing
                train_df = df
                test_df = df.iloc[-5:] # Mock test
            
            X_train = train_df[self.feature_cols]
            y_train = train_df['QuantitySold']
            X_test = test_df[self.feature_cols]
            y_test = test_df['QuantitySold']
            
            # Train model
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            predictions = self.model.predict(X_test)
            metrics = self.calculate_metrics(y_test, predictions)
            
            # Save artifacts
            self.save_model(metrics)
            
            print(f"✓ Model trained. MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            return metrics
            
        except FileNotFoundError:
            print("⚠ Processed data not found. Run data_processor.py first.")
            return None

    def calculate_metrics(self, actual, predicted):
        """Calculate performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Avoid division by zero for MAPE
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "Accuracy": float(100 - mape)
        }

    def save_model(self, metrics):
        """Save model and metrics"""
        joblib.dump(self.model, 'models/xgboost_model.pkl')
        with open('models/forecast_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def generate_forecast(self, days_ahead=14):
        """Generate future forecast with business logic adjustments"""
        print(f"Generating forecast for next {days_ahead} days...")
        
        try:
            # Load product master
            products = pd.read_csv('data/product_master.csv')
            
            # Load recent history for context (needed for Kalman)
            history_df = pd.read_csv('data/processed_features.csv')
            history_df['Date'] = pd.to_datetime(history_df['Date'])
            
            # Train Kalman models if not already done (simplified flow)
            if not self.kalman_models:
                self.train_kalman(history_df)

            future_dates = [datetime.now() + timedelta(days=x) for x in range(1, days_ahead + 1)]
            
            forecast_rows = []
            
            for date in future_dates:
                for _, product in products.iterrows():
                    # Create features for XGBoost
                    row = {
                        'Date': date,
                        'ProductID': product['ProductID'],
                        'Category': product['Category'],
                        'StoreID': 'ST001', 
                        'DayOfWeek': date.weekday(),
                        'IsWeekend': 1 if date.weekday() >= 5 else 0,
                        'Month': date.month,
                        'Week': date.isocalendar().week,
                        'DayOfMonth': date.day,
                        'Lag7_Sales': 100, 
                        'RollingMean7': 100,
                        'HasPromotion': 0,
                        'DiscountPct': 0,
                        'Temperature': 15,
                        'Rainfall': 2,
                        'IsUnseasonablyWarm': 0,
                        'CompetitorPriceIndex': 1.0,
                        'PromoIntensity': 0,
                        'ShelfLife': product['ShelfLife'],
                        'UnitPrice': product['UnitPrice']
                    }
                    forecast_rows.append(row)
            
            forecast_df = pd.DataFrame(forecast_rows)
            
            # --- XGBoost Prediction ---
            if self.model is None:
                self.model = joblib.load('models/xgboost_model.pkl')
                
            X_future = forecast_df[self.feature_cols]
            xgboost_pred = self.model.predict(X_future)
            
            # --- Kalman Prediction ---
            kalman_preds = []
            # We need to project state forward for each product
            # For simplicity in this batch structure, we'll iterate
            
            for index, row in forecast_df.iterrows():
                pid = row['ProductID']
                if pid in self.kalman_models:
                    # Retrieve state
                    current_mean = self.kalman_models[pid]['last_state_mean']
                    current_cov = self.kalman_models[pid]['last_state_cov']
                    kf = self.kalman_models[pid]['model']
                    
                    # Predict next step (simplified: just step forward using transition)
                    # For multi-step, we should update state iteratively.
                    # Here we are just mocking a "next step" forecast for the batch
                    # In a loop over future dates per product, we would update 'current_mean'
                    
                    # Let's do a proper iterative forecast per product for the date range
                    # But since forecast_df is mixed, we can't easily do it vectorised
                    pass 

            # Re-doing prediction loop to handle Kalman correctly per product
            xgboost_forecasts = []
            kalman_forecasts = []
            
            for pid in products['ProductID'].unique():
                prod_subset = forecast_df[forecast_df['ProductID'] == pid]
                if prod_subset.empty: continue
                
                # XGBoost
                X_feat = prod_subset[self.feature_cols]
                xgb_y = self.model.predict(X_feat)
                
                # Kalman
                kal_y = []
                if pid in self.kalman_models:
                    state_mean = self.kalman_models[pid]['last_state_mean']
                    state_cov = self.kalman_models[pid]['last_state_cov']
                    kf = self.kalman_models[pid]['model']
                    
                    for _ in range(len(prod_subset)):
                        state_mean, state_cov = kf.filter_update(state_mean, state_cov)
                        kal_y.append(state_mean[0]) # Observation is the Level
                else:
                    kal_y = [0] * len(prod_subset)
                
                # Create result rows
                for i in range(len(prod_subset)):
                    idx = prod_subset.index[i]
                    base_row = prod_subset.iloc[i].to_dict()
                    
                    # Add XGBoost Result
                    row_xgb = base_row.copy()
                    row_xgb['Model'] = 'XGBoost'
                    row_xgb['ForecastedDemand'] = xgb_y[i]
                    xgboost_forecasts.append(row_xgb)
                    
                    # Add Kalman Result
                    row_kal = base_row.copy()
                    row_kal['Model'] = 'Kalman Filter'
                    row_kal['ForecastedDemand'] = max(0, kal_y[i]) # Ensure non-negative
                    kalman_forecasts.append(row_kal)

            # Combine
            final_df = pd.DataFrame(xgboost_forecasts + kalman_forecasts)
            
            # Apply Business Rules
            self.apply_business_rules(final_df)
            
            # Save forecast
            final_df.to_csv('data/forecast_output.csv', index=False)
            print("✓ Forecast generated (XGBoost + Kalman) and saved")
            
            return final_df
            
        except Exception as e:
            print(f"✗ Error generating forecast: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def apply_business_rules(self, df):
        """Apply business logic adjustments to the forecast"""
        
        # 1. Weekend Boost (+15% for fresh categories)
        weekend_mask = (df['IsWeekend'] == 1) & (df['Category'].isin(['Berries', 'Leafy Salads', 'Tomatoes']))
        df.loc[weekend_mask, 'ForecastedDemand'] *= 1.15
        
        # 2. Midweek Reduction (-8%)
        midweek_mask = df['DayOfWeek'].isin([1, 2, 3]) # Tue, Wed, Thu
        df.loc[midweek_mask, 'ForecastedDemand'] *= 0.92
        
        # 3. Promotional Uplift (if promotion exists)
        # Note: In our mock future data, HasPromotion is 0, but logic remains
        promo_mask = df['HasPromotion'] == 1
        # df.loc[promo_mask, 'ForecastedDemand'] *= df.loc[promo_mask, 'ExpectedUplift'] 
        # Using a default if ExpectedUplift not present in df, or use column if exists
        
        # 4. Safety Stock Calculation
        df['SafetyStock'] = np.where(df['ShelfLife'] > 3, 
                                     df['ForecastedDemand'] * 0.15, 
                                     df['ForecastedDemand'] * 0.10)
        
        # 5. Final Recommended Order
        df['RecommendedOrder'] = np.ceil(df['ForecastedDemand'] + df['SafetyStock']).astype(int)
        
        # 6. Confidence Intervals (Simple proxy using MAPE)
        # Assuming 10% MAPE for simplicity if not calculated dynamically per row
        mape = 0.10 
        df['LowerBound'] = df['RecommendedOrder'] * (1 - mape)
        df['UpperBound'] = df['RecommendedOrder'] * (1 + mape)

def main():
    engine = ForecastingEngine()
    
    # Train
    metrics = engine.train_model()
    
    # Generate Forecast
    engine.generate_forecast()

if __name__ == "__main__":
    main()
