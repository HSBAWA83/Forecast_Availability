"""
Data Processing & Feature Engineering Module
Processes DILO.xlsx and creates enriched datasets for forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Processes retail supply chain data from Excel sheets
    Creates feature-rich datasets for ML forecasting
    """
    
    def __init__(self, excel_path='data/DILO.xlsx'):
        self.excel_path = excel_path
        self.data = {}
        
    def load_data(self):
        """Load all sheets from Excel file"""
        try:
            excel_file = pd.ExcelFile(self.excel_path)
            
            # Expected sheets based on problem statement
            sheet_mapping = {
                'ProductMaster': 'product_master',
                'Stores': 'stores',
                'Suppliers': 'suppliers',
                'Promotions': 'promotions',
                'CompetitionIndex': 'competition',
                'Weather': 'weather',
                'Forecast': 'forecast',
                'Sales': 'sales',
                'Inventory': 'inventory',
                'CustomerFeedback': 'feedback'
            }
            
            for sheet_name, key in sheet_mapping.items():
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Normalize columns if needed
                    if key == 'product_master':
                        # Map snake_case to CamelCase if found
                        rename_map = {
                            'sku_id': 'ProductID',
                            'sku_name': 'ProductName',
                            'category': 'Category',
                            'shelf_life_days': 'ShelfLife',
                            'base_price': 'UnitPrice',
                            'supplier_id': 'SupplierID'
                        }
                        df = df.rename(columns=rename_map)
                        
                        # Add missing columns with defaults
                        if 'UnitCost' not in df.columns:
                            df['UnitCost'] = df['UnitPrice'] * 0.7  # Assumption
                        if 'MinOrderQty' not in df.columns:
                            df['MinOrderQty'] = 50
                    
                    elif key == 'stores':
                        rename_map = {
                            'store_id': 'StoreID',
                            'store_name': 'StoreName',
                            'region': 'Region', 
                            'size': 'Size',
                            'format': 'Size',
                            'footfall': 'FootfallCategory',
                            'store_type': 'Type' 
                        }
                        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                        
                        if 'FootfallCategory' not in df.columns:
                            df['FootfallCategory'] = 'Medium' # Default
                            
                    elif key == 'promotions':
                        rename_map = {
                            'sku_id': 'ProductID',
                            'start_date': 'StartDate',
                            'end_date': 'EndDate',
                            'discount_pct': 'DiscountPct',
                            'mechanic': 'Mechanic'
                        }
                        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                        
                        if 'ExpectedUplift' not in df.columns:
                            df['ExpectedUplift'] = 1.3 # Default 30% uplift
                            
                    elif key == 'weather':
                        rename_map = {
                            'date': 'Date',
                            'store_id': 'StoreID',
                            'temperature_c': 'Temperature',
                            'rainfall_mm': 'Rainfall'
                        }
                        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                         
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                        
                        if 'IsUnseasonablyWarm' not in df.columns:
                            # Simple logic if temp missing or just default
                            if 'Temperature' in df.columns:
                                df['IsUnseasonablyWarm'] = ((df['Temperature'] > 22) & (df['Date'].dt.month.isin([4,5,9,10]))).astype(int)
                            else:
                                df['IsUnseasonablyWarm'] = 0

                    elif key == 'inventory':
                        rename_map = {
                            'date': 'Date',
                            'store_id': 'StoreID',
                            'sku_id': 'ProductID',
                            'soh_units': 'StockLevel',
                            'oos_flag': 'StockoutFlag',
                            'waste_units': 'WasteQty'
                        }
                        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])

                    elif key == 'feedback':
                         rename_map = {
                            'date': 'Date',
                            'store_id': 'StoreID',
                            'availability_complaints': 'ItemUnavailableComplaints',
                            'nps': 'NPSScore'
                        }
                         df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                         if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])

                    elif key == 'sales':

                        rename_map = {
                            'date': 'Date',
                            'store_id': 'StoreID',
                            'sku_id': 'ProductID',
                            'units_sold': 'QuantitySold',
                            'price': 'SalesPrice',
                            'promo_flag': 'HasPromotion'
                        }
                        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                        
                        # Ensure Date is datetime
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                        
                    self.data[key] = df
                    print(f"✓ Loaded {sheet_name}: {len(self.data[key])} records")
                else:
                    print(f"⚠ Sheet {sheet_name} not found, creating sample data")
                    # If this crashes because of strict dependencies on previous data, handle it
                    try:
                        self.data[key] = self.create_sample_data(key)
                    except Exception as e:
                        print(f"  ⚠ Could not create sample data for {key}: {e}")
                        self.data[key] = pd.DataFrame() # Fallback
            
            return True
        except FileNotFoundError:
            print(f"⚠ Excel file not found at {self.excel_path}")
            print("Creating sample datasets for demonstration...")
            self.create_all_sample_data()
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_sample_data(self, data_type):
        """Create sample data for missing sheets"""
        
        # Date range: Last 8 weeks (56 days) for historical + 2 weeks forecast
        end_date = datetime.now()
        start_date = end_date - timedelta(days=56)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if data_type == 'product_master':
            categories = ['Berries', 'Leafy Salads', 'Tomatoes', 'Bananas', 'Citrus', 'Root Vegetables']
            products = []
            for cat in categories:
                for i in range(3):  # 3 SKUs per category
                    products.append({
                        'ProductID': f"{cat[:3].upper()}{i+1:03d}",
                        'ProductName': f"{cat} - Premium Grade {i+1}",
                        'Category': cat,
                        'ShelfLife': np.random.randint(2, 7),
                        'UnitCost': np.random.uniform(1.5, 5.0),
                        'UnitPrice': np.random.uniform(2.5, 8.0),
                        'MinOrderQty': np.random.randint(50, 200)
                    })
            return pd.DataFrame(products)
        
        elif data_type == 'stores':
            regions = ['South East', 'London', 'Midlands', 'North']
            stores = []
            for i in range(20):
                stores.append({
                    'StoreID': f"ST{i+1:03d}",
                    'StoreName': f"Store {i+1}",
                    'Region': regions[i % len(regions)],
                    'Size': np.random.choice(['Small', 'Medium', 'Large']),
                    'FootfallCategory': np.random.choice(['High', 'Medium', 'Low'])
                })
            return pd.DataFrame(stores)
        
        elif data_type == 'sales':
            # Generate realistic sales data
            products = self.data.get('product_master', self.create_sample_data('product_master'))
            stores = self.data.get('stores', self.create_sample_data('stores'))
            
            sales_records = []
            for date in dates[:49]:  # Historical data only
                for _, product in products.iterrows():
                    for _, store in stores.iterrows():
                        # Base demand
                        base_demand = np.random.randint(50, 200)
                        
                        # Weekend boost (Fri, Sat, Sun)
                        weekend_multiplier = 1.4 if date.dayofweek >= 4 else 1.0
                        
                        # Category seasonality
                        if product['Category'] in ['Berries', 'Leafy Salads'] and date.month in [6, 7, 8]:
                            season_multiplier = 1.3
                        else:
                            season_multiplier = 1.0
                        
                        # Random variation
                        noise = np.random.uniform(0.85, 1.15)
                        
                        quantity = int(base_demand * weekend_multiplier * season_multiplier * noise)
                        
                        sales_records.append({
                            'Date': date,
                            'StoreID': store['StoreID'],
                            'ProductID': product['ProductID'],
                            'QuantitySold': quantity,
                            'Revenue': quantity * product['UnitPrice'],
                            'WasteQty': np.random.randint(0, int(quantity * 0.1))
                        })
            
            return pd.DataFrame(sales_records)
        
        elif data_type == 'promotions':
            products = self.data.get('product_master', self.create_sample_data('product_master'))
            promotions = []
            
            # Add some promotional periods
            promo_dates = [
                (dates[10], dates[12]),  # Weekend promo
                (dates[24], dates[26]),  # Mid-period promo
                (dates[38], dates[40])   # Recent promo
            ]
            
            for start, end in promo_dates:
                for _, product in products.sample(n=min(5, len(products))).iterrows():
                    promotions.append({
                        'PromoID': f"PROMO_{len(promotions)+1:03d}",
                        'ProductID': product['ProductID'],
                        'StartDate': start,
                        'EndDate': end,
                        'DiscountPct': np.random.uniform(15, 30),
                        'ExpectedUplift': np.random.uniform(1.3, 1.8)
                    })
            
            return pd.DataFrame(promotions)
        
        elif data_type == 'weather':
            weather_records = []
            regions = ['South East', 'London', 'Midlands', 'North']
            
            for date in dates:
                for region in regions:
                    temp = np.random.normal(18, 5) if date.month in [6, 7, 8] else np.random.normal(12, 4)
                    weather_records.append({
                        'Date': date,
                        'Region': region,
                        'Temperature': temp,
                        'Rainfall': max(0, np.random.normal(2, 3)),
                        'IsUnseasonablyWarm': 1 if (temp > 22 and date.month not in [6, 7, 8]) else 0
                    })
            
            return pd.DataFrame(weather_records)
        
        elif data_type == 'competition':
            products = self.data.get('product_master', self.create_sample_data('product_master'))
            comp_records = []
            
            for date in dates[:49]:
                for _, product in products.iterrows():
                    comp_records.append({
                        'Date': date,
                        'ProductID': product['ProductID'],
                        'CompetitorPriceIndex': np.random.uniform(0.85, 1.15),
                        'MarketSharePct': np.random.uniform(15, 35),
                        'PromoIntensity': np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                    })
            
            return pd.DataFrame(comp_records)
        
        elif data_type == 'inventory':
            products = self.data.get('product_master', self.create_sample_data('product_master'))
            stores = self.data.get('stores', self.create_sample_data('stores'))
            
            inv_records = []
            for date in dates[:49]:
                for _, store in stores.iterrows():
                    for _, product in products.iterrows():
                        inv_records.append({
                            'Date': date,
                            'StoreID': store['StoreID'],
                            'ProductID': product['ProductID'],
                            'StockLevel': np.random.randint(100, 500),
                            'StockoutFlag': np.random.choice([0, 1], p=[0.92, 0.08])
                        })
            
            return pd.DataFrame(inv_records)
        
        elif data_type == 'feedback':
            stores = self.data.get('stores', self.create_sample_data('stores'))
            feedback = []
            
            for date in dates[:49]:
                for _, store in stores.iterrows():
                    feedback.append({
                        'Date': date,
                        'StoreID': store['StoreID'],
                        'NPSScore': np.random.randint(6, 10),
                        'ItemUnavailableComplaints': np.random.randint(0, 15),
                        'QualityComplaints': np.random.randint(0, 8)
                    })
            
            return pd.DataFrame(feedback)
        
        elif data_type == 'forecast':
            # This will be generated by the forecasting engine
            return pd.DataFrame()
        
        return pd.DataFrame()
    
    def create_all_sample_data(self):
        """Create all sample datasets"""
        data_types = ['product_master', 'stores', 'suppliers', 'sales', 
                      'promotions', 'weather', 'competition', 'inventory', 'feedback']
        
        for dt in data_types:
            self.data[dt] = self.create_sample_data(dt)
    
    def create_features(self):
        """Create feature-rich dataset for ML"""
        
        sales = self.data['sales'].copy()
        products = self.data['product_master']
        stores = self.data['stores']
        promotions = self.data['promotions']
        weather = self.data['weather']
        competition = self.data['competition']
        
        # Merge product info
        sales = sales.merge(products[['ProductID', 'Category', 'ShelfLife', 'UnitPrice']], 
                           on='ProductID', how='left')
        
        # Merge store info
        sales = sales.merge(stores[['StoreID', 'Region', 'Size', 'FootfallCategory']], 
                           on='StoreID', how='left')
        
        # Merge weather
        # Check if weather is regional or store-level
        if 'Region' in weather.columns and 'StoreID' not in weather.columns:
            sales = sales.merge(weather[['Date', 'Region', 'Temperature', 'Rainfall', 'IsUnseasonablyWarm']], 
                               on=['Date', 'Region'], how='left')
        elif 'StoreID' in weather.columns:
            # Weather is store-level
             cols_to_use = ['Date', 'StoreID', 'Temperature', 'Rainfall', 'IsUnseasonablyWarm']
             cols_to_use = [c for c in cols_to_use if c in weather.columns]
             sales = sales.merge(weather[cols_to_use], 
                               on=['Date', 'StoreID'], how='left')
        
        # Fill missing weather (forward fill or default)
        sales['Temperature'] = sales['Temperature'].fillna(15)
        sales['Rainfall'] = sales['Rainfall'].fillna(0)
        sales['IsUnseasonablyWarm'] = sales['IsUnseasonablyWarm'].fillna(0)
        
        # Merge competition
        sales = sales.merge(competition[['Date', 'ProductID', 'CompetitorPriceIndex', 'PromoIntensity']], 
                           on=['Date', 'ProductID'], how='left')
        
        # Fill missing competition data (e.g. if dates don't overlap)
        sales['CompetitorPriceIndex'] = sales['CompetitorPriceIndex'].fillna(1.0)
        sales['PromoIntensity'] = sales['PromoIntensity'].fillna(0)
        
        # Add promotion flag
        sales['HasPromotion'] = 0
        sales['DiscountPct'] = 0.0
        sales['ExpectedUplift'] = 0.0
        for _, promo in promotions.iterrows():
            mask = ((sales['ProductID'] == promo['ProductID']) & 
                   (sales['Date'] >= promo['StartDate']) & 
                   (sales['Date'] <= promo['EndDate']))
            sales.loc[mask, 'HasPromotion'] = 1
            sales.loc[mask, 'DiscountPct'] = promo['DiscountPct']
            sales.loc[mask, 'ExpectedUplift'] = promo['ExpectedUplift']
        
        # Time features
        sales['DayOfWeek'] = sales['Date'].dt.dayofweek
        sales['IsWeekend'] = (sales['DayOfWeek'] >= 5).astype(int)
        sales['Month'] = sales['Date'].dt.month
        sales['Week'] = sales['Date'].dt.isocalendar().week
        sales['DayOfMonth'] = sales['Date'].dt.day
        
        # Lag features (previous week same day)
        sales = sales.sort_values(['StoreID', 'ProductID', 'Date'])
        sales['Lag7_Sales'] = sales.groupby(['StoreID', 'ProductID'])['QuantitySold'].shift(7)
        sales['Lag14_Sales'] = sales.groupby(['StoreID', 'ProductID'])['QuantitySold'].shift(14)
        
        # Rolling features
        sales['RollingMean7'] = sales.groupby(['StoreID', 'ProductID'])['QuantitySold'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        
        return sales
    
    def save_processed_data(self, features_df):
        """Save processed data"""
        features_df.to_csv('data/processed_features.csv', index=False)
        print(f"✓ Saved processed features: {len(features_df)} records")
        
        # Save individual datasets
        for key, df in self.data.items():
            if len(df) > 0:
                df.to_csv(f'data/{key}.csv', index=False)
                print(f"✓ Saved {key}: {len(df)} records")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("DATA PROCESSING & FEATURE ENGINEERING")
    print("="*80 + "\n")
    
    processor = DataProcessor()
    
    # Load data
    print("📥 Loading data...")
    processor.load_data()
    
    # Create features
    print("\n🔧 Creating features...")
    features = processor.create_features()
    
    # Save processed data
    print("\n💾 Saving processed data...")
    processor.save_processed_data(features)
    
    print("\n✅ Data processing complete!")
    print(f"   Total records: {len(features)}")
    print(f"   Features: {len(features.columns)}")
    print(f"   Date range: {features['Date'].min()} to {features['Date'].max()}")

if __name__ == "__main__":
    main()
