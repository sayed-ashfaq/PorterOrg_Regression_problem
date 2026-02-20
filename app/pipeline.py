import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ── Load all artifacts once at startup ────────────────────────────────
scaler = joblib.load('porter_model/scaler.pkl')
category_map = joblib.load('porter_model/category_encoding.pkl')
time_period_map = joblib.load('porter_model/time_period_encoding.pkl')
global_mean = joblib.load('porter_model/global_mean.pkl')
feature_cols = joblib.load('porter_model/feature_columns.pkl')
model = keras.models.load_model('porter_model/best_model_v2.keras')

print("All artifacts loaded ✅")

def get_time_period(hour: int) -> str:
    if 6 <= hour <= 9: 
        return 'breakfast'
    elif 11 <= hour <= 14: 
        return 'lunch'
    elif 17 <= hour <= 21: 
        return 'dinner'
    elif hour in [22, 23, 0, 1]: 
        return 'late_night'
    else: 
        return 'off_peak'

def preprocess(raw_input: dict) -> np.ndarray:
    data = raw_input.copy()
    
    # DateTime features
    created_at = pd.to_datetime(data['created_at'])
    data['hour'] = created_at.hour
    data['day_of_week'] = created_at.dayofweek
    data['month'] = created_at.month
    data['is_weekend'] = 1 if data['day_of_week'] >= 5 else 0
    
    # Time period encoding
    time_period = get_time_period(data['hour'])
    data['time_period_encoded'] = time_period_map.get(
        time_period, global_mean)
    
    # Category encoding
    category = data.get('store_primary_category', 'unknown')
    data['category_encoded'] = category_map.get(
        category, global_mean)
    
    # Log transform first
    for col in ['total_items', 'subtotal', 'min_item_price',
                'max_item_price', 'total_outstanding_orders']:
        data[col] = np.log1p(max(data[col], 0))
    
    # Engineer features after log transform
    data['price_range'] = (
        data['max_item_price'] - data['min_item_price'])
    data['avg_item_price'] = (
        data['subtotal'] / max(data['total_items'], 0.0001))
    data['demand_supply_ratio'] = (
        np.expm1(data['total_outstanding_orders']) /
        (raw_input['total_onshift_partners'] + 1))
    
    # One-hot encoding
    for i in [2, 3, 4, 5, 6]:
        data[f'market_id_{i}.0'] = (
            1 if raw_input['market_id'] == i else 0)
    for i in [2, 3, 4, 5, 6, 7]:
        data[f'order_protocol_{i}.0'] = (
            1 if raw_input['order_protocol'] == i else 0)
    
    # Align columns and scale
    df_input = pd.DataFrame([data])[feature_cols]
    return scaler.transform(df_input)

def predict(raw_input: dict) -> dict:
    X = preprocess(raw_input)
    y_log = model.predict(X, verbose=0).flatten()[0]
    y_minutes = float(np.expm1(y_log))
    y_minutes = max(5.0, min(120.0, y_minutes))
    
    return {
        'predicted_delivery_minutes': round(y_minutes, 1),
        'optimistic_minutes': round(max(5.0, y_minutes - 10), 1),
        'pessimistic_minutes': round(min(120.0, y_minutes + 10), 1)
    }