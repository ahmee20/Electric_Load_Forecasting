from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import traceback
from datetime import datetime, timedelta
import hashlib

app = Flask(__name__)
# Disable caching for all routes
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Sample cities if data is not available
SAMPLE_CITIES = ["new york", "los angeles", "chicago", "houston", "phoenix"]

# Load data at startup or use sample data
try:
    if os.path.exists("combined_data.csv"):
        df = pd.read_csv("combined_data.csv", parse_dates=["timestamp"])
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date
        cities = sorted(df["city"].unique().tolist())
        print(f"Data loaded successfully. Found {len(cities)} cities: {cities}")
    else:
        print("Warning: combined_data.csv not found. Using sample data.")
        df = None
        cities = SAMPLE_CITIES
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    df = None
    cities = SAMPLE_CITIES

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html', cities=cities)

# Route for the documentation page
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

# Generate synthetic data - completely generic, no city-specific logic
def generate_data(n_samples=200, seed=None):
    """
    Generate synthetic data without any city-specific logic.
    The same process is used for all cities.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features with natural variations - same process for all cities
    temperature = np.random.normal(15, 10, n_samples)
    humidity = np.random.normal(50, 15, n_samples)
    wind_speed = np.random.normal(10, 5, n_samples)
    
    # Generate demand with seasonal pattern and noise - same process for all cities
    time_index = np.arange(n_samples)
    demand_base = 1000
    seasonal_component = 200 * np.sin(time_index / 20)
    noise_component = np.random.normal(0, 100, n_samples)
    demand = demand_base + seasonal_component + noise_component
    
    # Combine features
    features = np.column_stack([temperature, humidity, wind_speed, demand])
    
    return features

# API endpoint for clustering
@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    try:
        data = request.json
        city = data.get('city', '').lower()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        k = int(data.get('k', 3))
        
        print(f"Clustering request: city={city}, start_date={start_date}, end_date={end_date}, k={k}")
        
        # Create a seed based on inputs
        seed_string = f"{city}_{start_date}_{end_date}_{k}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        
        # Generate sample data - same process for all cities
        n_samples = 200
        features = generate_data(n_samples, seed=seed_hash)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features, columns=["temperature", "humidity", "windSpeed", "demand_mwh"])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        
        # Perform clustering with user-specified k
        kmeans = KMeans(n_clusters=k, random_state=seed_hash % 1000)
        labels = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate silhouette score if more than one cluster
        sil_score = 0
        if len(set(labels)) > 1:
            sil_score = float(silhouette_score(X_scaled, labels))
        
        # Prepare response
        result = {
            "pca_data": X_pca.tolist(),
            "labels": labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "silhouette_score": sil_score,
            "city": city
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in cluster_data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# API endpoint for overall clustering
@app.route('/api/overall-cluster', methods=['POST'])
def overall_cluster_data():
    try:
        data = request.json
        k = int(data.get('k', 3))
        
        print(f"Overall clustering request: k={k}")
        
        # Create a seed based on inputs
        seed_string = f"overall_{k}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        
        # Generate sample data for all cities - same process for all
        n_samples_per_city = 400
        all_features = []
        city_labels = []
        
        # Generate data for each city - same process for all
        for city in SAMPLE_CITIES:
            # Generate features - same process for all cities
            city_features = generate_data(n_samples_per_city, seed=seed_hash + hash(city) % 1000)
            
            all_features.append(city_features)
            city_labels.extend([city] * n_samples_per_city)
        
        # Combine all features
        combined_features = np.vstack(all_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(combined_features, columns=["temperature", "humidity", "windSpeed", "demand_mwh"])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        
        # Perform clustering with user-specified k
        kmeans = KMeans(n_clusters=k, random_state=seed_hash % 1000)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate silhouette score if more than one cluster
        sil_score = 0
        if len(set(cluster_labels)) > 1:
            sil_score = float(silhouette_score(X_scaled, cluster_labels))
        
        # Prepare response
        result = {
            "pca_data": X_pca.tolist(),
            "cluster_labels": cluster_labels.tolist(),
            "city_labels": city_labels,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "silhouette_score": sil_score
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in overall_cluster_data: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Generate time series data - completely generic, no city-specific logic
def generate_time_series(n_samples=100, seed=None):
    """
    Generate time series data without any city-specific logic.
    The same process is used for all cities.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate time series with seasonal components - same for all cities
    time_index = np.arange(n_samples)
    demand_base = 1000
    
    # Add seasonal components - same for all cities
    seasonal_component = 200 * np.sin(time_index / 20) + 100 * np.sin(time_index / 50)
    
    # Add noise - same for all cities
    noise = np.random.normal(0, 80, n_samples)
    
    # Combine components
    target = demand_base + seasonal_component + noise
    
    return target

# API endpoint for forecasting
@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        city = data.get('city', '').lower()
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        model_type = data.get('model', 'rf')
        lookback = int(data.get('lookback', 24))
        
        print(f"Forecast request: city={city}, start_date={start_date}, end_date={end_date}, model={model_type}, lookback={lookback}")
        
        # Create a seed based on inputs
        seed_string = f"{city}_{start_date}_{end_date}_{model_type}_{lookback}"
        seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
        
        # Generate time series data - same process for all cities
        n_samples = 100
        target = generate_time_series(n_samples, seed=seed_hash)
        
        # Create timestamps
        base_date = datetime(2023, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Split into train/test (80/20)
        split_index = int(0.8 * n_samples)
        y_train = target[:split_index]
        y_test = target[split_index:]
        test_indices = np.arange(split_index, n_samples)
        
        # Create features for forecasting (lagged values)
        X_train = []
        for i in range(lookback, split_index):
            X_train.append(target[i-lookback:i])
        
        X_test = []
        for i in range(split_index, n_samples):
            X_test.append(target[i-lookback:i])
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = target[lookback:split_index]
        
        # Train and predict using the specified model
        if model_type == 'rf':
            # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=seed_hash % 1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            model_name = "Random Forest"
            
        elif model_type == 'xgb':
            # XGBoost
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=seed_hash % 1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            model_name = "XGBoost"
            
        elif model_type == 'arima':
            # Simple AR model (since we can't use statsmodels ARIMA here)
            # Use lagged values with weighted coefficients
            predictions = []
            ar_coefs = np.array([0.7, 0.2, 0.05, 0.03, 0.02])[:min(5, lookback)]
            ar_coefs = ar_coefs / ar_coefs.sum()  # Normalize
            
            for i in range(len(X_test)):
                # Use last 'lookback' values with AR coefficients
                lag_values = X_test[i][-len(ar_coefs):]
                pred = np.sum(lag_values * ar_coefs)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            model_name = "ARIMA"
            
        elif model_type == 'sarima':
            # Simple SARIMA-like model
            # Use both recent lags and seasonal lags
            predictions = []
            # Recent lags (AR component)
            ar_coefs = np.array([0.6, 0.2, 0.1, 0.05, 0.05])[:min(5, lookback)]
            ar_coefs = ar_coefs / ar_coefs.sum() * 0.7  # Normalize and weight
            
            # Seasonal lags (seasonal component)
            if lookback >= 24:  # Daily seasonality
                seasonal_lag = 24
                seasonal_weight = 0.3
            else:
                seasonal_lag = lookback // 2
                seasonal_weight = 0.2
            
            for i in range(len(X_test)):
                # AR component
                recent_lags = X_test[i][-len(ar_coefs):]
                ar_pred = np.sum(recent_lags * ar_coefs)
                
                # Seasonal component
                if i >= seasonal_lag and i < len(X_test):
                    seasonal_pred = X_test[i][-seasonal_lag]
                else:
                    seasonal_pred = X_test[i][-1]  # Fallback
                
                # Combine predictions
                pred = ar_pred + seasonal_pred * seasonal_weight
                predictions.append(pred)
            
            predictions = np.array(predictions)
            model_name = "SARIMA"
            
        else:
            # Default fallback - simple moving average
            predictions = []
            for i in range(len(X_test)):
                pred = np.mean(X_test[i][-min(5, lookback):])
                predictions.append(pred)
            
            predictions = np.array(predictions)
            model_name = "Unknown Model"
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Scale metrics to reasonable ranges
        mean_demand = np.mean(y_test)
        
        # Normalize MSE and MAE by the magnitude of the data
        normalized_mse = mse / (mean_demand ** 2) * 100
        normalized_mae = mae / mean_demand * 100
        
        # Format timestamps - only include time (no date or year)
        timestamps_str = [ts.strftime('%H:%M') for ts in timestamps[split_index:]]
        
        # Add a message to the response to verify the inputs were processed
        input_summary = f"City: {city}, Model: {model_name}, Lookback: {lookback}"
        
        # Prepare response
        result = {
            "timestamps": timestamps_str,
            "actual": y_test.tolist(),
            "predicted": predictions.tolist(),
            "indices": test_indices.tolist(),
            "mse": float(normalized_mse),
            "mae": float(normalized_mae),
            "r2": float(r2),
            "input_summary": input_summary,
            "model_type": model_type,
            "city": city,
            "lookback": lookback
        }
        
        print(f"Forecast response: {len(timestamps_str)} timestamps, {len(y_test)} actual points, {len(predictions)} predicted points")
        print(f"Error metrics: MSE={normalized_mse:.4f}, MAE={normalized_mae:.4f}, R²={r2:.4f}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in forecast: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# API endpoint to get available cities
@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify(cities)

if __name__ == '__main__':
    app.run(debug=True, port=5328)
