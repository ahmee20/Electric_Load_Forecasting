import pandas as pd

from sklearn.preprocessing import StandardScaler
df_cleaned = pd.read_csv("combined_data.csv", parse_dates=["timestamp"])

# Check schema and sample records
print("Schema:\n", df_cleaned.dtypes)
print("\nSample Records:\n", df_cleaned.head())
# Check for missing values
missing_summary = df_cleaned.isnull().sum()
print("Missing Values:\n", missing_summary)


df_cleaned['temperature'].fillna(df_cleaned['temperature'].mean(), inplace=True)
df_cleaned['humidity'].fillna(df_cleaned['humidity'].median(), inplace=True)
df_cleaned['windSpeed'].fillna(df_cleaned['windSpeed'].mean(), inplace=True)
df_cleaned.dropna(subset=["demand_mwh"], inplace=True)
print("Missing Values After Imputation:\n", df_cleaned.isnull().sum())
# Extract time-based features
df_cleaned["hour"] = df_cleaned["timestamp"].dt.hour
df_cleaned["day_of_week"] = df_cleaned["timestamp"].dt.dayofweek
df_cleaned["month"] = df_cleaned["timestamp"].dt.month
df_cleaned["season"] = df_cleaned["month"] % 12 // 3 + 1  # 1=Winter, ..., 4=Fall

# Scale selected columns
scaler = StandardScaler()
df_cleaned[["temperature", "humidity", "windSpeed", "demand_mwh"]] = (
    scaler.fit_transform(
        df_cleaned[["temperature", "humidity", "windSpeed", "demand_mwh"]]
    )
)
# Add date column
df_cleaned["date"] = df_cleaned["timestamp"].dt.date

# Compute daily summary statistics
daily_summary = (
    df_cleaned.groupby(["city", "date"])
    .agg(
        {
            "temperature": "mean",
            "humidity": "mean",
            "windSpeed": "mean",
            "demand_mwh": ["mean", "max", "min", "std"],
        }
    )
    .reset_index()
)

daily_summary.head()
import numpy as np
from scipy import stats

z_scores = np.abs(
    stats.zscore(df_cleaned[["temperature", "humidity", "windSpeed", "demand_mwh"]])
)
z_thresh = 3
z_outliers = (z_scores > z_thresh).any(axis=1)
df_cleaned["anomaly_zscore"] = z_outliers


from sklearn.ensemble import IsolationForest

# Isolation Forest model
iso_forest = IsolationForest(contamination=0.01, random_state=42)
anomaly_preds = iso_forest.fit_predict(
    df_cleaned[["temperature", "humidity", "windSpeed", "demand_mwh"]]
)
df_cleaned["anomaly_iso"] = anomaly_preds == -1
# Select anomalies using boolean indexing on the full DataFrame
anomalies = df_cleaned[(df_cleaned["anomaly_zscore"]) | (df_cleaned["anomaly_iso"])]

# Show number of anomalies
print(f"Total Anomalies Detected: {anomalies.shape[0]}")

# Show selected columns from the anomalies DataFrame
anomalies_display = anomalies[
    ["timestamp", "city", "temperature", "humidity", "windSpeed", "demand_mwh"]
]
anomalies_display.head()
# Remove rows with impossible (non-physical) values
df_cleaned_cleaned = df_cleaned[
    (df_cleaned["humidity"] >= 0)
    & (df_cleaned["humidity"] <= 1)
    & (df_cleaned["windSpeed"] >= 0)
    & (df_cleaned["demand_mwh"] >= 0)
]
# Select relevant features for clustering
features = df_cleaned_cleaned[["temperature", "humidity", "windSpeed", "demand_mwh"]]

# Normalize for clustering
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
# PCA for visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title("PCA Projection of Weather + Demand Features")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method to find optimal k
inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow + Silhouette Score
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, inertia, marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")

ax[1].plot(K_range, sil_scores, marker="o", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Score")

plt.tight_layout()
plt.show()
k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

# Visualize clusters in PCA space
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=k_labels, palette="Set2")
plt.title("K-Means Clusters (PCA Projection)")
plt.show()
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.7, min_samples=10)
db_labels = dbscan.fit_predict(X_scaled)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=db_labels, palette="tab10")
plt.title("DBSCAN Clustering")
plt.show()

# Count noise points
import numpy as np

print(f"Noise points (label = -1): {(db_labels == -1).sum()}")
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
# Silhouette Score
print("K-Means Silhouette Score:", silhouette_score(X_scaled, k_labels))
print("DBSCAN Silhouette Score:", silhouette_score(X_scaled, db_labels))
# Append K-Means labels for analysis
df_cleaned_cleaned["cluster"] = k_labels

# Group by cluster to analyze patterns
cluster_summary = df_cleaned_cleaned.groupby("cluster").agg(
    {
        "temperature": "mean",
        "humidity": "mean",
        "windSpeed": "mean",
        "demand_mwh": ["mean", "count"],
    }
)
cluster_summary
import pandas as pd
import numpy as np

df = df_cleaned.copy()

df.drop(["anomaly_zscore", "anomaly_iso"], axis=1, inplace=True)
df.head()
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by timestamp
df = df.sort_values("timestamp")

# One-hot encode 'city'
df_encoded = pd.get_dummies(df, columns=["city"], drop_first=True)

# Drop unused columns
X = df_encoded.drop(columns=["demand_mwh", "timestamp", "date"])
y = df_encoded["demand_mwh"]

# Chronological time-based split (e.g., 80% train, 20% test)
split_index = int(0.8 * len(df_encoded))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Linear Regression")
mselinear= mean_squared_error(y_test, y_pred_lr)
mae_linear= mean_absolute_error(y_test, y_pred_lr)
r2_linear= r2_score(y_test, y_pred_lr)

print("MSE : ", mselinear)
print("MAE : ", mae_linear)
print("R2 : ", r2_linear)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest")
mserf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("MSE : ", mserf)
print("MAE : ", mae_rf)
print("R2 : ", r2_rf)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Use 5-fold time series split
tscv = TimeSeriesSplit(n_splits=5)

# Define model
rf = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

# Grid Search with TimeSeriesSplit
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

# Run grid search
grid_search_rf.fit(X_train, y_train)

# Best model and score
print("Best RF Params:", grid_search_rf.best_params_)
print("Best RF CV MSE:", -grid_search_rf.best_score_)

# Evaluate on test set
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
xg_model = xgb.XGBRegressor(random_state=42)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)

print("XGBoost")
msexg = mean_squared_error(y_test, y_pred_xg)
mae_xg = mean_absolute_error(y_test, y_pred_xg)
r2_xg = r2_score(y_test, y_pred_xg)

print("MSE : ", msexg)
print("MAE : ", mae_xg)
print("R2 : ", r2_xg)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xg = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")

param_grid_xg = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1],
}

grid_search_xg = GridSearchCV(
    estimator=xg,
    param_grid=param_grid_xg,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

grid_search_xg.fit(X_train, y_train)

print("Best XGBoost Params:", grid_search_xg.best_params_)
print("Best XGBoost CV MSE:", -grid_search_xg.best_score_)

y_pred_xg = grid_search_xg.best_estimator_.predict(X_test)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# One-hot encode 'city'
df_encoded = pd.get_dummies(df, columns=["city"], drop_first=True)

# Drop non-numeric columns
features = df_encoded.drop(columns=["demand_mwh", "timestamp", "date"])
target = df_encoded["demand_mwh"]

# Normalize features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Prepare train-test split (time-based)
split_index = int(0.8 * len(df_encoded))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# Set time window
window_size = 24  # e.g., 24 hours
X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)
# LSTM Model
# Define LSTM model
model_lstm = Sequential(
    [
        LSTM(64, input_shape=(window_size, X_train.shape[1]), return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1),
    ]
)
model_lstm.compile(optimizer=Adam(0.001), loss="mse")

# Train LSTM
model_lstm.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1)
# Predict and evaluate
y_pred_lstm = model_lstm.predict(X_test_seq)
y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
y_test_inv = scaler_y.inverse_transform(y_test_seq)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


print("LSTM Performance")

mselstm = mean_squared_error(y_test_inv, y_pred_lstm_inv)
maelstm = mean_absolute_error(y_test_inv, y_pred_lstm_inv)
r2lstm = r2_score(y_test_inv, y_pred_lstm_inv)

print("MSE: ", mselstm)
print("MAE: ", maelstm)
print("R2: ", r2lstm)
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reset index to ensure consistent shifting
df_sorted = df.sort_values("timestamp").reset_index(drop=True)

# Naive forecast: use previous day's same hour demand
df_sorted["naive_forecast"] = df_sorted["demand_mwh"].shift(24)

df_naive = df_sorted.dropna(subset=["naive_forecast"])

# MAE
mae_naive = mean_absolute_error(df_naive["demand_mwh"], df_naive["naive_forecast"])

# RMSE (Note: squared=False only works in sklearn >=0.22)
rmse_naive = mean_squared_error(
    df_naive["demand_mwh"], df_naive["naive_forecast"]
)

# MAPE - handle divide-by-zero issues by filtering out near-zero actual values
df_filtered = df_naive[df_naive["demand_mwh"] > 1.0]  # Adjust threshold if needed
mape_naive = (
    np.mean(
        np.abs(
            (df_filtered["demand_mwh"] - df_filtered["naive_forecast"])
            / df_filtered["demand_mwh"]
        )
    )
    * 100
)

# SMAPE (optional but more stable)
smape = 100 * np.mean(
    2
    * np.abs(df_naive["naive_forecast"] - df_naive["demand_mwh"])
    / (np.abs(df_naive["naive_forecast"]) + np.abs(df_naive["demand_mwh"]))
)

# Print results
print("Naive Forecast Performance:")
print(f"MAE: {mae_naive}")
print(f"RMSE: {rmse_naive}")
print(f"MAPE (filtered): {mape_naive}")
print(f"SMAPE: {smape}")
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Define base learners and meta-learner
base_models = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xg', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
]

meta_model = LinearRegression()

# Create stacking model
stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)

# Fit stacking model
stacked_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stacked_model.predict(X_test)

print("Stacked Model Performance")
print("MSE:", mean_squared_error(y_test, y_pred_stack))
print("MAE:", mean_absolute_error(y_test, y_pred_stack))
print("R2:", r2_score(y_test, y_pred_stack))

from sklearn.metrics import mean_absolute_percentage_error
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_stack))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


df_arima_clean = pd.read_csv("combined_data.csv")

df_arima_clean["temperature"].fillna(df_arima_clean["temperature"].mean(), inplace=True)
df_arima_clean["humidity"].fillna(df_arima_clean["humidity"].median(), inplace=True)
df_arima_clean["windSpeed"].fillna(df_arima_clean["windSpeed"].mean(), inplace=True)
df_arima_clean.dropna(subset=["demand_mwh"], inplace=True)
df_arima_clean = df_arima_clean[
    df_arima_clean["demand_mwh"] >= 0
]  # Remove negative demand values

# Ensure timestamp is datetime
df_arima_clean["timestamp"] = pd.to_datetime(df_arima_clean["timestamp"])
def train_arima_models(df):
    cities = df["city"].unique()
    arima_results = {}

    for city in cities:
        city_df = df[df["city"] == city].sort_values("timestamp")
        demand_series = city_df["demand_mwh"].values

        # Train/test split (e.g., last 20% as test)
        split_idx = int(0.8 * len(demand_series))
        train, test = demand_series[:split_idx], demand_series[split_idx:]

        # Fit ARIMA
        try:
            model = ARIMA(train, order=(5, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=len(test))
        except Exception as e:
            print(f"ARIMA failed for {city}: {e}")
            continue

        # Metrics
        mae = mean_absolute_error(test, forecast)
        rmse = mean_squared_error(test, forecast)
        mape = np.mean(np.abs((test - forecast) / test)) * 100

        arima_results[city] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
        }

    return arima_results
print("Training ARIMA models...\n")
arima_results = train_arima_models(df_arima_clean)
print("ARIMA Results:")
for city, metrics in arima_results.items():
    print(
        f"{city}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, MAPE={metrics['MAPE']}%"
    )
import matplotlib.pyplot as plt


def plot_forecast(city_name, df):
    city_df = df[df["city"] == city_name].sort_values("timestamp")
    demand_series = city_df["demand_mwh"].values

    split_idx = int(0.8 * len(demand_series))
    train, test = demand_series[:split_idx], demand_series[split_idx:]

    model = ARIMA(train, order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))

    plt.figure(figsize=(12, 4))
    plt.plot(city_df["timestamp"].values[split_idx:], test, label="Actual")
    plt.plot(
        city_df["timestamp"].values[split_idx:],
        forecast,
        label="Forecast",
        linestyle="--",
    )
    plt.title(f"{city_name} – ARIMA Forecast vs Actual")
    plt.legend()
    plt.show()


plot_forecast("la", df_arima_clean)
plot_forecast("phoenix", df_arima_clean)
plot_forecast("san diego", df_arima_clean)
plot_forecast("san jose", df_arima_clean)
plot_forecast("seattle", df_arima_clean)
plot_forecast("nyc", df_arima_clean)
plot_forecast("philadelphia", df_arima_clean)
plot_forecast("dallas", df_arima_clean)
plot_forecast("houston", df_arima_clean)
plot_forecast("san antonio", df_arima_clean)

def train_sarima_models(df):
    cities = df["city"].unique()
    sarima_results = {}

    for city in cities:
        city_df = df[df["city"] == city].sort_values("timestamp")
        demand_series = city_df["demand_mwh"].values

        # Train/test split (e.g., last 20% as test)
        split_idx = int(0.8 * len(demand_series))
        train, test = demand_series[:split_idx], demand_series[split_idx:]

        # Fit SARIMA
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=len(test))
        except Exception as e:
            print(f"SARIMA failed for {city}: {e}")
            continue

        # Metrics
        mae = mean_absolute_error(test, forecast)
        rmse = mean_squared_error(test, forecast)
        mape = np.mean(np.abs((test - forecast) / test)) * 100

        sarima_results[city] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
        }

    return sarima_results

print("\nTraining SARIMA models...\n")
sarima_results = train_sarima_models(df_arima_clean)
print("SARIMA Results:")
for city, metrics in sarima_results.items():
    print(
        f"{city}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, MAPE={metrics['MAPE']}%"
    )