import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv(r"c:\Users\pc\Downloads\AQI_dataset.csv")

# Basic info
print("Dataset Shape:", df.shape)
print(df.head())

# Convert Date column to datetime and extract seasonal features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda m: 
    'Winter' if m in [12,1,2] else 
    'Summer' if m in [3,4,5] else 
    'Monsoon' if m in [6,7,8] else 'Post-Monsoon')

# Define features and target
X = df[['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','WindSpeed']]
y = (0.5*df['PM2.5'] + 0.3*df['PM10'] + 0.1*df['NO2'] + 0.1*df['SO2'])  # Approx AQI formula

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation - compute RMSE manually
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", rmse)
print("R²  :", r2_score(y_test, y_pred))

# Trend plots
plt.figure(figsize=(10,5))
plt.plot(df['Date'], y, label='AQI Trend', color='red')
plt.title("Air Quality Index Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.show()

# Seasonal Analysis
season_avg = df.groupby('Season')[['PM2.5','PM10','NO2','SO2']].mean()
season_avg.plot(kind='bar', figsize=(10,5))
plt.title("Average Pollution Levels per Season")
plt.ylabel("Concentration (µg/m³)")
plt.show()

# Weather impact
plt.scatter(df['Temperature'], y, color='orange', label='Temperature vs AQI')
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")
plt.legend()
plt.show()

plt.scatter(df['Humidity'], y, color='blue', label='Humidity vs AQI')
plt.xlabel("Humidity (%)")
plt.ylabel("AQI")
plt.legend()
plt.show()