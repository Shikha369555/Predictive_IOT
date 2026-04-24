import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
# Load data
df = pd.read_csv("data.csv")
# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
# Rolling features
df['temp_mean_3'] = df['temperature'].rolling(window=3).mean()
df['vib_std_3'] = df['vibration'].rolling(window=3).std()
# Lag features
df['temp_lag1'] = df['temperature'].shift(1)
df['temp_lag2'] = df['temperature'].shift(2)
# Drop NaN values
df = df.dropna()