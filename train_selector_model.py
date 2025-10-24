# train_selector_model.py
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import math
import os

# --- Suppress Warnings ---
warnings.filterwarnings(action='ignore', category=FutureWarning, module='pandas_ta')
warnings.filterwarnings(action='ignore', category=UserWarning, message='Could not infer format*')

print("--- Starting Model Training (v10 / v8 ADX Logic - OOS Split) ---")

# --- 1. Load and Merge Data ---
try:
    # --- Robust Loading Function (Handles clean yfinance data) ---
    def load_and_clean_yf_csv(filename, is_index=False):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Required file not found: {filename}")
        
        # yfinance auto_adjust=True data is clean, so we just load
        df_temp = pd.read_csv(filename, index_col=0, parse_dates=True)
        df_temp.index.name = 'Date'

        # Convert all other relevant columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX']:
             if col in df_temp.columns:
                  df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        return df_temp

    df_nifty = load_and_clean_yf_csv('NIFTY_50.csv', is_index=True)
    df_vix_raw = load_and_clean_yf_csv('INDIA_VIX.csv', is_index=True)
    
    df_vix = df_vix_raw[['Close']].rename(columns={'Close': 'VIX'})
    df = pd.merge(df_nifty, df_vix, left_index=True, right_index=True, how='inner')
    
    print(f"Merged NIFTY and VIX data. Shape: {df.shape}")
    if df.empty: raise ValueError("DataFrame empty after merging.")
    if not isinstance(df.index, pd.DatetimeIndex): # Check if index is datetime
         raise TypeError("Index is not DatetimeIndex after loading. Check CSV format.")
    print("Data loading complete.")
    
except FileNotFoundError as e: print(f"Error: {e}. Run get_stock_universe.py first."); exit()
except Exception as e: print(f"Error loading/merging: {e}"); exit()

# --- Convert all columns to numeric AFTER loading ---
print("Converting loaded data to numeric...")
for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX']:
     if col in df.columns:
          df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 2. Feature Engineering ---
print("Calculating features (incl. ADX & Sentiment Proxy)...")
df.ffill(inplace=True) # Fill any NaNs from coerce or loading
df['Sentiment_Score'] = df['Close'].pct_change(periods=5) * 100
df['Sentiment_Score'] = ta.sma(df['Sentiment_Score'], length=3)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['ATR_14'] = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
df.ta.adx(length=14, append=True); df.rename(columns={'ADX_14': 'ADX'}, inplace=True)
df.dropna(subset=['SMA_50', 'ATR_14', 'RSI_14', 'VIX', 'ADX'], inplace=True)
df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
df['VIX_SMA_10'] = ta.sma(df['VIX'], length=10)
df.dropna(inplace=True)
print(f"Calculated features. Data shape: {df.shape}")
if df.empty: print("Error: DataFrame empty after features."); exit()

# --- 3. Data Labeling (Using ADX Logic v8) ---
print("Creating regime labels (v8 ADX-based)...")
ADX_THRESHOLD = 22
def label_regime_broad_adx_v11(row):
    if row['VIX'] > 25 and row['VIX'] > row['VIX_SMA_10'] * 1.2: return "PANIC_SELLOFF"
    if row['ADX'] < ADX_THRESHOLD: return "CHOPPY_RANGE"
    if row['ADX'] > ADX_THRESHOLD:
        if row['RSI_14'] > 55: return "BULL_TREND"
        elif row['RSI_14'] < 45: return "BEAR_TREND"
        else: return "CHOPPY_RANGE"
    return "CHOPPY_RANGE"
df['Regime'] = df.apply(label_regime_broad_adx_v11, axis=1)
print(f"\nRegime counts (Full Data):\n{df['Regime'].value_counts()}\n")

# --- 4. Model Training (Train on first 80%, Hold Out Last 20%) ---
print("Preparing data for scikit-learn...")
features = ['RSI_14', 'ATR_14', 'Price_vs_SMA50', 'Sentiment_Score', 'VIX', 'VIX_SMA_10', 'ADX']
target = 'Regime'
X = df[features]
y = df[target]

test_size_fraction = 0.20
split_index = math.floor(len(df) * (1 - test_size_fraction))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_oos_test = X.iloc[split_index:]
y_oos_test = y.iloc[split_index:]

if not isinstance(X_train.index, pd.DatetimeIndex):
     print("Error: Index is not DatetimeIndex after loading/splitting.")
     exit()

oos_start_date = X_oos_test.index.min()
print(f"Training model on {len(X_train)} days (up to {X_train.index.max().date()})...")
print(f"Holding out {len(X_oos_test)} days for OOS test (from {oos_start_date.date()}).")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
print("\nModel training complete!")

# --- 5. Model Evaluation (Optional: On OOS period labels) ---
print("Evaluating trained model performance on held-out OOS data period...")
y_pred_oos = model.predict(X_oos_test)
accuracy_oos = accuracy_score(y_oos_test, y_pred_oos)
print(f"\n--- Model Accuracy on OOS Data Period: {accuracy_oos * 100:.2f}% ---")
print("\nClassification Report (OOS Data Period):")
print(classification_report(y_oos_test, y_pred_oos, zero_division=0))

# --- 6. Save the TRAINED Model ---
model_filename = "regime_model_oos.pkl"
joblib.dump(model, model_filename)
print(f"\nSuccess! Your *OOS* 'Selector' brain saved as '{model_filename}'")
print(f"Use this model in the backtester with fromdate='{oos_start_date.strftime('%Y-%m-%d')}'")
