# AI Hybrid Trading Agent

This project is a complete "from-scratch" AI-powered trading agent built in Python. It uses a hybrid, two-agent architecture to trade stocks from the NIFTY 50 universe, backtested on historical data from `yfinance`.

-   **Agent 1 (Regime Filter):** A Random Forest model (`regime_model_oos.pkl`) trained on the NIFTY 50 and INDIA VIX. It classifies the overall market "weather" into one of four regimes: `BULL_TREND`, `CHOPPY_RANGE`, `BEAR_TREND`, or `PANIC_SELLOFF`.
-   **Agent 2 (Stock Scanner):** A rule-based scanner that, given a "Risk-On" signal from Agent 1, scans NIFTY 50 stocks for specific buy signals based on the regime.

This strategy was developed iteratively and validated on an Out-of-Sample (OOS) data period (from 2023-09-08 onwards) to test for robustness.

## Architecture

1.  **Regime Filter (AI Brain):** The AI model decides *if* we should be trading.
2.  **Stock Scanner (Algorithms):**
    * If Regime = `BULL_TREND`, scan for stocks using a strict momentum signal (SMA 10/30 Crossover + RSI > 55).
    * If Regime = `CHOPPY_RANGE`, scan for stocks using a mean-reversion signal (Stochastic %K < 20).
    * If Regime = `BEAR_TREND` or `PANIC_SELLOFF`, sell all positions and do not enter new trades.
3.  **Risk Management:**
    * **Position Sizing:** Each trade is sized to a fixed 5% of the total portfolio value.
    * **Exit Rules:** Each trade has an individual 3% Stop-Loss and 12% Take-Profit.
    * **Portfolio Limit:** The agent will hold a maximum of 10 different stocks at one time.

## Files in This Repository

* **`get_stock_universe.py`**: Downloads all necessary historical data for the NIFTY 50 index, INDIA VIX, and all 50 component stocks.
* **`train_selector_model.py`**: Loads the NIFTY 50/VIX data, engineers features, labels regimes, and trains the `regime_model_oos.pkl` on the first 80% of data.
* **`run_hybrid_backtest.py`**: The main script. It loads the trained model, loads all 50+ data feeds, and runs the final Out-of-Sample (OOS) backtest on the held-out 20% of data.
* **`scrape_news.py`**: A utility module for scraping and analyzing real-time news sentiment (for future live implementation).
* **`requirements.txt`**: A list of all necessary Python libraries.

---

## Setup and Execution

### 1. Setup Environment

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt

2. Download All Data
Run the data downloader. This will take 5-10 minutes to create the stock_data/ folder and download all CSVs.

Bash

python get_stock_universe.py
3. Train the AI Brain
Run the training script. This will create the regime_model_oos.pkl file.

Bash

python train_selector_model.py
This will print an OOS start date (e.g., fromdate='2023-09-08').

4. Run the Final Backtest
Open run_hybrid_backtest.py.

Find the line oos_start_date_str = 'YYYY-MM-DD'.

Replace 'YYYY-MM-DD' with the actual date from the previous step (e.g., '2023-09-08').

Save the file.

Run the backtest (this will be slow as it processes 50+ data feeds):

Bash

python run_hybrid_backtest.py
This will run the full Out-of-Sample test and generate the final plot and performance report.


---

### `requirements.txt`

Create a new file named `requirements.txt` and paste this in:

backtrader pandas pandas-ta joblib scikit-learn yfinance matplotlib requests beautifulsoup4 transformers torch


---

### `get_stock_universe.py`

Create a new file named `get_stock_universe.py` and paste this code:

```python
# get_stock_universe.py
import yfinance as yf
import pandas as pd
import os
import time

# List of NIFTY 50 tickers (as of Oct 2025), formatted for yfinance
NIFTY_50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'BHARTIARTL.NS',
    'SBIN.NS', 'HINDUNILVR.NS', 'LTIM.NS', 'BAJFINANCE.NS', 'ITC.NS', 'KOTAKBANK.NS',
    'HCLTECH.NS', 'AXISBANK.NS', 'LT.NS', 'TATAMOTORS.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'TITAN.NS', 'WIPRO.NS', 'ONGC.NS',
    'NTPC.NS', 'ADANIENT.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'COALINDIA.NS', 'POWERGRID.NS',
    'ULTRACEMCO.NS', 'SBILIFE.NS', 'GRASIM.NS', 'NESTLEIND.NS', 'HDFCLIFE.NS',
    'ADANIPORTS.NS', 'TECHM.NS', 'DRREDDY.NS', 'HINDALCO.NS', 'CIPLA.NS',
    'SHRIRAMFIN.NS', 'BAJAJ-AUTO.NS', 'INDUSINDBK.NS', 'BRITANNIA.NS', 'APOLLOHOSP.NS',
    'EICHERMOT.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'BPCL.NS', 'SHREECEM.NS'
]

# Index tickers
INDEX_TICKERS = {
    "NIFTY_50": "^NSEI",     # Market direction
    "INDIA_VIX": "^INDIAVIX" # Market volatility
}

START_DATE = "2015-01-01"
END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
DATA_DIR = "stock_data" # Folder to store the 50 stock CSVs

# Create directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")

# --- 1. Download Index Data ---
print(f"--- Downloading Index Data ---")
for name, ticker in INDEX_TICKERS.items():
    print(f"Downloading {name} ({ticker})...")
    try:
        # auto_adjust=True gives clean data (no metadata rows)
        data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if not data.empty:
            output_filename = f"{name}.csv"
            data.to_csv(output_filename)
            print(f"Success! Saved {output_filename}")
        else:
            print(f"No data for {name}.")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
    time.sleep(1)

# --- 2. Download Stock Universe Data ---
print(f"\n--- Downloading Stock Universe Data ({len(NIFTY_50_TICKERS)} stocks) ---")

for ticker in NIFTY_50_TICKERS:
    print(f"Downloading {ticker}...")
    output_filename = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    if os.path.exists(output_filename):
        print(f"... {ticker} already exists. Skipping.")
        continue
        
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if not data.empty:
            data.to_csv(output_filename)
            print(f"Success! Saved {ticker}.csv")
        else:
            print(f"No data for {ticker}.")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
    
    time.sleep(1)

print("\n--- Stock Universe Download Complete ---")

## Final Commands
Run the training script (to create regime_model_oos.pkl):

Bash

python train_selector_model.py
Note the OOS start date it prints (e.g., 2023-09-08).

Edit run_hybrid_backtest.py and replace 'YYYY-MM-DD' in the oos_start_date_str variable with the date from step 2. Save the file.

Run the hybrid backtest:

Bash

python run_hybrid_backtest.py
