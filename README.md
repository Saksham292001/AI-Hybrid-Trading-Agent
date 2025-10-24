# AI Hybrid Trading Agent

This project is a complete "from-scratch" AI-powered trading agent built in Python. It uses a hybrid, two-agent architecture to trade stocks from the NIFTY 50 universe.

-   **Agent 1 (Regime Filter):** A Random Forest model trained on NIFTY 50 and INDIA VIX data to classify the overall market "weather" into one of four regimes: `BULL_TREND`, `CHOPPY_RANGE`, `BEAR_TREND`, or `PANIC_SELLOFF`.
-   **Agent 2 (Stock Scanner):** A rule-based scanner that, given a "Risk-On" signal from Agent 1, scans NIFTY 50 stocks for specific buy signals based on the regime (Momentum or Mean Reversion).

This strategy was backtested and validated on Out-of-Sample (OOS) data, demonstrating a robust, profitable (though low-frequency) trading model.

---

## Architecture

1.  **Regime Filter (AI Brain):** The AI model (`regime_model_oos.pkl`) is the core. It decides *if* we should be trading.
2.  **Stock Scanner (Algorithms):**
    * If Regime = `BULL_TREND`, scan for stocks using a strict momentum signal (SMA Crossover + RSI > 55).
    * If Regime = `CHOPPY_RANGE`, scan for stocks using a mean-reversion signal (Stochastic %K < 20).
    * If Regime = `BEAR_TREND` or `PANIC_SELLOFF`, sell all positions and do not enter new trades.
3.  **Risk Management:**
    * **Position Sizing:** Each trade is sized to risk a maximum of 2% of total portfolio value.
    * **Exit Rules:** Each trade has a 3% Stop-Loss and a 12% Take-Profit.
    * **Portfolio Limit:** The agent will hold a maximum of 10 different stocks at one time.

---

## Files in This Repository

* **`get_stock_universe.py`**: Downloads all necessary historical data for the NIFTY 50 index, INDIA VIX, and all 50 component stocks from `yfinance`.
* **`train_selector_model.py`**: Loads the NIFTY 50/VIX data, engineers features (RSI, ADX, SMA, etc.), labels regimes, trains the Random Forest model (`regime_model_oos.pkl`) on the first 80% of data, and validates it.
* **`run_hybrid_backtest.py`**: The main script. It loads the trained model, loads all 50+ data feeds, and runs a full Out-of-Sample (OOS) backtest (on the 20% of data the model never saw) using the hybrid "Filter + Scanner" strategy.
* **`scrape_news.py`**: A utility module for scraping and analyzing real-time news sentiment. (This is for the *next* phase of development: live trading).
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
