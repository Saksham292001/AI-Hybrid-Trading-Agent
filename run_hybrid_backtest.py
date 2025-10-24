# run_hybrid_backtest.py
import backtrader as bt
import pandas as pd
import pandas_ta as ta
import joblib
import warnings
import numpy as np
import os
from sklearn.exceptions import DataConversionWarning
import datetime

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.utils.validation')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='pandas_ta')
warnings.filterwarnings(action='ignore', category=UserWarning, message='Could not infer format*')

# --- CONFIGURATION ---
MODEL_FILENAME = "regime_model_oos.pkl"
DATA_DIR = "stock_data"
STOCK_TICKERS = [
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
# For faster testing, uncomment this line:
# STOCK_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS'] # Smaller test list

# --- Corrected Loading Function ---
def load_and_clean_yf_csv(filename, is_index=False):
    df_temp = pd.read_csv(filename, header=0, low_memory=False)
    # The 'auto_adjust=True' data from yfinance is clean,
    # so we just load it by the first column (index_col=0)
    df_temp = pd.read_csv(filename, index_col=0, parse_dates=True)
    df_temp.index.name = 'Date'
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX']:
         if col in df_temp.columns:
              df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    if not is_index:
         for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
             if col in df_temp.columns:
                  df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp.ffill(inplace=True)
    return df_temp

# --- Hybrid Strategy (Filter + Scanner) ---
class HybridStrategy(bt.Strategy):
    # --- THIS IS THE FIX ---
    # Parameters for the strategy are defined here
    params = (
        ('stop_loss_pct', 0.03),    # Use optimized 3%
        ('take_profit_pct', 0.12),   # Use optimized 12%
        ('rsi_period', 14),
        ('stoch_period', 14),
        ('sma_fast', 10),
        ('sma_slow', 30),
        ('max_positions', 10), # Hold up to 10 stocks
    )
    # --- END FIX ---

    def __init__(self):
        print("Initializing Hybrid Strategy...")
        try:
            self.model = joblib.load(MODEL_FILENAME)
        except Exception as e: print(f"FATAL ERROR loading model: {e}"); raise
        self.nifty = self.datas[0]
        self.vix = self.datas[1]
        self.stocks = self.datas[2:]
        self.nifty_rsi = bt.indicators.RSI(self.nifty.close, period=14)
        self.nifty_sma50 = bt.indicators.SimpleMovingAverage(self.nifty.close, period=50)
        self.nifty_atr = bt.indicators.ATR(self.nifty, period=14)
        self.nifty_adx = bt.indicators.ADX(self.nifty, period=14)
        self.vix_sma10 = bt.indicators.SimpleMovingAverage(self.vix.close, period=10)
        self.nifty_sentiment = (self.nifty.close / self.nifty.close(-5) - 1) * 100
        self.stock_rsi = {}
        self.stock_sma_fast = {}
        self.stock_sma_slow = {}
        self.stock_stoch = {}
        for d in self.stocks:
            self.stock_rsi[d._name] = bt.indicators.RSI(d.close, period=self.p.rsi_period)
            self.stock_sma_fast[d._name] = bt.indicators.SimpleMovingAverage(d.close, period=self.p.sma_fast)
            self.stock_sma_slow[d._name] = bt.indicators.SimpleMovingAverage(d.close, period=self.p.sma_slow)
            self.stock_stoch[d._name] = bt.indicators.Stochastic(d, period=self.p.stoch_period)
        self.live_orders = {}
        self.entry_prices = {}
        self.stop_loss_prices = {}
        self.take_profit_prices = {}
        self.position_count = 0
    def notify_order(self, order):
        stock_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            self.live_orders[stock_name] = order; return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_prices[stock_name] = order.executed.price
                # This line will now work because self.p.sl_pct is defined in params
                self.stop_loss_prices[stock_name] = self.entry_prices[stock_name] * (1.0 - self.p.sl_pct)
                self.take_profit_prices[stock_name] = self.entry_prices[stock_name] * (1.0 + self.p.tp_pct)
                self.position_count += 1
            elif order.issell():
                self.entry_prices.pop(stock_name, None)
                self.stop_loss_prices.pop(stock_name, None)
                self.take_profit_prices.pop(stock_name, None)
                self.position_count -= 1
            self.live_orders.pop(stock_name, None)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
             print(f"{self.nifty.datetime.date(0)} Order Failed/Rejected: {stock_name}")
             self.live_orders.pop(stock_name, None)
    def next(self):
        if self.datas[0].datetime.date(0) != self.nifty.datetime.date(0): return
        if len(self.nifty) < 60: return
        try:
            features = [
                self.nifty_rsi[0], self.nifty_atr[0],
                (self.nifty.close[0] - self.nifty_sma50[0]) / self.nifty_sma50[0],
                self.nifty_sentiment[0],
                self.vix.close[0], self.vix_sma10[0], self.nifty_adx[0]
            ]
            if any(np.isnan(f) for f in features): return
            current_regime = self.model.predict([features])[0]
        except Exception as e:
            print(f"Error in Agent 1 prediction: {e}"); return
        for d in self.stocks:
            stock_name = d._name
            position = self.getposition(d)
            if not position: continue
            current_price = d.close[0]
            if stock_name in self.live_orders: continue
            if current_price <= self.stop_loss_prices.get(stock_name, -1): self.sell(data=d)
            elif current_price >= self.take_profit_prices.get(stock_name, float('inf')): self.sell(data=d)
            elif current_regime == "BEAR_TREND" or current_regime == "PANIC_SELLOFF": self.sell(data=d)
            elif current_regime == "CHOPPY_RANGE" and self.stock_rsi[d._name][0] > 70: self.sell(data=d)
            elif current_regime == "BULL_TREND" and self.stock_sma_fast[d._name][0] < self.stock_sma_slow[d._name][0]: self.sell(data=d)
        if current_regime in ["BULL_TREND", "CHOPPY_RANGE"]:
            if self.position_count >= self.p.max_positions: return
            for d in self.stocks:
                stock_name = d._name
                if not self.getposition(d) and stock_name not in self.live_orders:
                    if self.position_count >= self.p.max_positions: break
                    try:
                        if current_regime == "BULL_TREND":
                            if self.stock_sma_fast[d._name][0] > self.stock_sma_slow[d._name][0] and self.stock_rsi[d._name][0] > 55:
                                self.buy(data=d); self.position_count += 1
                        elif current_regime == "CHOPPY_RANGE":
                            if self.stock_stoch[d._name].percK[0] < 20:
                                self.buy(data=d); self.position_count += 1
                    except IndexError: pass
    def stop(self):
        print(f"\n--- HYBRID BACKTEST COMPLETE ---")
        print(f"Final Portfolio Value: {self.broker.getvalue():,.2f}")

# --- Step 5: Run the Hybrid Backtest (OOS) ---
if __name__ == '__main__':
    # !!! IMPORTANT: Set this date based on the output from your training script !!!
    oos_start_date_str = '2023-09-08' # <<< SET YOUR DATE HERE (e.g., from last successful train run)
    # !!! ------------------------------------------------------------------ !!!
    try:
         oos_start_dt = pd.Timestamp(oos_start_date_str)
         print(f"OOS Start Date confirmed: {oos_start_dt.date()}")
    except ValueError:
         print(f"Error: Invalid OOS start date: '{oos_start_date_str}'. Use YYYY-MM-DD."); exit()

    print("Starting Hybrid Backtrader simulation (OOS)...")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(HybridStrategy)
    try:
        nifty_df = load_and_clean_yf_csv('NIFTY_50.csv', is_index=True)
        nifty_feed = bt.feeds.PandasData(dataname=nifty_df, name="NIFTY_50_Index", fromdate=oos_start_dt)
        cerebro.adddata(nifty_feed)
        vix_df = load_and_clean_yf_csv('INDIA_VIX.csv', is_index=True)
        vix_df = vix_df.reindex(nifty_df.index, method='ffill')
        vix_feed = bt.feeds.PandasData(dataname=vix_df, name="INDIA_VIX_Index", fromdate=oos_start_dt)
        cerebro.adddata(vix_feed)
        print("NIFTY 50 and VIX OOS data feeds added.")
    except Exception as e:
        print(f"Error loading NIFTY/VIX data feeds: {e}"); exit()
    print(f"Loading {len(STOCK_TICKERS)} stock data feeds for OOS period...")
    loaded_tickers_count = 0
    for ticker in STOCK_TICKERS:
        try:
            stock_df = load_and_clean_yf_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
            stock_df_aligned = stock_df.reindex(nifty_df.index, method='ffill')
            stock_df_aligned.dropna(inplace=True)
            if not stock_df_aligned.empty and stock_df_aligned.index.min() <= oos_start_dt:
                data_feed = bt.feeds.PandasData(dataname=stock_df_aligned, name=ticker, fromdate=oos_start_dt)
                cerebro.adddata(data_feed)
                loaded_tickers_count += 1
            else: print(f"Skipping {ticker}: No aligned data for OOS period.")
        except FileNotFoundError: print(f"Warning: Data file not found for {ticker}. Skipping.")
        except Exception as e: print(f"Error loading data for {ticker}: {e}. Skipping.")
    print(f"All data feeds loaded ({loaded_tickers_count} stocks).")
    cerebro.broker.set_cash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=5) 
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.04)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    print(f"\nRunning Hybrid Out-of-Sample backtest...")
    results = cerebro.run()
    print("\n--- HYBRID OOS RUN COMPLETE ---")
    print(f"\n--- FINAL HYBRID AI OOS REPORT ({oos_start_date_str} - Present) ---")
    if results:
        thestrat = results[0]
        start_value = cerebro.broker.startingcash
        final_value = thestrat.broker.getvalue()
        print(f"Starting Portfolio Value (OOS): {start_value:,.2f}")
        print(f"Final Portfolio Value (OOS):   {final_value:,.2f}")
        print(f"Total Profit/Loss (OOS):       {final_value - start_value:,.2f}")
        print("\n--- OOS Analyzer Results ---")
        try:
            trade_analysis = thestrat.analyzers.trades.get_analysis()
            sharpe_analysis = thestrat.analyzers.sharpe.get_analysis()
            drawdown_analysis = thestrat.analyzers.drawdown.get_analysis()
            trades = trade_analysis.get('total', {}).get('total', 0) if trade_analysis else 0
            wins = trade_analysis.get('won', {}).get('total', 0) if trade_analysis else 0
            losses = trade_analysis.get('lost', {}).get('total', 0) if trade_analysis else 0
            mdd = drawdown_analysis.get('max', {}).get('drawdown', 'N/A') if drawdown_analysis else 'N/A'
            sharpe = sharpe_analysis.get('sharperatio', 'N/A') if sharpe_analysis else 'N/A'
            oos_return = ((final_value / start_value) - 1) * 100 if start_value else 0
            print(f"Total Trades (OOS): {trades}")
            print(f"Winning Trades (OOS): {wins}")
            print(f"Losing Trades (OOS): {losses}")
            mdd_display = f"{mdd:.2f}%" if isinstance(mdd, (int, float)) else mdd
            print(f"Max Drawdown (OOS): {mdd_display}")
            sharpe_display = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else sharpe
            print(f"Annualized Sharpe Ratio (OOS): {sharpe_display}")
            print(f"Total Return % (OOS): {oos_return:.2f}%")
        except Exception as e:
            print(f"An unexpected error occurred processing OOS analyzers: {e}")
        print("\nGenerating final plot (OOS Period)...")
        try: cerebro.plot(style='candlestick', barup='green', bardown='red', numfigs=1)
        except Exception as plot_err: print(f"Error generating plot: {plot_err}")
    else:
        print("OOS Backtest did not produce results.")
