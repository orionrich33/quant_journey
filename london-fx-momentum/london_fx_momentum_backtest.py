import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------------
# Fetch FX Data
# -------------------------------
def fetch_fx_data(ticker, start_date, end_date):
    """Download 1-minute FX data, convert to London timezone, and flatten columns."""
    raw = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1m'
    )
    # Timezone handling
    if raw.index.tz is None:
        data = raw.tz_localize('UTC').tz_convert('Europe/London')
    else:
        data = raw.tz_convert('Europe/London')
    # Flatten MultiIndex columns (keep level0 names)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# -------------------------------
# Get Price at or Near a Given Time
# -------------------------------
def get_price_at_time(day_data, hour, minute, window=5):
    """
    Return the Open price exactly at time or within the next `window` minutes.
    Raises KeyError if no data in that interval.
    """
    # exact timestamp
    exact = day_data.between_time(f"{hour:02d}:{minute:02d}", f"{hour:02d}:{minute:02d}")
    if not exact.empty:
        return exact['Open'].iloc[0]
    # fallback window
    start_min = minute
    end_min = minute + window
    window_df = day_data.between_time(f"{hour:02d}:{start_min:02d}", f"{hour:02d}:{end_min:02d}")
    if not window_df.empty:
        return window_df['Open'].iloc[0]
    raise KeyError(f"No price for {hour:02d}:{minute:02d} ±{window}m")

# -------------------------------
# Main Backtest Logic
# -------------------------------
def main():
    # Parameters
    ticker     = 'EURUSD=X'
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=7)

    # Fetch and inspect data
    data = fetch_fx_data(ticker, start_date, end_date)
    print("\nData sample (first 5 rows):")
    print(data.head())

    # Restrict to London morning session (07:00–10:00)
    morning = data.between_time('07:00', '10:00')
    print("\nMorning session sample:")
    print(morning.head())
    unique_days = pd.unique(morning.index.date)
    print("\nDays in dataset:", unique_days)

    # Run backtest
    trades = []
    for day in unique_days:
        day_slice = morning[morning.index.date == day]
        try:
            p7  = get_price_at_time(day_slice, 7, 0)
            p8  = get_price_at_time(day_slice, 8, 0)
            p10 = get_price_at_time(day_slice, 10, 0)
            direction = 'long' if p8 > p7 else 'short'
            ret = (p10 - p8) / p8 if direction == 'long' else (p8 - p10) / p8
            trades.append({'date': day, 'direction': direction, 'return': ret})
            print(f"{day}: {direction} | ret={ret:.5f}")
        except KeyError as ke:
            print(f"Skipped {day}: {ke}")
        except Exception as e:
            print(f"Error on {day}: {e}")

    # Compile results
    results_df = pd.DataFrame(trades)
    if results_df.empty:
        print("\nNo trades executed. Please verify data availability and time filters.")
        return

    # Summary
    avg_ret = results_df['return'].mean() * 100
    win_rt  = (results_df['return'] > 0).mean() * 100
    print("\n--- STRATEGY RESULTS ---")
    print(f"Average daily return: {avg_ret:.4f}%")
    print(f"Win rate: {win_rt:.2f}%")
    print("Detailed trades:")
    print(results_df)

    # Plot distribution
    plt.hist(results_df['return'], bins=15, edgecolor='black')
    plt.title('Distribution of Daily Returns (London FX Momentum)')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()


