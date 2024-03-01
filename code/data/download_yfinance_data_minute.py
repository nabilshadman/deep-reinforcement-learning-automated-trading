# import relevant libraries
import yfinance as yf
import pandas as pd


# define function to download equity prices from yfinance
def download_close_prices(tickers, interval, start_date, end_date):
    close_prices = {}
    for ticker in tickers:
        try:
            # download data
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            # extract close prices
            close_prices[ticker] = data['Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    return close_prices


# list of tickers
tickers = ["DIA", "VOO", "QQQ"] # DOW30, S&P500, and NASDAQ ETFs

# define interval, start date, and end date
interval = "1m"                # 1 minute interval
start_date = "2024-02-05"      # start date
end_date = "2024-02-09"        # end date

# download close prices
close_prices = download_close_prices(tickers, interval, start_date, end_date)

# combine close prices into a single DataFrame
df_close_prices = pd.DataFrame(close_prices)

# save the DataFrame to a CSV file
df_close_prices.to_csv('equities_close_prices_minute.csv')

print("Close prices saved to 'equities_close_prices_minute.csv'")
