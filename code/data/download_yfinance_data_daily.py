# import relevant libraries
import yfinance as yf
import pandas as pd


# define function to download equity prices from yfinance
def download_close_prices(tickers, start_date, end_date):
    equity_data = {}
    for ticker in tickers:
        try:
            # download equity data
            data = yf.download(ticker, start=start_date, end=end_date)
            # extract close prices
            close_prices = data['Close']
            # store close prices in the dictionary
            equity_data[ticker] = close_prices
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    return equity_data


# list of tickers
# tickers = ["BA", "CAT", "CVX", "DIS", "GE", "HON", "IBM", "JNJ", "KO", "MMM",
#            "MRK", "PG", "XOM", "MCD", "AXP", "DD", "PFE", "WMT", "TRV", "INTC",
#            "JPM", "HP", "NKE", "AAPL", "HD", "AMGN", "VZ", "UNH", "MSFT", "CSCO"]

tickers = ["AAPL", "JPM", "WMT"]

# define start and end dates
start_date = "2018-01-02"
end_date = "2023-12-30"

# download close prices
equity_close_prices = download_close_prices(tickers, start_date, end_date)

# convert the dictionary into a DataFrame
df = pd.DataFrame(equity_close_prices)

# save the DataFrame to a CSV file
df.to_csv('equities_close_prices_daily.csv')

print("Close prices saved to 'equities_close_prices_daily.csv'")
