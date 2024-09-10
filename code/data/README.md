# Data

This folder contains datasets and scripts used for downloading and processing financial market data. These datasets include historical daily and minute-level closing prices for various equities and commodities exchange traderd funds (ETFs). The data is utilised in the Deep Reinforcement Learning (DRL) models for training and evaluation.

## Data Files
The data files in this folder contain historical daily closing prices for a range of asset tickers, from single equities to diversified collections of up to 30 tickers, covering the period from 2018 to 2023. These files include data for various combinations of equities such as technology (e.g., AAPL), retail (e.g., WMT), and finance (e.g., JPM), as well as commodities ETFs (USO, GLD, UNG). For instance, the file equities_daily_close_10_tickers_2018_2023.csv contains daily close prices for 10 selected equities across different industries, used for scaling experiments. These datasets are essential for training and evaluating the DQN and PPO models.  

## Data Download Scripts

- **`download_yfinance_data_daily.py`**: A Python script to download daily financial market data using the Yahoo Finance (yfinance) API. You can specify the stock tickers and date ranges to retrieve historical data.
  
- **`download_yfinance_data_minute.py`**: A Python script to download minute-level financial market data using the Yahoo Finance (yfinance) API. This script allows for high-frequency data collection over shorter periods.


## Usage

To download new financial data, use the provided Python scripts with the appropriate parameters for the tickers and date range. Example usage for downloading daily data:

```bash
python download_yfinance_data_daily.py
```

For minute-level data collection:  
```bash
python download_yfinance_data_minute.py
```
