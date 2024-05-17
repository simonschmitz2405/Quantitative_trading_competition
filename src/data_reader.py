import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf


class DataReader:
    def __init__(self, start_date):
        self.start_date = start_date

    def run(self):
        self.stock_data = pd.DataFrame()
        self.sp500_symbols = self._get_sp500_symbols()
        # self.sp500_symbols = ["MMM"]
        self.stock_data = pd.DataFrame()
        self._fetch_stock_data(start_date=self.start_date)
        self._save_stock_data_to_csv()
        self._get_market_return_and_US_Treasury()

    def _get_sp500_symbols(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"id": "constituents"})
            symbols = []
            rows = table.find_all("tr")[1:]
            for row in rows:
                symbol = row.find_all("td")[0].text.strip()
                symbols.append(symbol)
            return symbols
        else:
            print("Failed to retrieve data from Wikipedia.")
            return None

    def _get_stock_data(self, ticker, start_date):
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = yf.download(ticker, start=start_date)
        hist_div = pd.DataFrame(stock.history(start=start_date)["Dividends"])
        hist_div.index = hist_div.index.strftime("%Y-%m-%d")
        hist.index = hist.index.strftime("%Y-%m-%d")
        hist = pd.merge(hist, hist_div, how="left", left_index=True, right_index=True)
        hist_dict = hist.to_dict()
        dates = []
        hist_output = {
            "DATE": [],
            "TICKER": [],
            "CLOSE": [],
            "VOLUME": [],
            "RETURN": [],
            "RET_EX_DIV": [],
            "SHARES_OUT": [],
            "BETA": [info["beta"]] * len(hist_dict["Volume"]),
            "MARKET_CAP": [info["marketCap"]] * len(hist_dict["Volume"]),
        }
        shrout = stock.get_shares_full(start=start_date)

        for i in hist_dict["Volume"].keys():
            dates.append(str(i)[:10])
        hist_output["DATE"] = dates

        shrout_dates = []
        for i in shrout.to_dict().keys():
            shrout_dates.append(str(i)[:10])

        for i in range(len(dates)):
            hist_output["TICKER"].append(ticker)
            hist_output["CLOSE"].append(hist["Close"][i])
            hist_output["VOLUME"].append(hist["Volume"][i])
            if i == 0:
                hist_output["RETURN"].append(None)
                hist_output["RET_EX_DIV"].append(None)
            else:
                hist_output["RETURN"].append(
                    (hist["Close"][i] - hist["Close"][i - 1] + hist["Dividends"][i])
                    / hist["Close"][i - 1]
                )
                hist_output["RET_EX_DIV"].append(
                    (hist["Close"][i] - hist["Close"][i - 1]) / hist["Close"][i - 1]
                )
            if shrout_dates.__contains__(dates[i]):
                for j in range(len(shrout_dates)):
                    if shrout_dates[j] == dates[i]:
                        hist_output["SHARES_OUT"].append(shrout[j])
            else:
                hist_output["SHARES_OUT"].append(None)

        hist_output = pd.DataFrame(hist_output).ffill()
        return hist_output

    def _fetch_stock_data(self, start_date="2024-05-05"):
        for symbol in self.sp500_symbols:
            try:
                if "." in symbol:
                    symbol = symbol.replace(".", "-")
                self.stock_data = pd.concat(
                    [self.stock_data, self._get_stock_data(symbol, start_date)], ignore_index=True
                )
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")

    def _save_stock_data_to_csv(
        self,
        filename=r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv",
    ):
        self.stock_data.to_csv(filename, index=False)


    def _get_market_return_and_US_Treasury(self):
        # Define the ticker symbols for S&P 500 and US Treasury interest rate
        sp500_ticker = "^GSPC"
        interest_rate = "^TNX"
        
        # Download historical data for S&P 500 and interest rate
        sp500_data = yf.download(sp500_ticker, start=self.start_date)
        interest_rate_data = yf.download(interest_rate, start=self.start_date)
        
        # Calculate daily returns for both datasets
        sp500_daily_returns = sp500_data["Adj Close"].pct_change()
        interest_rate_daily_returns = interest_rate_data["Adj Close"].pct_change()
        
        # Combine daily returns into a DataFrame
        market_interest = pd.DataFrame({
            'Date': sp500_data.index,  # Use the index of the S&P 500 data as the date column
            'SP500_Returns': sp500_daily_returns,
            'Interest_Rate_Returns': interest_rate_daily_returns
        })
        market_interest.set_index('Date')

        market_interest.to_csv(r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\market_interest.csv", index=False)

        return market_interest  



# data_reader = DataReader("2024-05-05")
