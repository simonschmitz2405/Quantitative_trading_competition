import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

class SP500StockData:
    def __init__(self, start_date):
        self.start_date = start_date
        self.sp500_symbols = self.get_sp500_symbols()
    
    def get_sp500_symbols(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
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

    def get_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        hist = yf.download(ticker, start=self.start_date)
        hist_div = pd.DataFrame(stock.history(start=self.start_date)['Dividends'])
        hist_div.index = hist_div.index.strftime('%Y-%m-%d')
        hist.index = hist.index.strftime('%Y-%m-%d')
        hist = pd.merge(hist, hist_div, how='left', left_index=True, right_index=True)
        hist_dict = hist.to_dict()
        dates = []
        hist_output = {
            "DATE": [],
            "TICKER": [],
            "CLOSE": [],
            "VOLUME": [],
            "RETURN": [],
            "RET_EX_DIV": [],
            "SHARES_OUT": []
        }
        shrout = stock.get_shares_full(start=self.start_date)

        for i in hist_dict['Volume'].keys():
            dates.append(str(i)[:10])
        hist_output['DATE'] = dates

        shrout_dates = []
        for i in shrout.to_dict().keys():
            shrout_dates.append(str(i)[:10])

        for i in range(len(dates)):
            hist_output['TICKER'].append(ticker)
            hist_output['CLOSE'].append(hist['Close'][i])
            hist_output['VOLUME'].append(hist['Volume'][i])
            if i == 0:
                hist_output['RETURN'].append(None)
                hist_output['RET_EX_DIV'].append(None)
            else:
                hist_output['RETURN'].append((hist['Close'][i] - hist['Close'][i-1] + hist['Dividends'][i])/hist['Close'][i-1])
                hist_output['RET_EX_DIV'].append((hist['Close'][i] - hist['Close'][i-1])/hist['Close'][i-1])
            if(shrout_dates.__contains__(dates[i])):
                for j in range(len(shrout_dates)):
                    if(shrout_dates[j] == dates[i]):
                        hist_output['SHARES_OUT'].append(shrout[j])
            else:
                hist_output['SHARES_OUT'].append(None)

        hist_output = pd.DataFrame(hist_output).ffill()

        return hist_output
    
    def fetch_stock_data(self):
        stock_data = pd.DataFrame()
        for symbol in self.sp500_symbols:
            try:
                if '.' in symbol:
                    symbol = symbol.replace('.', '-')
                stock_data = pd.concat([stock_data, self.get_stock_data(symbol)], ignore_index=True)
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")
        return stock_data

