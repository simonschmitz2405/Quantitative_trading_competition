from trading_bot import TradingBot
from data_reader import SP500StockData
import pandas as pd


def main():
    """The main function of the trading bot."""
    sp500_stock_data = SP500StockData(start_date="2024-01-01")
    print("S&P 500 Ticker Symbols:")
    print(sp500_stock_data.sp500_symbols)
    stock_data = sp500_stock_data.fetch_stock_data()
    stock_data.to_csv("sp500_stock_data.csv", index=False)
    print("S&P 500 Stock Data: ")
    
    #print(stock_data.to_csv(r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv", index=False))

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\n\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
