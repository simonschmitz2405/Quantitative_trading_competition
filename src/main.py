from trading_bot import TradingBot
from data_reader import DataReader
import pandas as pd



def main():
    """The main function of the trading bot."""
    # Initialize the DataReader class and get the stock data
    data_reader = DataReader("2023-05-05")
    data_reader.run()
    trading = TradingBot()


    # Read the stock data from the CSV file
    # stock_data = pd.read_csv(r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv")
    # print(stock_data)

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\n\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
