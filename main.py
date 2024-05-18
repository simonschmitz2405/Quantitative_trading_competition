from src.portfolio import Portfolio
from src.data_reader import DataReader
import pandas as pd



def main():
    """The main function of the trading bot."""
    # Initialize the DataReader class and get the stock data
    data_reader = DataReader("2024-01-01")
    data_reader.run()
    portfolio = Portfolio(streakLength=5, thresholdType="returnRaw")
    portfolio.visualize_portfolio()

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\n\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
