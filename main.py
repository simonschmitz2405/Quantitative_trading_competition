from src.portfolio import Portfolio
from src.data_reader import DataReader
import pandas as pd
import numpy as np



def main():
    """The main function of the trading bot."""
    # Initialize the DataReader class and get the stock data
    # data_reader = DataReader("2024-01-01")
    # data_reader.run()

    streakLength = [5]
    thresholdType = ["returnRaw"]
    valueWeighted = [True]

    for countStreak, streak in enumerate(streakLength):
        for countThres, threshold in enumerate(thresholdType):
            for countValue, value in enumerate(valueWeighted):
                portfolio = Portfolio(streakLength=streak, thresholdType=threshold, valueWeighted=value)

    # portfolio1.visualize_portfolio()

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
