from src.portfolio import Portfolio
from src.data_reader import DataReader


def main():
    """The main function of the trading bot."""
    # Initialize the DataReader class and get the stock data
    # data_reader = DataReader("2024-01-01")
    # data_reader.run()

    # Hyperparameters
    streakLength = 5
    maxStock = 10
   
    portfolio = Portfolio(streakLength=streakLength, maxStock=maxStock)
    portfolio.calculate_performance_all()

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
