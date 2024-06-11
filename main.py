from src.portfolio import Portfolio
from src.data_reader import DataReader


def main():
    """The main function of the trading bot."""
    # Initialize the DataReader class and get the stock data 
    # IMPORTANT: The start date should be the first trading day of the year 2023 since
    # beta is calculated using daily returns the last past years.
    data_reader = DataReader("2000-01-01")
    data_reader.run()

    # Hyperparameters
    streakLength = 5
    maxStock = 20
   
    portfolio = Portfolio(streakLength=streakLength, maxStock=maxStock)
    portfolio.calculate_performance_all()
    result = portfolio.trade(4)

    print(result)

    try:
        print("#============================" "=============================#\n")

        print("The trading bot has finished running.")
    except KeyboardInterrupt:
        print("\nThe trading bot has been interrupted.")


if __name__ == "__main__":
    main()
