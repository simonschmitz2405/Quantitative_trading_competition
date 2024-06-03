import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy import stats
import statsmodels.api as sm

class Portfolio:
    def __init__(self, streakLength, thresholdType, valueWeighted, maxStocks) -> None:
        """Initialize the Portfolio class."""
        # Define the logging
        logging.basicConfig(
            level=logging.INFO,
            filename=r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\src\logfile.log",
            filemode="w",
        )

        # Define the paths
        self._pathSp500StockData = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv"
        self._pathMarketReturns = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\market_interest.csv"
        self._pathFfWeekly = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\ff_daily.csv"

        # Read the CSV file
        self.stockData, self.marketReturns, self.ff_daily = self._read_csv()

        # Initialize the hyperparameters
        self.streakLength = streakLength
        self.thresholdingType = thresholdType
        self.valueWeighted = valueWeighted
        self.maxStocks = maxStocks
        self.longShort = None  # TODO: Implement

        # Initialize the dataFrames which are used later
        # self.returns = None

        # Run the trading strategy
        self.dailyReturnsLong, self.dailyReturnsShort = self._run()

        # Calculate the list of the risk-free rate
        self.riskFreeRate, self.marketReturns = self._calculate_riskfree_market_rate()


        # Calculate the perfomrance of the portfolio
        self._calculate_performance()
        # self._calculate_annualized_return()
        # self._performance_portfolio()

    def _run(self) -> None:
        """Run the trading strategy."""
        # Print the parameters of the trading strategy
        self._print_parameter()

        # Prepare the data for the trading strategy
        self.returns = self._prepare_data()

        # Calculate the daily returns for the long and short portfolios
        dailyReturnsLong, dailyReturnsShort = self._calculate_long_short_portfolio()

        return dailyReturnsLong, dailyReturnsShort

    def _print_parameter(self) -> None:
        """Print the parameters of the trading strategy."""
        print("#============================" "=============================#\n")
        print("PORTFOLIO\n")
        print("Parameters of the trading strategy:")
        print("#---------------------------")
        print(f"Streak length: {self.streakLength}")
        print(f"Thresholding type: {self.thresholdingType}")
        print(f"Weighting: {self.valueWeighted}")
        print(f"Max Stocks: {self.maxStocks}")
        print(f"Long/Short: {self.longShort} \n")

    def _choose_thresholding_type(self, thresholdType) -> int:
        """Choose the thresholding type."""
        if thresholdType == "marketExcessReturn":
            return "Streak_Market_Return"
        elif thresholdType == "returnRaw":
            return "Streak_Raw_Return"
        else:
            print(
                "Invalid thresholding type. Please choose one of the following: marketExcessReturn, returnRaw"
            )

    def _read_csv(self) -> pd.DataFrame:
        """Read the CSV file."""
        try:
            stockData = pd.read_csv(self._pathSp500StockData)
            marketReturns = pd.read_csv(self._pathMarketReturns)
            ff_weekly = pd.read_csv(self._pathFfWeekly)
        except Exception as e:
            print(f"Error loading data: {e}")
        return stockData, marketReturns, ff_weekly

    def _prepare_data(self):
        """Prepare the data for the trading strategy."""
        self.stockData["DATE"] = pd.to_datetime(self.stockData["DATE"])
        prices = self.stockData.pivot(index="DATE", columns="TICKER", values="CLOSE")
        returns = prices.pct_change()
        self.marketReturns.set_index("DATE", inplace=True)
        self.marketReturns.index = pd.to_datetime(self.marketReturns.index)
        return returns

    def _get_previous_returns(self, formation) -> pd.DataFrame:
        """Get the previous returns for the given streak length."""
        previous_returns = {}
        for i in range(1, self.streakLength + 1):
            dates_list = self.stockData["DATE"].tolist()
            index = dates_list.index(formation)

            previous_returns[f"ret_{i}"] = self.returns.loc[dates_list[index - i]]
        return pd.DataFrame(previous_returns)

    def _get_previous_market_excess_returns(self, formation) -> pd.DataFrame:
        """Get the previous market excess returns for the given streak length."""
        previous_market_excess_returns = {}
        for i in range(1, self.streakLength + 1):
            dates_list = self.stockData["DATE"].tolist()
            index = dates_list.index(formation)
            previous_market_excess_returns[f"ret_market_excess{i}"] = (
                self.returns.loc[dates_list[index - i]]
                - self.marketReturns.loc[
                    dates_list[index - i], "SP500_Returns"
                ]
            )
        return pd.DataFrame(previous_market_excess_returns)


    def _calculate_streak(self, x) -> int:
        """Calculate if there is a streak raw return."""
        return 1 if (x > 0).all() or (x < 0).all() else 0
    
    def _calculate_loser_winner_streaks(self, returnDay) -> tuple:
        """Calculate the loser and winner streaks for the given formation date."""
        returnStreakRaw = pd.DataFrame(self._get_previous_returns(returnDay))
        returnStreakMarket = pd.DataFrame(self._get_previous_market_excess_returns(returnDay))
        streaks = pd.DataFrame(index=returnStreakRaw.index)
        streaks["Streak_Raw_Return"] = returnStreakRaw.apply(
            lambda x: self._calculate_streak(x), axis=1
        ) * returnStreakRaw.mean(axis=1)
        streaks["Streak_Market_Return"] = returnStreakMarket.apply(
            lambda x: self._calculate_streak(x), axis=1
        ) * returnStreakMarket.mean(axis=1)
        thresholdType = self._choose_thresholding_type(self.thresholdingType)
        losersRawReturn = streaks[streaks[thresholdType] < 0 ].nsmallest(self.maxStocks, thresholdType).index
        winnersRawReturn = streaks[streaks[thresholdType] > 0].nlargest(self.maxStocks, thresholdType).index
        # print("Amount of Stock for this day that are losers", len(losersRawReturn))
        # print("Amount of Stock for this day that are winners", len(winnersRawReturn))
        # print('Date', returnDay)
        # print("LoserIndex", losersRawReturn)
        losRetRawReturn = None
        winRetRawReturn = None

        # Check if the value weighted return should be calculated
        if self.valueWeighted is True:
            weightsLoser = []
            weightsWinner = []

            # Calculate the Returns for the winners and losers
            if not losersRawReturn.tolist():
                losRetRawReturn = 0
            else:
                # Get the returns for the losers and winners
                losRetRawReturn = self.returns.loc[
                    returnDay, self.returns.columns.isin(losersRawReturn)
                ]
            if not winnersRawReturn.tolist():
                winRetRawReturn = 0
            else:
                winRetRawReturn = self.returns.loc[
                    returnDay, self.returns.columns.isin(winnersRawReturn)
                ]

            # Calculate the weights for the winners and losers
            for ticker in losersRawReturn.tolist():
                weightsLoser.append(
                    self.stockData.loc[
                        (self.stockData["TICKER"] == ticker)
                        & (self.stockData["DATE"] == returnDay),
                        "MARKET_CAP",
                    ].values[0]
                )

            for ticker in winnersRawReturn.tolist():
                weightsWinner.append(
                    self.stockData.loc[
                        (self.stockData["TICKER"] == ticker)
                        & (self.stockData["DATE"] == returnDay),
                        "MARKET_CAP",
                    ].values[0]
                )

            # Calculate the relative weights
            weightsLoser = weightsLoser / np.sum(weightsLoser)
            weightsWinner = weightsWinner / np.sum(weightsWinner)

            if not losersRawReturn.tolist():
                weightsLoser = 0

            if not winnersRawReturn.tolist():
                weightsWinner = 0

            # Calculate the value weighted return
            losRetRawReturn = np.dot(losRetRawReturn, weightsLoser)
            winRetRawReturn = np.dot(winRetRawReturn, weightsWinner)

        else:
            if not losersRawReturn.tolist():
                losRetRawReturn = 0
            else:
                # Get the returns for the losers and winners
                losRetRawReturn = self.returns.loc[
                    returnDay, self.returns.columns.isin(losersRawReturn)
                ].mean()
            if not winnersRawReturn.tolist():
                winRetRawReturn = 0
            else:
                winRetRawReturn = self.returns.loc[
                    returnDay, self.returns.columns.isin(winnersRawReturn)
                ].mean()

        return losRetRawReturn, winRetRawReturn

    def _calculate_long_short_portfolio(self) -> tuple:
        """Calculate the daily returns for given streak length."""
        correction = len(self.marketReturns.index) - len(self.ff_daily.index)
        result = np.zeros((3, len(self.returns.index[8:-correction])))
        for i, date in enumerate(self.returns.index[8:-correction]):
            print(date)
            result[0, i], result[1, i]= self._calculate_loser_winner_streaks(pd.Timestamp(date))
        
        logging.info("Daily Returns Long: %s", result[0, :])
        logging.info("Daily Returns Short: %s", result[1, :])

        return result[0, :], result[1, :]

    def _calculate_riskfree_market_rate(self) -> float:
        """Calculate the risk-free rate."""
        correction = len(self.marketReturns.index) - len(self.ff_daily.index)
        return self.ff_daily["RF"].tolist()[8:], self.marketReturns["SP500_Returns"].tolist()[8:-correction]

    def _get_t_stats(self, vector, lags=1) -> tuple:
        """Calculate the t-stats for the long and short portfolios with Newey-West standard errors."""
        # Add a constant to the vector for OLS regression
        X = np.ones(len(vector))
        model = sm.OLS(vector, X)

        # Fit the model
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        # Get the t-statistic and p-value
        t = results.tvalues[0]
        p = results.pvalues[0]

        return t, p


    def _plot_data(self) -> None:
        """Plot the daily returns for the long and short portfolios."""
        logging.info("Daily Returns Long", self.dailyReturnsLong)
        logging.info("Daily Returns Short", self.dailyReturnsShort)


    def _calculate_performance(self) -> None:
        """Calculate the performance of the portfolio."""
        # Daily average returns
        dailyReturnLong = np.array(self.dailyReturnsLong).mean()
        dailyReturnShort = np.array(self.dailyReturnsShort).mean()
        dailyReturnRiskFree = np.array(self.riskFreeRate).mean()
        dailyReturnMarket = np.array(self.marketReturns).mean()
        # Daily standard deviation
        dailystandarddeviationLong = np.array(self.dailyReturnsLong).std()
        dailystandarddeviationShort = np.array(self.dailyReturnsShort).std()
        dailystandarddeviationRiskFree = np.array(self.riskFreeRate).std()
        dailystandarddeviationMarket = np.array(self.marketReturns).std()
        # Annualized average returns
        annualizedReturnLong = dailyReturnLong * 252
        annualizedReturnShort = dailyReturnShort * 252
        annualizedReturnRiskFree = dailyReturnRiskFree * 252
        annualizedReturnMarket = dailyReturnMarket * 252
        # Annualized standard deviation
        annualizedStandardDeviationLong = dailystandarddeviationLong * np.sqrt(252)
        annualizedStandardDeviationShort = dailystandarddeviationShort * np.sqrt(252)
        annualizedStandardDeviationRiskFree = dailystandarddeviationRiskFree * np.sqrt(252)
        annualizedStandardDeviationMarket = dailystandarddeviationMarket * np.sqrt(252)
        # Sharpe Ratio annualized
        sharpeRatioLong = (annualizedReturnLong - annualizedReturnRiskFree) / annualizedStandardDeviationLong
        sharpeRatioShort = (annualizedReturnShort - annualizedReturnRiskFree) / annualizedStandardDeviationShort

        dailyexcessriskfreelong = np.array(self.dailyReturnsLong) - np.array(self.riskFreeRate)
        dailyexcessriskfreeshort = np.array(self.dailyReturnsShort) - np.array(self.riskFreeRate)
        dailyexcessmarketlong = np.array(self.dailyReturnsLong) - np.array(self.marketReturns)
        dailyexcessmarketshort = np.array(self.dailyReturnsShort) - np.array(self.marketReturns)

        tstatsexcessriskfreelong = self._get_t_stats(dailyexcessriskfreelong)
        tstatsexcessriskfreeshort = self._get_t_stats(dailyexcessriskfreeshort)
        tstatsexcessmarketlong = self._get_t_stats(dailyexcessmarketlong)
        tstatsexcessmarketshort = self._get_t_stats(dailyexcessmarketshort)
    


        print("Performance of the portfolio:")
        print("#---------------------------")
        print("Daily average returns long: ", dailyReturnLong)
        print("Daily average returns short: ", dailyReturnShort)
        print("Daily average risk free rate: ", dailyReturnRiskFree)
        print("Daily average market return: ", dailyReturnMarket)
        print("#---------------------------")
        print("Daily standard deviation long: ", dailystandarddeviationLong)
        print("Daily standard deviation short: ", dailystandarddeviationShort)
        print("Daily standard deviation risk free rate: ", dailystandarddeviationRiskFree)
        print("Daily standard deviation market return: ", dailystandarddeviationMarket)
        print("#---------------------------")
        print("Annualized average returns long: ", annualizedReturnLong)
        print("Annualized average returns short: ", annualizedReturnShort)
        print("Annualized average risk free rate: ", annualizedReturnRiskFree)
        print("Annualized average market return: ", annualizedReturnMarket)
        print("#---------------------------")
        print("Annualized standard deviation long: ", annualizedStandardDeviationLong)
        print("Annualized standard deviation short: ", annualizedStandardDeviationShort)
        print("Annualized standard deviation risk free rate: ", annualizedStandardDeviationRiskFree)
        print("Annualized standard deviation market return: ", annualizedStandardDeviationMarket)
        print("#---------------------------")
        print("Excess return risk free annualized long :", annualizedReturnLong - annualizedReturnRiskFree)
        print("Excess return risk free annualized short :", annualizedReturnShort - annualizedReturnRiskFree)
        print("Excess return market annualized long :", annualizedReturnLong - annualizedReturnMarket)
        print("Excess return market annualized short :", annualizedReturnShort - annualizedReturnMarket)
        print("#---------------------------")
        print("Sharpe Ratio long: ", sharpeRatioLong)
        print("Sharpe Ratio short: ", sharpeRatioShort)
        print("#---------------------------")
        print("T-Stats excess risk free long: ", tstatsexcessriskfreelong)
        print("T-Stats excess risk free short: ", tstatsexcessriskfreeshort)
        print("T-Stats excess market long: ", tstatsexcessmarketlong)
        print("T-Stats excess market short: ", tstatsexcessmarketshort)

        logging.info("Daily average returns long: %s", dailyReturnLong)
        logging.info("Daily average returns short: %s", dailyReturnShort)
        logging.info("Daily average risk free rate: %s", dailyReturnRiskFree)
        logging.info("Daily average market return: %s", dailyReturnMarket)
        logging.info("Daily standard deviation long: %s", dailystandarddeviationLong)
        logging.info("Daily standard deviation short: %s", dailystandarddeviationShort)
        logging.info("Daily standard deviation risk free rate: %s", dailystandarddeviationRiskFree)
        logging.info("Daily standard deviation market return: %s", dailystandarddeviationMarket)
        logging.info("Annualized average returns long: %s", annualizedReturnLong)
        logging.info("Annualized average returns short: %s", annualizedReturnShort)
        logging.info("Annualized average risk free rate: %s", annualizedReturnRiskFree)
        logging.info("Annualized average market return: %s", annualizedReturnMarket)
        logging.info("Annualized standard deviation long: %s", annualizedStandardDeviationLong)
        logging.info("Annualized standard deviation short: %s", annualizedStandardDeviationShort)
        logging.info("Annualized standard deviation risk free rate: %s", annualizedStandardDeviationRiskFree)
        logging.info("Annualized standard deviation market return: %s", annualizedStandardDeviationMarket)
        logging.info("Excess return risk free annualized long :%s", annualizedReturnLong - annualizedReturnRiskFree)
        logging.info("Excess return risk free annualized short :%s", annualizedReturnShort - annualizedReturnRiskFree)
        logging.info("Excess return market annualized long :%s", annualizedReturnLong - annualizedReturnMarket)
        logging.info("Excess return market annualized short :%s", annualizedReturnShort - annualizedReturnMarket)
        logging.info("Sharpe Ratio long: %s", sharpeRatioLong)
        logging.info("Sharpe Ratio short: %s", sharpeRatioShort)
        logging.info("T-Stats excess risk free long: %s", tstatsexcessriskfreelong)
        logging.info("T-Stats excess risk free short: %s", tstatsexcessriskfreeshort)
        logging.info("T-Stats excess market long: %s", tstatsexcessmarketlong)
        logging.info("T-Stats excess market short: %s", tstatsexcessmarketshort)


    def visualize_portfolio(self) -> None:
        """Visualize the portfolio."""
        # cleanReturnsLong = [x for x in self.dailyReturnsLong if str(x) != "nan"]
        # cleanReturnsShort = [x for x in self.dailyReturnsShort if str(x) != "nan"]

        initialBudget = 1
        budgetLong = [initialBudget]
        budgetMarket = [initialBudget]
        # budgetShort = [initialBudget]

        min_length = min(
            len(self.dailyReturnsLong),
            len(self.dailyReturnsShort),
            len(self.marketReturns["SP500_Returns"]),
        )

        for i in range(1, min_length):
            budgetLong.append(budgetLong[i - 1] * (1 + self.dailyReturnsLong[i]))
            # budgetShort.append(budgetShort[i - 1] * (1 + self.dailyReturnsShort[i]))
            budgetMarket.append(budgetMarket[i - 1] * (1 + self.marketReturns["SP500_Returns"][i]))
        plt.plot(budgetLong, label="Long Portfolio")
        plt.plot(budgetMarket, label="Market Portfolio")
        # plt.plot(budgetShort, label="Short Portfolio")
        plt.legend()
        plt.show()