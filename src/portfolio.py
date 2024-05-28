import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BusinessDay
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import StringIO
import logging


class Portfolio:
    def __init__(self, streakLength=4, thresholdType="returnRaw") -> None:
        """Initialize the Portfolio class."""
        # Define the logging
        logging.basicConfig(level=logging.INFO, filename=r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\src\logfile.log", filemode="w")

        # Define the paths
        self._pathSp500StockData = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv"
        self._pathMarketReturns = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\market_interest.csv"
        self._pathFfWeekly = r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\ff_weekly.csv"

        # Read the CSV file
        self.stockData, self.marketReturns, self.ff_weekly = self._read_csv()

        # Initialize the hyperparameters
        self.streakLength = streakLength
        self.thresholdingType = thresholdType
        self.holdingPeriod = None  # TODO: Implement
        self.weighting = None  # TODO: Implement
        self.longShort = None  # TODO: Implement

        # Initialize the dataFrames which are used later
        # self.returns = None

        # Run the trading strategy
        self.dailyReturnsLong, self.dailyReturnsShort, self.datesLong, self.datesShort = self._run()

        # Calculate the perfomrance of the portfolio
        self._calculate_annualized_return()
        self._performance_portfolio()

    def _run(self) -> None:
        """Run the trading strategy."""
        # Print the parameters of the trading strategy
        self._print_parameter()
        # Prepare the data for the trading strategy
        self.returns = self._prepare_data()
        # Calculate the daily returns for the long and short portfolios
        dailyReturnsLong, dailyReturnsShort, datesLong, datesShort = (
            self._calculate_long_short_portfolio()
        )
        return dailyReturnsLong, dailyReturnsShort, datesLong, datesShort

    def _print_parameter(self) -> None:
        """Print the parameters of the trading strategy."""
        print("#============================" "=============================#\n")
        print("Parameters of the trading strategy:\n")
        print(f"Streak length: {self.streakLength}")
        print(f"Thresholding type: {self.thresholdingType}")
        print(f"Holding period: {self.holdingPeriod}")
        print(f"Weighting: {self.weighting}")
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

    def _get_previous_business_day(self, date) -> dt.datetime:
        """Get the previous business day."""
        while date not in self.returns.index:
            date -= BusinessDay(1)
        return date

    def _get_previous_returns(self, formation, streak_length=5) -> pd.DataFrame:
        """Get the previous returns for the given streak length."""
        previous_returns = {}
        for i in range(1, self.streakLength + 1):
            previous_day = formation - pd.offsets.BusinessDay(i)
            previous_returns[f"ret_{i}"] = self.returns.loc[
                self._get_previous_business_day(previous_day)
            ]
        return pd.DataFrame(previous_returns)

    def _get_previous_market_excess_returns(self, formation, streak_length=5) -> pd.DataFrame:
        """Get the previous market excess returns for the given streak length."""
        previous_market_excess_returns = {}
        for i in range(1, self.streakLength + 1):
            previous_day = formation - pd.offsets.BusinessDay(i)
            previous_market_excess_returns[f"ret_market_excess{i}"] = (
                self.returns.loc[self._get_previous_business_day(previous_day)]
                - self.marketReturns.loc[
                    self._get_previous_business_day(previous_day), "SP500_Returns"
                ]
            )
        return pd.DataFrame(previous_market_excess_returns)

    def _calculate_streak(self, x) -> int:
        """Calculate if there is a streak raw return."""
        return 1 if (x > 0).all() or (x < 0).all() else 0

    def _calculate_loser_winner_streaks(self, formation, streak_length=5) -> tuple:
        """Calculate the loser and winner streaks for the given formation date."""
        returnStreakRaw = pd.DataFrame(self._get_previous_returns(formation, streak_length))
        returnStreakMarket = pd.DataFrame(
            self._get_previous_market_excess_returns(formation, streak_length)
        )
        streaks = pd.DataFrame(index=returnStreakRaw.index)
        streaks["Streak_Raw_Return"] = returnStreakRaw.apply(
            lambda x: self._calculate_streak(x), axis=1
        ) * returnStreakRaw.mean(axis=1)
        streaks["Streak_Market_Return"] = returnStreakMarket.apply(
            lambda x: self._calculate_streak(x), axis=1
        ) * returnStreakMarket.mean(axis=1)
        thresholdType = self._choose_thresholding_type(self.thresholdingType)
        losersRawReturn = streaks[streaks[thresholdType] < 0].index
        winnersRawReturn = streaks[streaks[thresholdType] > 0].index
        # print("Amount of Stock for this day that are losers", len(losersRawReturn))

        # TODO: Calculate the weighted returns for the losers and winners
        # Calculate the average raw return for losers and winners (Equal weighted)
        losRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(losersRawReturn)
        ].mean()
        winRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(winnersRawReturn)
        ].mean()

        # return losret, winret
        return losRetRawReturn, winRetRawReturn

    def _calculate_long_short_portfolio(self, streakLength=5) -> tuple:
        """Calculate the daily returns for given streak length."""
        returnsLong = []
        returnsShort = []
        datesLong = []
        datesShort = []

        for date in self.returns.index[8:]:
            long_return, short_return = self._calculate_loser_winner_streaks(
                pd.Timestamp(date), streakLength
            )

            # Füge die Long-Rendite und das Datum hinzu, wenn ein Long-Rendite-Wert berechnet wurde
            if pd.notna(long_return):
                returnsLong.append(long_return)
                datesLong.append(pd.Timestamp(date).date())

            # Füge die Short-Rendite und das Datum hinzu, wenn ein Short-Rendite-Wert berechnet wurde
            if pd.notna(short_return):
                returnsShort.append(short_return)
                datesShort.append(pd.Timestamp(date).date())

        return returnsLong, returnsShort, datesLong, datesShort

    def _plot_data(self) -> None:
        """Plot the daily returns for the long and short portfolios."""
        logging.info('Daily Returns Long', self.dailyReturnsLong)
        logging.info('Daily Returns Short', self.dailyReturnsShort)

    def _calculate_annualized_return(self) -> None:
        """Calculate the annualized return."""
        cleanReturnsLong = [x for x in self.dailyReturnsLong if str(x) != "nan"]
        cleanReturnsShort = [x for x in self.dailyReturnsShort if str(x) != "nan"]

        longData = pd.DataFrame({"Date": self.datesLong, "Return": self.dailyReturnsLong})

        shortData = pd.DataFrame({"Date": self.datesShort, "Return": self.dailyReturnsShort})
        longData["Date"] = pd.to_datetime(longData["Date"])
        longData.set_index("Date", inplace=True)

        shortData["Date"] = pd.to_datetime(shortData["Date"])
        shortData.set_index("Date", inplace=True)

        dailyReturnLong = np.array(cleanReturnsLong).mean()
        dailyReturnShort = np.array(cleanReturnsShort).mean()

        monthly_Return_Long = longData["Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_Return_Short = shortData["Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        annual_Return_Long = longData["Return"].resample("Y").apply(lambda x: (1 + x).prod() - 1)
        annual_Return_Short = shortData["Return"].resample("Y").apply(lambda x: (1 + x).prod() - 1)

        aver_Monthly_Return_Long = monthly_Return_Long.mean()
        aver_Monthly_Return_Short = monthly_Return_Short.mean()
        global aver_annual_Return_Long
        aver_annual_Return_Long = annual_Return_Long.mean()
        aver_annual_Return_Short = annual_Return_Short.mean()

        total_Return_long = np.prod([(1 + r) for r in cleanReturnsLong]) - 1

        print("Average daily long return ", dailyReturnLong)
        print("Average monthly long return ", aver_Monthly_Return_Long)
        print("Average annual long return ", aver_annual_Return_Long)
        print("Total long return ", total_Return_long)

    def _performance_portfolio(self) -> None:
        """Calculate the performance of the portfolio."""
        print("#============================" "=============================#\n")
        print("Performance of the portfolio:\n")
        ### Volatility ###

        returnLong_variance = np.array(self.dailyReturnsLong).var()
        returnLong_volatility = np.sqrt(returnLong_variance)
        print("Varianz des long Portfolio:", returnLong_variance)
        print("Volatilität des long Portfolio:", returnLong_volatility)

        ###Risk Free###

        self.ff_weekly["Date"] = pd.to_datetime(self.ff_weekly["Date"])
        self.ff_weekly.set_index("Date", inplace=True)
        riskFree_Monthly_Rate_ = (
            self.ff_weekly["RF"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        )
        riskFree_annual_Rate_ = (
            self.ff_weekly["RF"].resample("Y").apply(lambda x: (1 + x).prod() - 1)
        )

        aver_riskFree_Rate = riskFree_annual_Rate_.mean()

        ###Sharp Ratio###

        sharp_Ratio = (aver_annual_Return_Long - aver_riskFree_Rate) / returnLong_volatility
        print("Sharp Ratio: ", sharp_Ratio)

    def visualize_portfolio(self) -> None:
        """Visualize the portfolio."""
        cleanReturnsLong = [x for x in self.dailyReturnsLong if str(x) != "nan"]
        cleanReturnsShort = [x for x in self.dailyReturnsShort if str(x) != "nan"]

        initialBudget = 1
        budgetLong = [initialBudget]
        budgetMarket = [initialBudget]
        budgetShort = [initialBudget]

        min_length = min(
            len(cleanReturnsLong), len(cleanReturnsShort), len(self.marketReturns["SP500_Returns"])
        )

        for i in range(1, min_length):
            budgetLong.append(budgetLong[i - 1] * (1 + cleanReturnsLong[i]))
            budgetShort.append(budgetShort[i - 1] * (1 + self.dailyReturnsShort[i]))
            budgetMarket.append(budgetMarket[i - 1] * (1 + self.marketReturns["SP500_Returns"][i]))
        plt.plot(budgetLong, label="Long Portfolio")
        plt.plot(budgetMarket, label="Market Portfolio")
        plt.plot(budgetShort, label="Short Portfolio")
        plt.legend()
        plt.show()
