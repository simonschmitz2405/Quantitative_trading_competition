import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BusinessDay
import matplotlib.pyplot as plt
import numpy as np


class Portfolio:
    def __init__(self, streakLength=4, thresholdType="returnRaw") -> None:
        """Initialize the Portfolio class."""
        # Read the CSV file
        self.stockData, self.marketInterest = self._read_csv()

        # Initialize the hyperparameters
        self.streakLength = streakLength
        self.thresholdingType = thresholdType
        self.holdingPeriod = None # TODO: Implement
        self.weighting = None # TODO: Implement
        self.longShort = None # TODO: Implement

        # Initialize the dataFrames
        self.returns = None
        self.threshCapm = None

        # Run the trading strategy
        self.dailyReturnsLong, self.dailyReturnsShort = self._run()
        self._calculate_annualized_return()

    def _run(self) -> None:
        """Run the trading strategy."""
        self.returns, self.threshCapm = self._prepare_data()
        # formation_date = pd.Timestamp("2024-05-15")
        # self.los, self.win = self.calculate_loser_winner_streaks(formation_date, self.streak_length)
        # print("Output", self.los, self.win)

        dailyReturnsLong, dailyReturnsShort = self._calculate_long_short_portfolio()
        print("Returns long", dailyReturnsLong)
        print("Returns short", dailyReturnsShort)
        return dailyReturnsLong, dailyReturnsShort

    def _choose_thresholding_type(self, thresholdType) -> int:
        """Choose the thresholding type."""
        if thresholdType == "CAPM":
            return "Streak_CAPM"
        elif thresholdType == "marketExcessReturn":
            return "Streak_Market_Return"
        elif thresholdType == "returnRaw":
            return "Streak_Raw_Return"
        else:
            print(
                "Invalid thresholding type. Please choose one of the following: CAPM, marketExcessReturn, returnRaw"
            )

    def _read_csv(self) -> pd.DataFrame:
        """Read the CSV file."""
        try:
            stockData = pd.read_csv(
                r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv"
            )
            marketInterest = pd.read_csv(
                r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\market_interest.csv"
            )
        except Exception as e:
            print(f"Error loading data: {e}")
        return stockData, marketInterest

    def _prepare_data(self):
        """Prepare the data for the trading strategy."""
        self.stockData["DATE"] = pd.to_datetime(self.stockData["DATE"])
        prices = self.stockData.pivot(index="DATE", columns="TICKER", values="CLOSE")
        returns = prices.pct_change()
        threshCapm = pd.DataFrame(index=prices.index, columns=prices.columns)

        return returns, threshCapm

    def _get_previous_business_day(self, dt) -> dt.datetime:
        while True:
            if dt.weekday() < 5 and dt not in self.returns.index:
                dt -= BusinessDay(1)
            else:
                return dt

    # def _get_previous_business_day(self, date) -> dt.datetime:
    #     """Get the previous business day."""
    #     while date not in self.returns.index:
    #         date -= BusinessDay(1)
    #     return date

    def _get_previous_returns(self, formation, streak_length=5) -> pd.DataFrame:
        """Get the previous returns for the given streak length."""
        previous_returns = {}
        for i in range(1, self.streakLength + 1):
            previous_day = formation - pd.offsets.BusinessDay(i)
            previous_returns[f"ret_{i}"] = self.returns.loc[
                self._get_previous_business_day(previous_day)
            ]
        return pd.DataFrame(previous_returns)

    def _calculate_streak_raw_return(self, x) -> int:
        """Calculate if there is a streak raw return."""
        return 1 if (x > 0).all() or (x < 0).all() else 0

    def _calculate_loser_winner_streaks(self, formation, streak_length=5) -> tuple:
        """Calculate the loser and winner streaks for the given formation date."""
        returnStreak = pd.DataFrame(self._get_previous_returns(formation, streak_length))
        returnStreak["Streak_Raw_Return"] = returnStreak.apply(
            lambda x: self._calculate_streak_raw_return(x), axis=1
        ) * returnStreak.mean(axis=1)
        returnStreak["Streak_Market_Return"] = 0  # TODO: implement
        returnStreak["Streak_CAPM"] = 0  # TODO: implement
        thresholdType = self._choose_thresholding_type(self.thresholdingType) 
        losersRawReturn = returnStreak[returnStreak[thresholdType] < 0].index
        print("Amount of Stock for this day that are losers", len(losersRawReturn))
        winnersRawReturn = returnStreak[returnStreak[thresholdType] > 0].index

        # TODO: Calculate the weighted returns for the losers and winners
        # Calculate the average raw return for losers and winners (Equal weighted)
        losRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(losersRawReturn)
        ].mean()
        winRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(winnersRawReturn)
        ].mean() * (-1)

        # return losret, winret
        return losRetRawReturn, winRetRawReturn

    def _calculate_long_short_portfolio(self, streakLength=5) -> tuple:
        """Calculate the daily returns for given streak length."""
        returnsLong = []
        returnsShort = []
        dates = []

        # Go through each day but starting from the 6th day since we need 5 days to calculate the streak
        for date in self.returns.index[8:]:
            returnsLong.append(
                self._calculate_loser_winner_streaks(pd.Timestamp(date), streakLength)[0]
            )
            returnsShort.append(
                self._calculate_loser_winner_streaks(pd.Timestamp(date), streakLength)[1]
            )
            dates.append(date)
        return returnsLong, returnsShort

    def _plot_data(self) -> None:
        """Plot the daily returns for the long and short portfolios."""
        print(self.returns)
        print(self.threshCapm)

    def _calculate_annualized_return(self) -> None:
        """Calculate the annualized return."""
        cleanReturnsLong = [x for x in self.dailyReturnsLong if str(x) != "nan"]
        cleanReturnsShort = [x for x in self.dailyReturnsShort if str(x) != "nan"]

        dailyReturnLong = np.array(cleanReturnsLong).mean()
        dailyReturnShort = np.array(cleanReturnsShort).mean()

        print("Average daily return long portfolio: ", dailyReturnLong)
        print("Average yearyly return long portfolio: ", (1+ dailyReturnLong)**365 -1)

    def visualize_portfolio(self):
        """Visualize the portfolio."""
        cleanReturnsLong = [x for x in self.dailyReturnsLong if str(x) != "nan"]

        initialBudget = 1
        budgetLong = [initialBudget]
        budgetMarket = [initialBudget]
        budgetShort = [initialBudget]
        for i in range(1, len(cleanReturnsLong)):
            budgetLong.append(budgetLong[i - 1] * (1 + cleanReturnsLong[i]))
            budgetShort.append(budgetShort[i - 1] * (1 + self.dailyReturnsShort[i]))
            budgetMarket.append(budgetMarket[i - 1] * (1 + self.marketInterest["SP500_Returns"][i]))
        plt.plot(budgetLong, label="Long Portfolio")
        plt.plot(budgetMarket, label="Market Portfolio")
        plt.plot(budgetShort, label="Short Portfolio")
        plt.legend()
        plt.show()


# trading = TradingBot()
