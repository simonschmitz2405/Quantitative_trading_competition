import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BusinessDay
import matplotlib.pyplot as plt
import numpy as np


class TradingBot:
    def __init__(self, streak_length=5) -> None:
        try:
            self.stock_data = pd.read_csv(
                r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\sp500_stock_data.csv"
            )
            self.market_interest = pd.read_csv(
                r"C:\Users\simon\OneDrive\Dokumente\[1] Uni\[1] Master\2. Semester Sommersemester 2024\Quantitative_trading_competition\Code\Quantitative_trading_competition\data\market_interest.csv"
            )
        except Exception as e:
            print(f"Error loading data: {e}")
        self.returns, self.Thresh_CAPM = self.prepare_data()
        self.streak_length = streak_length
        # formation_date = pd.Timestamp("2024-05-15")
        # self.los, self.win = self.calculate_loser_winner_streaks(formation_date, self.streak_length)
        # print("Output", self.los, self.win)

        self.returns_long, self.returns_short, self.dates = self.calculate_long_short_portfolio()
        print('Returns long',self.returns_long)
        print('Returns short',self.returns_short)
        print('Dates',self.dates)

        # self.plot_data()

    def prepare_data(self) -> tuple:
        tickers = self.stock_data["TICKER"].unique()
        prices = pd.DataFrame(index=self.stock_data["DATE"].unique(), columns=tickers)
        prices.index = pd.to_datetime(prices.index)
        returns = prices.pct_change()
        for ticker in tickers:
            for date in prices.index:
                prices.loc[date, ticker] = self.stock_data.loc[
                    (self.stock_data["TICKER"] == ticker) & (self.stock_data["DATE"] == date)
                ]["CLOSE"].values

        Thresh_CAPM = pd.DataFrame(index=self.stock_data["DATE"].unique(), columns=tickers)
        Thresh_CAPM.index = pd.to_datetime(Thresh_CAPM.index)
        Thresh_CAPM.index = Thresh_CAPM.index.strftime("%Y-%m-%d")

        # for ticker in tickers:
        #     for date in Thresh_CAPM.index:
        #         Thresh_CAPM.loc[date, ticker] = (
        #             self.market_interest.loc[(self.market_interest["Date"] == date)][
        #                 "Interest_Rate_Returns"
        #             ].values
        #             + self.stock_data.loc[
        #                 (self.stock_data["TICKER"] == ticker) & (self.stock_data["DATE"] == date)
        #             ]["BETA"].values
        #             * self.market_interest.loc[(self.market_interest["Date"] == date)][
        #                 "SP500_Returns"
        #             ].values
        #         )

        return returns, Thresh_CAPM

    # def get_previous_business_day(self, dt) -> dt.datetime:
    #     while True:
    #         if dt.weekday() < 5 and dt not in self.returns.index:
    #             dt -= BusinessDay(1)
    #         else:
    #             return dt
            
    def get_previous_business_day(self, date) -> dt.datetime:
        while date not in self.returns.index:
            date -= BusinessDay(1)
        return date

    def get_previous_returns(self, formation, streak_length=5) -> pd.DataFrame:
        previous_returns = {}
        for i in range(1, self.streak_length + 1):
            previous_day = formation - pd.offsets.BusinessDay(i)
            previous_returns[f"ret_{i}"] = self.returns.loc[
                self.get_previous_business_day(previous_day)
            ]
        return pd.DataFrame(previous_returns)

    def calculate_streak_raw_return(self, x) -> int:
        return 1 if (x > 0).all() or (x < 0).all() else 0

    def calculate_loser_winner_streaks(self, formation, streak_length=5) -> tuple:
        return_streak = pd.DataFrame(self.get_previous_returns(formation, streak_length))
        return_streak["Streak_Raw_Return"] = return_streak.apply(
            lambda x: self.calculate_streak_raw_return(x), axis=1
        ) * return_streak.mean(axis=1)
        return_streak["Streak_Market_Return"] = 0  # TODO: implement
        return_streak["Streak_CAPM"] = 0  # TODO: implement
        # print(return_streak)
        losersRawReturn = return_streak[return_streak["Streak_Raw_Return"] < 0].index
        winnersRawReturn = return_streak[return_streak["Streak_Raw_Return"] > 0].index

        # TODO: Calculate the weighted returns for the losers and winners
        # Calculate the average raw return for losers and winners (Equal weighted)
        # print('formation', formation)
        # print(self.returns)
        # print(self.returns.index)
        losRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(losersRawReturn)
        ].mean()
        winRetRawReturn = self.returns.loc[
            formation, self.returns.columns.isin(winnersRawReturn)
        ].mean() * (-1)

        # return losret, winret
        return losRetRawReturn, winRetRawReturn
    

    def calculate_long_short_portfolio(self, streak_length=5) -> tuple:
        """Calculate the daily returns for given streak length."""
        returns_long = []
        returns_short = []
        dates = []

        # Go through each day but starting from the 6th day since we need 5 days to calculate the streak
        for date in self.returns.index[5:]:
            returns_long.append(self.calculate_loser_winner_streaks(date, streak_length)[0])
            returns_short.append(self.calculate_loser_winner_streaks(date, streak_length)[1])
            dates.append(date)
        return returns_long, returns_short, dates
    

                

    def plot_data(self) -> None:
        print(self.returns)
        print(self.Thresh_CAPM)


# trading = TradingBot()
