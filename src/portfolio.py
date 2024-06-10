import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy import stats
import statsmodels.api as sm
import json
from pathlib import Path
from statsmodels.regression.rolling import RollingOLS


class Portfolio:
    """The Portfolio class."""

    def __init__(self, streakLength, maxStock) -> None:
        """Initialize the Portfolio class."""

        # Define the Paths
        base_path = Path(__file__).resolve().parent.parent
        self._pathSp500StockData = base_path / "data" / "sp500_stock_data.csv"
        self._pathMarketReturns = base_path / "data" / "market_interest.csv"
        self._pathFfDaily = base_path / "data" / "ff_daily.csv"
        self._parameters = base_path / "src" / "parameter.json"
        self._logging = base_path / "src" / "logfile.log"

        # Define the logging
        logging.basicConfig(
            level=logging.INFO,
            filename=self._logging,
            filemode="w",
        )

        # Read the CSV file an store data as dataframes
        self.stockData, self.marketReturns, self.interestRate = self._read_csv()

        # Initialize the hyperparameters
        self.streakLength = streakLength
        self.maxStock = maxStock

        # Run the trading strategy
        self._run()

        # Calculate the list (!) of the risk-free rate
        self.riskFreeRate, self.marketReturns = self._calculate_riskfree_market_rate()

    def _run(self) -> None:
        """Run the trading strategy."""
        # Prepare the data for the trading strategy
        self._prepare_data()

        # Calculate the beta of the stock
        self._calculate_beta()

        # Calculate the lags
        self._calculate_lags()

        # Calculate the streaks
        self._calculate_streaks()

        self.Returns = self._calculate_returns()

        return None

    def _read_csv(self) -> tuple:
        """Read the CSV file."""
        try:
            stockData = pd.read_csv(self._pathSp500StockData)
            marketReturns = pd.read_csv(self._pathMarketReturns)
            interestRate = pd.read_csv(self._pathFfDaily)
        except Exception as e:
            print(f"Error loading data: {e}")
        return stockData, marketReturns, interestRate

    def _prepare_data(self) -> None:
        """Prepare the data for the trading strategy."""
        # Sort stock data by DATE for each ticker
        self.stockData.sort_values(by=["TICKER", "DATE"], inplace=True)
        self.stockData = pd.merge(self.stockData, self.marketReturns, on="DATE", how="left")
        self.interestRate.rename(columns={"Date": "DATE"}, inplace=True)
        self.stockData = pd.merge(self.stockData, self.interestRate, on="DATE", how="left")

    def _calculate_lags(self) -> None:
        # Create lagged return columns
        for lag in range(1, self.streakLength + 1):
            self.stockData[f"LAG_{lag}"] = self.stockData.groupby("TICKER")["RETURN"].shift(lag)

        # Calculate excess market return lagged columns
        for lag in range(1, self.streakLength + 1):
            self.stockData[f"LAG_{lag}_MARKET"] = self.stockData.groupby("TICKER")["RETURN"].shift(
                lag
            ) - self.stockData.groupby("TICKER")["SP500_Returns"].shift(lag)

        for lag in range(1, self.streakLength + 1):
            self.stockData[f"LAG_{lag}_CAPM"] = self.stockData.groupby("TICKER")["RETURN"].shift(
                lag
            ) - (
                self.stockData.groupby("TICKER")["RF"].shift(lag)
                + self.stockData.groupby("TICKER")["BETA"].shift(lag)
                * (
                    self.stockData.groupby("TICKER")["SP500_Returns"].shift(lag)
                    - self.stockData.groupby("TICKER")["RF"].shift(lag)
                )
            )

        # Calculate the market cap weighted returns which will be normalized by value afterwards
        self.stockData["RETURN*MARKET_CAP"] = (
            self.stockData["RETURN"] * self.stockData["MARKET_CAP"]
        )

        self.stockData.fillna(0, inplace=True)

        return None

    def _calculate_beta(self) -> None:
        """Calculate the beta of the stock."""
        window = 252
        beta_results = pd.DataFrame()

        self.stockData = self.stockData.sort_values(by=["TICKER", "DATE"])

        grouped = self.stockData.groupby("TICKER")
        for name, group in grouped:
            if len(group) >= window:
                model = RollingOLS(
                    group["RETURN"], sm.add_constant(group["SP500_Returns"]), window=window
                )
                rres = model.fit()
                betas = rres.params["SP500_Returns"]
                temp = group[["DATE", "TICKER"]].copy()
                temp["BETA"] = betas.values
                beta_results = pd.concat([beta_results, temp])

        self.stockData = pd.merge(self.stockData, beta_results, on=["DATE", "TICKER"], how="left")
        return None

    def _calculate_streaks(self) -> None:
        """Calculate the streaks for the total dataFrame."""
        # Calculate streaks for each lag for the two types of thresholding
        for streak in range(1, self.streakLength + 1):
            # Thresholding based on raw returns
            streak_condition_raw = (
                self.stockData[[f"LAG_{i}" for i in range(1, streak + 1)]] > 0
            ).all(axis=1) | (self.stockData[[f"LAG_{i}" for i in range(1, streak + 1)]] < 0).all(
                axis=1
            )
            # Thresholding based on excess returns
            streak_condition_excess = (
                self.stockData[[f"LAG_{i}_MARKET" for i in range(1, streak + 1)]] > 0
            ).all(axis=1) | (
                self.stockData[[f"LAG_{i}_MARKET" for i in range(1, streak + 1)]] < 0
            ).all(axis=1)

            # streak_condition_capm = (
            #     self.stockData[[f"LAG_{i}_CAPM" for i in range(1, streak + 1)]] > 0

            # Create columns for the streaks based on the thresholding conditions
            # Streak: 1, No streak: 0
            self.stockData[f"STREAK_{streak}"] = np.where(streak_condition_raw, 1, 0)
            self.stockData[f"STREAK_{streak}_MARKET"] = np.where(
                streak_condition_excess, 1, self.stockData[f"STREAK_{streak}"]
            )

            # Multiply streaks by the mean of the respective lagged returns
            # Positive Streak: Positive mean, Negative Streak: Negative mean
            self.stockData[f"STREAK_{streak}"] *= self.stockData[
                [f"LAG_{i}" for i in range(1, streak + 1)]
            ].mean(axis=1)
            self.stockData[f"STREAK_{streak}_MARKET"] *= self.stockData[
                [f"LAG_{i}_MARKET" for i in range(1, streak + 1)]
            ].mean(axis=1)

        return None

    def _get_top_rows(self, group, streak, condition, market) -> pd.DataFrame:
        """Helper function to get top maxStock rows for a given condition."""
        if condition == "long":
            if market == True:
                return group[group[f"STREAK_{streak}_MARKET"] < 0].nsmallest(self.maxStock, f"STREAK_{streak}_MARKET")
            else:
                return group[group[f"STREAK_{streak}"] < 0].nsmallest(self.maxStock, f"STREAK_{streak}")
        if condition == "short":
            if market == True:
                return group[group[f"STREAK_{streak}_MARKET"] > 0].nlargest(self.maxStock, f"STREAK_{streak}_MARKET")
            else:
                return group[group[f"STREAK_{streak}"] > 0].nlargest(self.maxStock, f"STREAK_{streak}")
        elif condition == "trade":
            if market == True:
                return group[group[f"STREAK_{streak}_MARKET"] < 0]
            else:
                return group[group[f"STREAK_{streak}"] < 0]

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate the loser and winner streaks for the given formation date."""
        Returns = pd.DataFrame()

        # Ensure 'DATE' column in stockData is datetime
        self.stockData["DATE"] = pd.to_datetime(self.stockData["DATE"])

        # Initialize the Returns DataFrame with unique dates from stockData
        unique_dates = self.stockData["DATE"].unique()
        Returns["DATE"] = pd.to_datetime(unique_dates)

        # Initialize dictionaries to store long and short streaks
        # Thresholding raw Returns
        long_streaks = {}
        short_streaks = {}
        # Thresholding excess Returns
        long_streaks_market = {}
        short_streaks_market = {}

        for streak in range(1, self.streakLength + 1):
            # Define the key names for the long and short streaks for raw and excess returns
            key_name_long = f"Long_{streak}"
            key_name_short = f"Short_{streak}"
            key_name_long_market = f"Long_{streak}_MARKET"
            key_name_short_market = f"Short_{streak}_MARKET"

            if streak == 4:
                streak_today = (
                    self.stockData.groupby("DATE")
                    .apply(self._get_top_rows, streak=streak, condition="trade", market=True)
                    .reset_index(drop=True)
                )
                today = pd.Timestamp("today").normalize()
                streak_today = streak_today[streak_today["DATE"] == today]
                filter = streak_today["RETURN"] < streak_today["SP500_Returns"]
                streak_today.where(filter, inplace=True)
                streak_today.dropna(inplace=True)
                streak_today = streak_today.nsmallest(self.maxStock, "STREAK_4_MARKET")
                sum_market_cap = streak_today["MARKET_CAP"].sum()
                streak_today["WEIGHT"] = streak_today["MARKET_CAP"] / sum_market_cap
                print(streak_today.info())
                # print("TRADE: \n", streak_today["TICKER","WEIGHT"])
                print(streak_today)

            # Filter data for long and short streaks for raw returns
            long_streaks[key_name_long] = (
                self.stockData.groupby("DATE")
                .apply(self._get_top_rows, streak=streak, condition="long", market=False)
                .reset_index(drop=True)
            )
            short_streaks[key_name_short] = (
                self.stockData.groupby("DATE")
                .apply(self._get_top_rows, streak=streak, condition="short", market=False)
                .reset_index(drop=True)
            )

            # Filter data for long and short streaks for excess returns
            long_streaks_market[key_name_long_market] = (
                self.stockData.groupby("DATE")
                .apply(self._get_top_rows, streak=streak, condition="long", market=True)
                .reset_index(drop=True)
            )
            short_streaks_market[key_name_short_market] = (
                self.stockData.groupby("DATE")
                .apply(self._get_top_rows, streak=streak, condition="short", market = True)
                .reset_index(drop=True)
            )

            # Ensure 'DATE' columns in long_streaks are datetime
            long_streaks[key_name_long]["DATE"] = pd.to_datetime(
                long_streaks[key_name_long]["DATE"]
            )
            short_streaks[key_name_short]["DATE"] = pd.to_datetime(
                short_streaks[key_name_short]["DATE"]
            )
            long_streaks_market[key_name_long_market]["DATE"] = pd.to_datetime(
                long_streaks_market[key_name_long_market]["DATE"]
            )
            short_streaks_market[key_name_short_market]["DATE"] = pd.to_datetime(
                short_streaks_market[key_name_short_market]["DATE"]
            )

            value_long = long_streaks[key_name_long].groupby("DATE")["MARKET_CAP"].sum()
            value_short = short_streaks[key_name_short].groupby("DATE")["MARKET_CAP"].sum()
            value_long_market = (
                long_streaks_market[key_name_long_market].groupby("DATE")["MARKET_CAP"].sum()
            )
            value_short_market = (
                short_streaks_market[key_name_short_market].groupby("DATE")["MARKET_CAP"].sum()
            )


            # if streak == 4:
            #     today = pd.Timestamp("today").normalize()
            #     streak_today = long_streaks[key_name_long][
            #         long_streaks[key_name_long]["DATE"] == today
            #     ]

            #     print("HIER")
            #     print(streak_today)
            #     sum_market_cap = streak_today["MARKET_CAP"].sum()
            #     print("sum", sum_market_cap)
            #     streak_today["WEIGHT"] = streak_today["MARKET_CAP"] / sum_market_cap
            #     print("HIER2")
            #     print(streak_today)
                # print("TRADE: \n", streak_today["TICKER","WEIGHT"])




            # Calculate the mean returns for equal weights for long and short streaks
            mean_returns_long = (
                long_streaks[key_name_long]
                .groupby("DATE")["RETURN"]
                .mean()
                .rename(f"LONG_{streak}_EQUAL_RAW")
            )
            mean_returns_short = (
                short_streaks[key_name_short]
                .groupby("DATE")["RETURN"]
                .mean()
                .rename(f"SHORT_{streak}_EQUAL_RAW")
            )
            mean_returns_long_market = (
                long_streaks_market[key_name_long_market]
                .groupby("DATE")["RETURN"]
                .mean()
                .rename(f"LONG_{streak}_EQUAL_MARKET")
            )
            mean_returns_short_market = (
                short_streaks_market[key_name_short_market]
                .groupby("DATE")["RETURN"]
                .mean()
                .rename(f"SHORT_{streak}_EQUAL_MARKET")
            )

            # Calculate the relative returns for long and short streaks
            relative_returns_long = (
                long_streaks[key_name_long]
                .groupby("DATE")["RETURN*MARKET_CAP"]
                .sum()
                .div(value_long)
                .rename(f"LONG_{streak}_RELATIVE_RAW")
            )
            relative_returns_short = (
                short_streaks[key_name_short]
                .groupby("DATE")["RETURN*MARKET_CAP"]
                .sum()
                .div(value_short)
                .rename(f"SHORT_{streak}_RELATIVE_RAW")
            )
            relative_returns_long_market = (
                long_streaks_market[key_name_long_market]
                .groupby("DATE")["RETURN*MARKET_CAP"]
                .sum()
                .div(value_long_market)
                .rename(f"LONG_{streak}_RELATIVE_MARKET")
            )
            relative_returns_short_market = (
                short_streaks_market[key_name_short_market]
                .groupby("DATE")["RETURN*MARKET_CAP"]
                .sum()
                .div(value_short_market)
                .rename(f"SHORT_{streak}_RELATIVE_MARKET")
            )

            # Merge the mean returns into the Returns DataFrame
            Returns = pd.merge(Returns, mean_returns_long.reset_index(), on="DATE", how="left")
            Returns = pd.merge(Returns, mean_returns_short.reset_index(), on="DATE", how="left")
            Returns = pd.merge(
                Returns, mean_returns_long_market.reset_index(), on="DATE", how="left"
            )
            Returns = pd.merge(
                Returns, mean_returns_short_market.reset_index(), on="DATE", how="left"
            )
            Returns = pd.merge(Returns, relative_returns_long.reset_index(), on="DATE", how="left")
            Returns = pd.merge(Returns, relative_returns_short.reset_index(), on="DATE", how="left")
            Returns = pd.merge(
                Returns, relative_returns_long_market.reset_index(), on="DATE", how="left"
            )
            Returns = pd.merge(
                Returns, relative_returns_short_market.reset_index(), on="DATE", how="left"
            )

        Returns.drop(columns=["DATE"], inplace=True)
        Returns.fillna(0, inplace=True)
        return Returns

    def _calculate_riskfree_market_rate(self) -> tuple:
        """Calculate the risk-free rate."""
        self.marketReturns.fillna(0, inplace=True)
        return self.interestRate["RF"].tolist(), self.marketReturns["SP500_Returns"].tolist()

    def _get_t_stats(self, vector, lags=1) -> tuple:
        """Calculate the t-stats for the long and short portfolios with Newey-West standard errors."""
        # Add a constant to the vector for OLS regression
        X = np.ones(len(vector))
        model = sm.OLS(vector, X)

        # Fit the model
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

        # Get the t-statistic and p-value
        t = results.tvalues[0]
        p = results.pvalues[0]

        return t, p

    def _calculate_performance(self, dailyReturns) -> None:
        """Calculate the performance of the portfolio."""

        # Daily average returns
        dailyReturn = np.array(dailyReturns).mean()
        dailyReturnRiskFree = np.array(self.riskFreeRate).mean()
        dailyReturnMarket = np.array(self.marketReturns).mean()
        # Daily standard deviation
        dailystandarddeviation = np.array(dailyReturns).std()
        dailystandarddeviationRiskFree = np.array(self.riskFreeRate).std()
        dailystandarddeviationMarket = np.array(self.marketReturns).std()
        # Annualized average returns
        annualizedReturn = dailyReturn * 252
        annualizedReturnRiskFree = dailyReturnRiskFree * 252
        annualizedReturnMarket = dailyReturnMarket * 252
        # Annualized standard deviation
        annualizedStandardDeviation = dailystandarddeviation * np.sqrt(252)
        annualizedStandardDeviationRiskFree = dailystandarddeviationRiskFree * np.sqrt(252)
        annualizedStandardDeviationMarket = dailystandarddeviationMarket * np.sqrt(252)
        # Sharpe Ratio annualized
        sharpeRatioLong = (
            annualizedReturn - annualizedReturnRiskFree
        ) / annualizedStandardDeviation

        dailyexcessriskfreelong = np.array(dailyReturns) - np.array(self.riskFreeRate)
        dailyexcessmarketlong = np.array(dailyReturns) - np.array(self.marketReturns)

        tstatsexcessriskfreelong = self._get_t_stats(dailyexcessriskfreelong)
        tstatsexcessmarketlong = self._get_t_stats(dailyexcessmarketlong)

        print("Performance of the portfolio:")
        print("#---------------------------")
        print("Daily average returns: ", dailyReturn)
        print("Daily average risk free rate: ", dailyReturnRiskFree)
        print("Daily average market return: ", dailyReturnMarket)
        print("#---------------------------")
        print("Daily standard deviation: ", dailystandarddeviation)
        print("Daily standard deviation risk free rate: ", dailystandarddeviationRiskFree)
        print("Daily standard deviation market return: ", dailystandarddeviationMarket)
        print("#---------------------------")
        print("Annualized average returns: ", annualizedReturn)
        print("Annualized average risk free rate: ", annualizedReturnRiskFree)
        print("Annualized average market return: ", annualizedReturnMarket)
        print("#---------------------------")
        print("Annualized standard deviation: ", annualizedStandardDeviation)
        print("Annualized standard deviation risk free rate: ", annualizedStandardDeviationRiskFree)
        print("Annualized standard deviation market return: ", annualizedStandardDeviationMarket)
        print("#---------------------------")
        print(
            "Excess return risk free annualized:",
            annualizedReturn - annualizedReturnRiskFree,
        )

        print("Excess return market annualized:", annualizedReturn - annualizedReturnMarket)
        print("#---------------------------")
        print("Sharpe Ratio: ", sharpeRatioLong)
        print("#---------------------------")
        print("T-Stats excess risk free: ", tstatsexcessriskfreelong)
        print("T-Stats excess market: ", tstatsexcessmarketlong)

        logging.info("Daily average returns: %s", dailyReturn)
        logging.info("Daily average risk free rate: %s", dailyReturnRiskFree)
        logging.info("Daily average market return: %s", dailyReturnMarket)
        logging.info("Daily standard deviation: %s", dailystandarddeviation)
        logging.info("Daily standard deviation risk free rate: %s", dailystandarddeviationRiskFree)
        logging.info("Daily standard deviation market return: %s", dailystandarddeviationMarket)
        logging.info("Annualized average returns: %s", annualizedReturn)
        logging.info("Annualized average risk free rate: %s", annualizedReturnRiskFree)
        logging.info("Annualized average market return: %s", annualizedReturnMarket)
        logging.info("Annualized standard deviation: %s", annualizedStandardDeviation)
        logging.info(
            "Annualized standard deviation risk free rate: %s", annualizedStandardDeviationRiskFree
        )
        logging.info(
            "Annualized standard deviation market return: %s", annualizedStandardDeviationMarket
        )
        logging.info(
            "Excess return risk free annualized:%s",
            annualizedReturn - annualizedReturnRiskFree,
        )

        logging.info(
            "Excess return market annualized:%s",
            annualizedReturn - annualizedReturnMarket,
        )

        logging.info("Sharpe Ratio: %s", sharpeRatioLong)
        logging.info("T-Stats excess risk free: %s", tstatsexcessriskfreelong)
        logging.info("T-Stats excess market: %s", tstatsexcessmarketlong)

    def _print_portfolio_details(self, column) -> None:
        """Print the details of the portfolio."""
        with open(self._parameters) as file:
            portfolio_details = json.load(file)

        details = portfolio_details.get(column)
        if details:
            print("Parameter of the portfolio:")
            print("#---------------------------")
            print(f"Portfolio: {details['Portfolio']}")
            print(f"Streak: {details['Streak']}")
            print(f"Weighting: {details['Weighting']}")
            print(f"Thresholding: {details['Thresholding']}")
            print("Max Stocks: ", self.maxStock)
        else:
            print(f"No details found for column: {column}")

    def calculate_performance_all(self) -> None:
        """Calculate the performance of the portfolio for all columns in the Returns DataFrame."""
        for column in self.Returns.columns:
            print("#============================" "=============================#\n")
            self._print_portfolio_details(column)
            self._calculate_performance(self.Returns[column].tolist())
        return None
