# Quantitative Backtesting Engine: Trading Financial Anomalies

This repository contains the backtesting implementation developed for the 'Quantitative Trading Competition' seminar at KIT Karlsruhe. 

The core framework is designed to evaluate a statistical arbitrage strategy based on the paper *Streaks on Daily Returns*. The strategy systematically identifies behavioral market anomalies by classifying stocks into positive or negative streaks based on consecutive daily over- or underperformance against a defined threshold. The baseline hypothesis assumes short-term mean reversion: shorting positive streaks and going long on negative streaks.

## Out-of-Sample Performance
Based on the backtested optimization across historical S&P 500 data:
- **Annualized Sharpe Ratio:** 1.54
- **Daily Excess Return:** 56.3 bps
- **Statistical Significance:** Newey-West adjusted t-statistics > 8.4

## Strategy Parameters

The trading logic is modular. The following hyperparameters can be configured in `src/parameter.json`:

- **Streak Length:** The number of consecutive days an asset must over/underperform to trigger a trading signal.
- **Thresholding Type:** The baseline used to classify performance:
  - `returnRaw`: Absolute positive or negative daily return.
  - `marketExcessReturn`: Return relative to the broader market index.
  - `CAPM`: Return relative to the Capital Asset Pricing Model expected return.
- **Weighting:** Capital allocation method for the triggered stocks:
  - `Equal Weighting`
  - `Relative Weighting`
- **maxStock:** The maximum number of equities traded in a single day.

## Repository Structure

- `data/`: Historical dataset directory (e.g., S&P 500, Fama-French factors)
- `src/`: Core logic including data ingestion and portfolio execution
- `Explanation.ipynb`: Jupyter notebook detailing the mathematical proofs and visualizations
- `main.py`: Main execution script
- `parameter.json`: Configuration file for backtesting parameters

## Setup and Execution

To run the backtesting engine locally, follow these steps:

1. Clone the repository and navigate to the root directory:
```bash
git clone [https://github.com/simonschmitz2405/quantitative_trading_competition.git](https://github.com/simonschmitz2405/quantitative_trading_competition.git)
cd quantitative_trading_competition
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```


3. Modify the strategy hyperparameters as needed in `src/parameter.json`
4. Execute the backtest
```bash
python main.py
```

## Contributors

- [Simon Schmitz]
