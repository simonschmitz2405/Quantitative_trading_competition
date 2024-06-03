# Quantitative Trading Competition - Trading Financial Anamalies in the Market

This repository is the Backtesting Implementation for the seminar 'Quantitative Trading Competition', held at KIT Karlsruhe in the summer semester 2024. The main goal of the seminar is to develop a trading strategy based on the paper 'Streaks on daily returns'. Hereby daily, stocks are classified as n positive or negative streaks if they over or underperformed based on a certain threshold. The idea is to go the postive streaks short and go the negative streaks long. In this implementation the parameter of the strategy are the following:
- Streak length: How many consecutive a streak has to under- or overperformed
- Thresholding type: Which threshold is used to classify an under- or overperformance. In this implementation there are two threshold types:
    - returnRaw: Check if the return was positive or negative
    - marketExcessReturn: Check if the return was higher or lower than the respective market return
- Weighting: Which percentages of losers or winner we buy/sell. In this implementation there are two options:
    - Equal weighting
    - Value weighting
- Long/Short: How much leverage is used for the given portfolio.


. The repository contains the following files:

- `data/`: Folder containing the data used in the seminar
- `src/`: Folder containing the source code of the trading strategy
- `README.md`: This file

## Contributors

- [Simon Schmitz]