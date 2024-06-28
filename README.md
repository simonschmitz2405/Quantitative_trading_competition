# Quantitative Trading Competition - Trading Financial Anamalies in the Market

This repository is the Backtesting Implementation for the seminar 'Quantitative Trading Competition', held at KIT Karlsruhe in the summer semester 2024. The main goal of the seminar is to develop a trading strategy based on the paper 'Streaks on daily returns'. Hereby daily, stocks are classified as n positive or negative streaks if they over or underperformed based on a certain threshold. The idea is to go the postive streaks short and go the negative streaks long. In this implementation the parameter of the strategy are the following:
- Streak Length: Determines the number of consecutive days a streak must over or underperform.

- Thresholding Type: Specifies the threshold used to classify over or underperformance. Three threshold types are implemented:

    - returnRaw: Determines if the return was positive or negative.
    - marketExcessReturn: Compares the return to the respective market return.
    - CAPM: Compares the return respective to CAPM.

- Weighing: Determines in which percentage the streak stocks are bought:
    - Equal Weighting
    - Relative Weighting

- maxStock: Maximum number of stocks traded each day.

. The repository contains the following files:

- `data/`: Folder containing the data used in the seminar
- `src/`: Folder containing the source code of the trading strategy
- `README.md`: This file

## Contributors

- [Simon Schmitz]

## Usage:

To utilize the trading strategy implementation, follow these steps:

Clone the repository to your local machine.
Navigate to the src/ directory.
Modify the strategy parameters as needed in the provided source code.
Execute the trading strategy implementation.
Analyze the results and adjust parameters as necessary for optimization.
