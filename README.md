# Portfolio Optimization with a Genetic Algorithm

A Python implementation of a genetic algorithm (GA) that searches for asset weights in an investment portfolio. A Tkinter GUI lets you tune the GA parameters and plots the fitness evolution. A Jupyter notebook downloads market data with `yfinance`, computes annualized returns and covariance matrices, and provides a SciPy (SLSQP) optimizer as a baseline.

> Academic project — Course: **Artificial Intelligence** · Author: **Hasti Shiripoor**

## Features

- Real-valued chromosome encoding: one weight per asset, weights are non-negative and sum to 1
- Fitness = portfolio return / portfolio risk, with an exponential penalty when risk exceeds a threshold (the "Risk Aversion" field)
- Selection: roulette wheel (handles negative fitness) or tournament
- Single-point crossover, mutation, and elitism
- Tkinter GUI: parameter inputs, file selection, weights and percentage allocation, fitness plot
- Notebook with data download (`yfinance`) and a SciPy baseline

## Project Structure

```text
├── src/
│   └── genetic_portfolio.py       # GA + Tkinter GUI
├── notebooks/
│   └── get_data.ipynb             # data download and SciPy baseline
├── data/
│   ├── stocks_10/                 # returns.txt, risks.txt
│   ├── stocks_15/
│   └── commodities_fx_6/
├── docs/
│   └── screenshots/
├── requirements.txt
└── README.md
```

## Requirements

Python 3.x (developed with 3.12), plus the packages in `requirements.txt`. Tkinter is included in the standard Windows installer from python.org.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python src/genetic_portfolio.py
```

1. Click **Proceed to Genetic Algorithm**.
2. Set the parameters (population size, generations, crossover/mutation rates, elitism, selection method).
3. Click **Select Returns File** and **Select Risks File** and choose a pair from `data/`.
4. Click **Run**.

When a returns file is selected, the number of assets is read from the file.

## Data

Each dataset folder contains `returns.txt` (annualized mean daily return × 252, one value per asset) and `risks.txt` (annualized covariance matrix, daily covariance × 252). The asset order is as listed below. Source: Yahoo Finance via `yfinance`.

| Folder | Assets (in order) | Date range |
|---|---|---|
| `stocks_10` | AAPL, GOOGL, AMZN, MSFT, TSLA, META, NFLX, NVDA, JPM, V | 2020-01-01 to 2023-12-31 |
| `stocks_15` | AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, BRK-B, JNJ, XOM, V, PG, UNH, DIS, KO | 2020-01-01 to 2024-12-31 |
| `commodities_fx_6` | GC=F, SI=F, CL=F, PL=F, RUB=X, JPY=X | 2020-01-01 to 2024-12-31 (verify against the notebook) |

The files are static snapshots; re-running the notebook later may give slightly different numbers.

## Results

Test runs on these datasets with the default parameters (population 200, 100 generations, crossover 0.85, mutation 0.05, elitism 3, risk threshold 1). Values vary slightly from run to run.

| Dataset | GA, roulette | GA, tournament | SciPy baseline |
|---|---:|---:|---:|
| `stocks_10` | 1.3704 | 1.3704 | 1.3704 |
| `stocks_15` | 1.5703 | 1.5705 | 1.5705 |
| `commodities_fx_6` | 1.1270 | 1.1270 | 1.1270 |

The metric is return / risk (no risk-free rate). Add screenshots of the GUI to `docs/screenshots/`.

## Known Limitations

- The metric is return / risk without a risk-free rate, so it is not a full Sharpe ratio.
- Annualized returns from a short historical window are noisy. Some assets (e.g. `RUB=X`, `GC=F`) show extreme values in `commodities_fx_6`; treat those results with caution.
- The "Risk Aversion" field acts as a risk threshold: the penalty only applies when portfolio risk exceeds it.
- The notebook uses the `'Adj Close'` column. Newer `yfinance` versions adjust prices by default and may not return it; if you get a `KeyError`, use `'Close'` or pass `auto_adjust=False`.

## Future Improvements

- Add a risk-free rate and per-asset weight limits
- Run multiple GA trials and report mean and standard deviation
- Save the downloaded data directly from the notebook

## Data and License Notes

Market data comes from Yahoo Finance; check their terms of use before redistributing. No license has been specified for this repository. Add an appropriate license only after confirming the project's reuse requirements and any applicable course rules.