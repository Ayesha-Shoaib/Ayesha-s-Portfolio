# Ayesha-s-Portfolio
Data Science Portfolio

I focused on collecting and analyzing data for a Canadian investor's portfolio, comprising a riskless asset (T-bill ETF) and a risky asset (S&P500 stock). Historical data was gathered for the Canadian T-bill ETF and S&P500 stock index from May 19, 2000, to June 6, 2018. Additionally, CAD/USD exchange rates were utilized to convert S&P500 returns to CAD.

For the T-bill ETF, various metrics were calculated, including gross annual bond yield, log returns, deannualized bond yield, and daily percentage change. Graphs depicting the daily net bond yield and log returns were plotted, and averages and standard deviations of returns were computed.

For the S&P500, calculations included daily net return in USD and its conversion to CAD, as well as the corresponding gross net return. Graphs for daily net return and log returns were generated, along with analyses for the S&P500 ETF stock index.

I then delved into constructing different weighted portfolios. The initial steps involved computing historical averages, variances, and standard deviations for the risky asset. Subsequently, various weighted portfolios, including equal weights and dynamically re-weighted portfolios, were created and analyzed. Time series graphs were plotted, and the Sharpe Ratio was calculated for portfolio evaluation.

The fourth part aimed at constructing an optimal mean-variance portfolio for investors with varying risk aversions. Excess returns were computed for different risk aversion levels, revealing that high-risk-averse individuals tended to allocate less weight to the risky asset. Mean, variance and Sharpe Ratio were determined for each investor, and portfolio returns and values over time were visualized.

The final stage centered on testing for a unit root in the S&P500 risky asset's price and return. Augmented Dickey-Fuller (ADF) tests were employed to assess the presence of a unit root. The results indicated a unit root in the price series, leading to the adoption of an ARIMA model for compensation.

For daily returns, ADF tests were conducted to evaluate the presence of a random walk. Results suggested an absence of significant evidence for a random walk but indicated the lack of a trend or drift. Consequently, an ARMA model was chosen, emphasizing autoregressive and weighted moving average processes.

I used R and Python both.

```R:

install.packages(c("tidyquant", "ggplot2", "TTR", "Metrics", "tseries", "forecast"))

library(tidyquant)
library(ggplot2)
library(TTR)
library(Metrics)
library(tseries)
library(forecast)

# Load data using tidyquant
symbols <- c("CADUSD=X", "^IRX", "^GSPC")
start_date <- "2000-05-19"
end_date <- "2018-06-06"

# Fetch data
financial_data <- tq_get(symbols, from = start_date, to = end_date)

# Data Preprocessing
financial_data <- financial_data %>%
  tq_transmute(select = adjusted,
               mutate_fun = periodReturn,
               period = "daily",
               col_rename = "return") %>%
  tk_tbl()

# Calculate log returns, deannualized bond yield, and other required metrics
financial_data <- financial_data %>%
  mutate(log_returns = log(1 + return),
         deannualized_yield = return / 253)

# Portfolio Analysis
weights <- c(0.5, 0.5)
financial_data <- financial_data %>%
  mutate(portfolio_return = weights[1] * deannualized_yield + weights[2] * return)

# Plotting Graphs
ggplot(financial_data, aes(x = date)) +
  geom_line(aes(y = deannualized_yield, color = "Daily Net Bond Yield")) +
  geom_line(aes(y = log_returns, color = "Log Returns Bond")) +
  labs(title = "Bond Yield Analysis", x = "Date", y = "Value", color = "Series") +
  theme_minimal()

# Sharpe Ratio
sharpe_ratio <- (mean(financial_data$portfolio_return) - mean(financial_data$deannualized_yield)) /
                sd(financial_data$portfolio_return)

# Unit Root Testing and ARIMA Modeling
adf_test_price <- adf.test(financial_data$adjusted, alternative = "stationary")

adf_test_return <- adf.test(financial_data$return, alternative = "stationary")

```Python:

```bash
pip install yfinance
```

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Download S&P500 data
sp500_data = yf.download('^GSPC', start='2000-05-19', end='2018-06-06')

# Download Canadian 3-month treasury bills data (I'm using T-bill ETF as an example)
bond_data = yf.download('BIL', start='2000-05-19', end='2018-06-06')

# Choose another source for exchange rate data, or skip this step if not needed

# Combine datasets
combined_data = pd.merge(bond_data, sp500_data, left_index=True, right_index=True, how='inner')

# Handle missing data
combined_data = combined_data.ffill().bfill()

# Data Preprocessing
combined_data['log_returns_bond'] = np.log(1 + combined_data['Close_x'].pct_change())
combined_data['deannualized_yield'] = combined_data['Close_x'] / 253

# Portfolio Analysis
weights = {'bond': 0.5, 'sp500': 0.5}
combined_data['portfolio_return'] = weights['bond'] * combined_data['deannualized_yield'] + \
                                     weights['sp500'] * combined_data['Close_y'].pct_change()

# Calculate mean, variance, standard deviation
mean_portfolio = combined_data['portfolio_return'].mean()
variance_portfolio = combined_data['portfolio_return'].var()
std_dev_portfolio = combined_data['portfolio_return'].std()

# Plotting Graphs
plt.plot(combined_data.index, combined_data['deannualized_yield'], label='Daily Net Bond Yield')
plt.plot(combined_data.index, combined_data['log_returns_bond'], label='Log Returns Bond')
plt.title('Bond Yield Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Sharpe Ratio
sharpe_ratio = (mean_portfolio - combined_data['deannualized_yield'].mean()) / std_dev_portfolio

# Unit Root Testing and ARIMA Modeling
result_price = sm.tsa.adfuller(combined_data['Close_y'])
# Check the result and choose the appropriate model for price


result_return = sm.tsa.adfuller(combined_data['portfolio_return'].dropna())
```
