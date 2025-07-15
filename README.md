# Stock-Portfolio
This study aims to evaluate whether aggressive or defensive stocks provide superior riskadjusted returns for investors seeking to optimize their portfolios during the ongoing tariff wars. Considering the heightened market volatility caused by the implementation of reciprocal tariffs in April 2025, we also touched on identifying financial instruments that can help maximize riskreturn tradeoff for individual investors. 

## Research Methodology
To address this objective, we compared the Sharpe ratios of aggressive and defensive stock portfolios, using the portfolio with the higher ratio as the optimal risky portfolio, which is then combined with risk-free assets to form the complete portfolio.

## Stock Portfolio
### Aggressive Stocks
For the aggressive stocks, we selected from the technology sector due to their extensive global supply chains, significant import-export volumes, high sensitivity to cost fluctuations and market uncertainty, and strong reliance on innovation. We selected Apple, Microsoft, and NVIDIA for aggressive stocks

### Defensive Stocks
For defensive stocks, we chose stocks from customer staples industries, as these assets are characterized by diversified product offerings, stable and consistent consumer demand,and lower sensitivity to market volatility, particularly for necessities. We selected gold, Procter & Gamble, and Walmart for defensive stocks.

## Data Sources
We obtained stock performance data from Yahoo! Finance, covering 2022 to 2025, to capture the post-COVID, tariff war environment. To enhance the robustness of our approach, we conducted both two-asset and three-asset optimization models (the latter incorporating the S&P 500 as the market portfolio), with the objective of determining portfolio weights that accommodate varying levels of risk aversion. Ultimately, we assume that all investors will hold the same optimal risky portfolio, adjusting allocations between the risky and risk-free assets according to individual risk tolerance.

## 1. Quantitative Analysis
## Defensive Stocks
For the defensive portfolio, where we expect the selected assets to have beta < 1, we chose gold, Procter & Gamble, and Walmart, as they operate in the consumer staple and retail sector where demand for goods is more stable and less sensitive to market fluctuations, especially in the aftermath of COVID-19 and amid the ongoing tariff wars in 2025. As noted earlier, this study focuses on the post-COVID period, and hence, our data sample covers the years 2022 to 2025. The stock price movements of these assets can be seen in Appendix 1. For easier comparison, we utilized a dual-axis graph to represent the PG stock (orange line).

### 1a. Download the closing price for Gold, Walmart, and P&G
``` Python
pip install yfinance --upgrade --no-cache-dir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
# Downloading data for 2 companies & gold
stocks = yf.download(['GLD', 'PG','WMT'], start="2022-05-31", end="2025-05-31").Close
print(stocks)
```
<img width="896" height="300" alt="image" src="https://github.com/user-attachments/assets/248c4a15-1605-4c18-94c0-ffc735747e9d" />

### 1b. Compute the annual returns and annual std deviation
``` Python
daily_simple_returns = stocks.pct_change()
daily_mean_return = daily_simple_returns.aggregate(np.mean)
daily_std = daily_simple_returns.aggregate(np.std)

annual_simple_returns = daily_mean_return*252
annual_std = daily_std*np.sqrt(252)

print("Daily simple returns:")
print(daily_simple_returns)
print("\n")

print("Daily mean returns:")
print(daily_mean_return)
print("\n")

print("Daily std:")
print(daily_std)
print("\n")

print("Annual returns:")
print((annual_simple_returns)*100)
print("\n")

print("Annual std:")
print((annual_std)*100)
```
<img width="327" height="832" alt="image" src="https://github.com/user-attachments/assets/6e17145c-0f4d-420b-9fa5-2eba59e1b4db" />

### 1c. Compute the annual covariace and daily correlation
``` Python
daily_covariance = daily_simple_returns.cov()
annual_covariance = daily_covariance*252

daily_correlation = daily_simple_returns.corr()
print("annual_covariance:")
print(annual_covariance)
print("\n")

print("Daily correlation:")
print(daily_correlation)

#Correlation: all have positive correlation but under 1.0. The correlation ranging from 0.526468 to 0.677539
```
<img width="293" height="241" alt="image" src="https://github.com/user-attachments/assets/4f19b81c-a4f3-4670-ae49-2bda23f3203c" />

``` Python
# Plot the respective graphs
#problem - different scale
stocks.plot(title='Plot of GLD, PG, WMT Prices', figsize=(14,10), grid=True);
```
<img width="1139" height="765" alt="image" src="https://github.com/user-attachments/assets/54e2a87b-90f1-473f-bb32-2892405a1ff6" />

``` Python
# Include a secondary axis for better visualization
#secondary y-axis - for NVDIA a
stocks.plot(secondary_y = ['PG'], title='Plot of GLD, PG, WMT Prices', figsize=(14,10), grid=True);
```
<img width="1182" height="763" alt="image" src="https://github.com/user-attachments/assets/cab8c9c6-dc32-4387-9ad4-a6e258fdfbf4" />

## 2. Assessing The Stock's Beta
### 2a. Regression Analysis
``` Python
!pip install statsmodels

# Modules or regression purpose
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as sd

# For plotting purpose
import seaborn as sns

# Specify the starting and ending dates for the time series
start = "2022-05-31"
end  = "2025-05-31"

ticker = ['GLD', 'PG','WMT', '^GSPC']
#S&P 500 Index for market benchmark

stocks = yf.download(ticker, start, end).Close
print(stocks)

#stocks_df=pd.DataFrame(stocks)
stocks_return = stocks.pct_change()
stocks_return = stocks_return.rename(columns = {"GLD": "GoldRet", "PG": "PGRet","WMT": "WMTRet", "^GSPC": "SP500Ret"})
print(stocks_return)
```
<img width="980" height="561" alt="image" src="https://github.com/user-attachments/assets/5ab58e83-35e1-4f48-8a4e-bbf646120400" />

``` Python
# OLS regression of Gold against S&P500
reg4 = smf.ols(formula='GoldRet ~ SP500Ret',data=stocks_return).fit()
print(reg4.summary()) #beta <1

# OLS regression of P&G against S&P500
reg5 = smf.ols(formula='PGRet ~ SP500Ret',data=stocks_return).fit()
print(reg5.summary()) #beta <1

# OLS regression of WMT against S&P500
reg6 = smf.ols(formula='WMTRet ~ SP500Ret',data=stocks_return).fit()
print(reg6.summary()) #beta <1
```
<img width="731" height="873" alt="image" src="https://github.com/user-attachments/assets/7e5d6bfe-309d-415f-8f21-2383a44cda96" />
<img width="706" height="436" alt="image" src="https://github.com/user-attachments/assets/bcd60a9d-a50b-4a88-b39f-9394cd1edcc4" />

### 2b. Plotting the Regression Analysis
``` Python
import matplotlib.pyplot as plt
import seaborn as sns

# Set up a 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Regression Plots: Stock Returns vs. S&P 500', fontsize=16)

# Plot GOLD vs S&P500
sns.regplot(ax=axes[0, 0], x='SP500Ret', y='GoldRet', data=stocks_return)
axes[0, 0].set_title('GOLD vs. S&P 500')

# Plot PG vs S&P500
sns.regplot(ax=axes[0, 1], x='SP500Ret', y='PGRet', data=stocks_return)
axes[0, 1].set_title('PG vs. S&P 500')

# Plot WMT vs S&P500
sns.regplot(ax=axes[1, 0], x='SP500Ret', y='WMTRet', data=stocks_return)
axes[1, 0].set_title('WMT vs. S&P 500')

# Improve layout
fig.delaxes(axes[1, 1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```
<img width="1391" height="886" alt="image" src="https://github.com/user-attachments/assets/05eda928-e192-48ce-9afd-97ec61e51ad9" />

**Explanation:**

To further validate that the selected stocks are defensive stocks, a regression analysis against the market portfolio (S&P 500) is conducted. Based on the regression analysis results, it is proven that the beta for GLD, P&G, and Walmart are all below 1 which means that the changes in each of the individual stocks are less than the market changes and therefore they will be classified under defensive stocks.

## 3. Portfolio Optimization
Next, we defined the weight of each stock in the portfolio and computed the corresponding return, volatility, and Sharpe ratio.

### 3a. Compute the Portfolio's Returns, Volatility, Sharpe Ratio and the Respective Weights
``` Python
# Create empty lists to store returns, volatility and weights of imaginary portfolios
port_returns = []
port_volatility = []
port_sharpe = []
stock_weights = []

# set the number of combinations (1,000) for imaginary portfolios and risk-free rate = 4.46%
num_assets = 3
num_portfolios = 1000
rf = 0.0446

for single_portfolio in range(num_portfolios):
    # Select random weights for portfolio holdings
    weights = np.random.random(num_assets)

    # Rebalance weights to sum to 1
    weights /= np.sum(weights)

    # Formula for returns and variance using matrix multiplication
    returns = np.dot(weights, annual_simple_returns)
    variance = np.dot(np.dot(weights, annual_covariance), weights.T)
    sharpe = (returns-rf)/np.sqrt(variance)
    std = np.sqrt(variance)

    # Appending the values to the respective lists
    port_returns.append(returns)
    port_volatility.append(std)
    port_sharpe.append(sharpe)
    stock_weights.append(weights)

    # Printing the weights of each stock and the corresponding expected annual return and risk of portfolio
    print('Weight in GLD, PG, WMT:', weights)
    print('Expected annual return of portfolio:', returns)
    print('Annual volatility of portfolio:', std)
```
<img width="453" height="816" alt="image" src="https://github.com/user-attachments/assets/a1c0c49c-4b35-459b-be30-29ba06437813" />

### 3b. Convert to DataFrame
``` Python
ticker = ['GLD','WMT','PG']
# A dictionary for Returns, Risk, Sharpe Ratio values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': port_sharpe}

# Extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(ticker):
    portfolio[symbol+' weight'] = [weight[counter] for weight in stock_weights]
#   print(counter, symbol)

# Make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# Get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' weight' for stock in ticker]
print(df)
```
<img width="589" height="243" alt="image" src="https://github.com/user-attachments/assets/7f4be168-57f2-4fae-bc21-c6813b038cc5" />

### 3c. Minimum Variance Portfolio and Optimal Portfolio
From the list of weight combinations, we generated the efficient frontier graph, where the red dot represents the optimal portfolio and the yellow star represents the minimum variance portfolio (portfolio with the lowest risk level for a given level of expected return)

``` Python
# Find minimum variance portfolio & optimal portfolio with max Sharpe ratio in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# Use the min, max values to locate and create the two special portfolios
min_variance_port = df.loc[df['Volatility'] == min_volatility]
max_sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]

print('Minimum Variance Portfolio:')
print(min_variance_port)
print('\n')
print('Optimal Portfolio:')
print(max_sharpe_portfolio)
```
<img width="592" height="141" alt="image" src="https://github.com/user-attachments/assets/b0ccce31-222e-49d2-bc6e-6ba1bf4d0a8b" />

**Explanation:**
The optimal risky portfolio (defensive) achieved a return of 0.2606 with a standard deviation of 0.1381, resulting in a Sharpe ratio of 1.5640. The portfolio weights are Gold (WGLD) = 0.4845, Walmart (WWMT) = 0.0025, and Procter & Gamble (WPG) = 0.5131, which sums up to a total weight of 1.00.

### 3d. Plotting the Resulting Frontier
``` Python
# Plot frontier, max sharpe & min Volatility values with a scatterplot
df.plot.scatter(x = 'Volatility', y = 'Returns', figsize = (10, 8))

plt.scatter(x = max_sharpe_portfolio['Volatility'], y = max_sharpe_portfolio['Returns'], c = 'red', marker = 'o', s = 200)
plt.scatter(x = min_variance_port['Volatility'], y = min_variance_port['Returns'], c = 'yellow', edgecolors = 'red', marker = '*', s = 300)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Resulting Frontier')
plt.show();
```
<img width="861" height="695" alt="image" src="https://github.com/user-attachments/assets/42e57732-9f1e-46b5-a0b0-172e26923167" />

**Explanation:**
From this list of weight combinations, we generated the efficient frontier graph , where the red dot represents the optimal portfolio and the yellow star represents the minimum variance portfolio, the portfolio with the lowest risk level for a given level of expected return.

## 4. Y Maximum
After determining the proportion to invest in each stocks, the next step is to incorporate US government treasury bills (a risk-free asset) to the portfolio. Thus, now we need to determine the optimal capital allocation between the risky and risk-free assets.

``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

# Calculate y max
y_star = (expected_return - rf) / (A * (volatility ** 2))
print("Optimal proportion to invest in the risky portfolio (y*):", y_star)

#y* > 1 because a rational investor with the risk profile would benefit from leveraging their investment in the risky portfolio.
```
<img width="589" height="24" alt="image" src="https://github.com/user-attachments/assets/d1c7d841-6433-45ff-b2dc-011510862285" />

## 5. Complete Portfolio Statistics (Mix ofr Risky and Risk Free Asset)
``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

max_sharpe_complete_portfolio = pd.DataFrame()

y_star = min(y_star, 1.0)
max_sharpe_complete_portfolio['Returns'] = y_star * max_sharpe_portfolio['Returns'] + (1-y_star) * rf
max_sharpe_complete_portfolio['Volatility'] = y_star * max_sharpe_portfolio['Volatility']
max_sharpe_complete_portfolio['Sharpe Ratio'] = max_sharpe_portfolio['Sharpe Ratio']
max_sharpe_complete_portfolio['risk_free_weight'] = 1- y_star
max_sharpe_complete_portfolio['GLD weight'] = y_star * max_sharpe_portfolio['GLD weight']
max_sharpe_complete_portfolio['WMT weight'] = y_star * max_sharpe_portfolio['WMT weight']
max_sharpe_complete_portfolio['PG weight'] = y_star * max_sharpe_portfolio['PG weight']

print('Optimal Risky Portfolio:')
print(max_sharpe_portfolio)
print('\n')
print('Optimal Complete Portfolio:')
print(max_sharpe_complete_portfolio)
```
<img width="575" height="194" alt="image" src="https://github.com/user-attachments/assets/7a2ded0d-6a86-42ec-8677-fb2543bdd318" />

``` Python
import matplotlib.pyplot as plt

#weight, labels
weights = max_sharpe_complete_portfolio.loc[:, ['risk_free_weight', 'GLD weight', 'WMT weight', 'PG weight']].iloc[0]
labels = ['Risk-Free Asset', 'GLD', 'WMT', 'PG']
colors = ['#2CA6A4', '#FFD66B', '#F89FA1', '#7EC8E3']
explode = [0.05 if w == max(weights) else 0 for w in weights]

# pie chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(weights,
                                  labels=labels,
                                  colors=colors,
                                  explode=explode,
                                  startangle=120,
                                  autopct='%1.1f%%',
                                  pctdistance=0.8,
                                  labeldistance=1.1,
                                  shadow= False,
                                  textprops={'fontsize': 12},
                                  wedgeprops={'edgecolor': 'white'})

# make the label bold
for autotext in autotexts:
    autotext.set_weight('bold')

ax.legend(wedges, labels,
          title='Assets',
          loc='center left',
          bbox_to_anchor=(1, 0.5),
          fontsize=12,
          title_fontsize=13)

ax.set_title("Optimal Complete Portfolio Allocation", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```
<img width="787" height="559" alt="image" src="https://github.com/user-attachments/assets/89af3e46-6031-4526-a6f9-757f60ee8289" />

**Explanation:**
To diversify the portfolio, the objective is to find the best mix between the risky portfolio and the risk-free asset. In this analysis, the risk-free asset rate is derived from the U.S. Treasury bill (T-bill) which had a yield of 4.46% as of June 2, 2025. We assume the T-bill carries no risk, implying a standard deviation of zero. Based on the optimal proportion to invest in the risky portfolio (y*) of 1.132, investors should allocate 113.2% of their capital to the risky portfolio, borrowing the additional 13.2% at the risk-free rate to increase exposure. This strategy is justified by the very strong risk-adjusted return offered by the risky portfolio. The portfolio composition consists of 53.4% in gold, 45.2% in P&G, and 1.4% in Walmart.

## 6. CAL & Efficient Frontier
``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)
num_portfolios = 3000

# Extract expected return and volatility from the optimal portfolio row
exp_ret_p = max_sharpe_portfolio['Returns'].values[0]
std_dev_p = max_sharpe_portfolio['Volatility'].values[0]

# Empty lists to store weight in the risky portfolio
# as well as returns, volatility, and utility value for the complete portfolios
port_returns = []
port_volatility = []
port_utility = []
riskyport_weights = []
# We start with 0% investment in risky portfolio
weight_p = 0.000


# for loop to iterate through number of complete portfolios required; round each relevant value to 4 d.p.
for single_portfolio in range(1400, num_portfolios):
    weight_p = round(weight_p, 4)
    weight_rf = round((1-weight_p), 4)
    returns = round(weight_p*exp_ret_p + weight_rf*rf, 4)
    std_dev = round(weight_p*std_dev_p, 4)
    utility = round(returns - 0.5*A*std_dev*std_dev, 4)

# We increase the weight by 0.001 and append the resulting values to the respective lists each time
    weight_p = weight_p + 0.001
    port_returns.append(returns)
    port_volatility.append(std_dev)
    port_utility.append(utility)
    riskyport_weights.append(weight_p)

 # A dictionary for weight in risky portfolio, returns, risk, and Utility values of each complete portfolio
# every increase in risk (0.1%), how would it affect returns, volatility, utility!!!
portfolio = {'Weight_Risky': riskyport_weights,
             'Returns': port_returns,
             'Volatility': port_volatility,
             'Utility': port_utility}

# Make a nice dataframe
df1 = pd.DataFrame(portfolio)

print(df1)
```
<img width="389" height="247" alt="image" src="https://github.com/user-attachments/assets/53a0086f-be88-447e-92ec-c856b8b81c50" />

``` Python
# Plot the Capital Allocation Line
plt.figure(figsize=(10, 8))
plt.scatter(df['Volatility'], df['Returns'], s=10, alpha=0.5, label='Efficient Frontier')
plt.plot(df1['Volatility'], df1['Returns'],label='CAL', color='orange')
plt.scatter(x = max_sharpe_portfolio['Volatility'], y = max_sharpe_portfolio['Returns'], c = 'red', marker = 'o', s = 50,
            label='Tangency Portfolio (P)')

plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier with Capital Allocation Line')
plt.show();
```
<img width="855" height="695" alt="image" src="https://github.com/user-attachments/assets/be750b2b-9651-4460-86d2-67b29ebe77ec" />

## 7. Sensitivity Analysis
This analysis is to analyze how the utility level will change as the level of risk aversion increases.

``` Python
import numpy as np
import pandas as pd

# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

A_range = np.arange(1, 11)  # A从1到10

results = []
for A in A_range:
    y_star = (expected_return - rf) / (A * volatility ** 2)

    y_star = min(y_star, 1.0)
    # port_return、volatility、utility
    portfolio_return = rf + y_star * (expected_return - rf)
    portfolio_volatility = y_star * volatility
    utility = portfolio_return - 0.5 * A * (portfolio_volatility ** 2)
    results.append({
        'A': A,
        'Y*': y_star,
        'Risk Free Weight': 1-y_star,
        'Portfolio Return': portfolio_return,
        'Portfolio Volatility': portfolio_volatility,
        'Utility': utility
    })

df_sensitivity = pd.DataFrame(results)
print(df_sensitivity)
```
<img width="614" height="192" alt="image" src="https://github.com/user-attachments/assets/f322c871-6bf5-4d6d-8a35-27789e083e05" />

``` Python
styled_df = df_sensitivity.style\
    .format({
        'Y*': '{:.1%}',
        'Risk Free Weight': '{:.1%}',
        'Portfolio Return': '{:.2%}',
        'Portfolio Volatility': '{:.2%}',
        'Utility': '{:.3f}'
    })\
    .background_gradient(cmap='Blues', subset=['Utility'])\
    .set_caption("Table: Optimal Complete Portfolio Metrics for Different A")\
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])\
    .set_properties(**{'text-align': 'center'})

styled_df
```
<img width="630" height="364" alt="image" src="https://github.com/user-attachments/assets/52e53ab2-4617-44f9-9f9e-80e66a40241b" />

## 3 Assets Portfolio: Risky, Risk-Free, and Market Portfolio
We extended our diversified portfolio to a 3-asset allocation by including the market portfolio index, represented by the S&P 500 (^GSPC). Market data is sourced from Yahoo!Finance, for the period 31 May 2022 to 31 May 2025, consistent with the timeframe used for the other risky assets. Based on the calculated optimal proportion to invest in the risky portfolio (y*) of 1.141, investors should allocate 114.1% of their capital to the risky portfolio, borrowing the additional 14.1% at the risk-free rate to increase exposure. The portfolio is made up of 45.2% in gold, 0.7% in P&G, 48.5% in Walmart, and 5.6% in the S&P 500. The 3-asset portfolio provides enhanced diversification and a more comprehensive market representation, making it a more flexible and strategically balanced approach for long-term investors.

``` Python
#Daily simple returns with SP500
print(stocks_return)
daily_mean_return_D = stocks_return.aggregate(np.mean)
print(daily_mean_return_D)

daily_std = stocks_return.aggregate(np.std)

annual_simple_returns = daily_mean_return_D*252
annual_std = daily_std*np.sqrt(252)

daily_covariance = stocks_return.cov()
annual_covariance = daily_covariance*252
print(annual_covariance)

daily_correlation = daily_simple_returns.corr()
```
<img width="391" height="473" alt="image" src="https://github.com/user-attachments/assets/9fabe8a0-e7d1-463a-ad3e-801de47cbdfb" />

### Compute the Portfolio's Returns, Volatility, Sharpe Ratio and the Respective Weights
``` Python
# Create empty lists to store returns, volatility and weights of imaginary portfolios
port_returns_D = []
port_volatility_D = []
port_sharpe_D = []
stock_weights_D = []

# set the number of combinations (1,000) for imaginary portfolios and risk-free rate = 4.46%
num_assets = 4
num_portfolios = 1000
rf = 0.0446

for single_portfolio in range(num_portfolios):
    # Select random weights for portfolio holdings
    weights = np.random.random(num_assets)

    # Rebalance weights to sum to 1
    weights /= np.sum(weights)

    # Formula for returns and variance using matrix multiplication
    returns = np.dot(weights, annual_simple_returns)
    variance = np.dot(np.dot(weights, annual_covariance), weights.T)
    sharpe = (returns-rf)/np.sqrt(variance)
    std = np.sqrt(variance)

    # Appending the values to the respective lists
    port_returns_D.append(returns)
    port_volatility_D.append(std)
    port_sharpe_D.append(sharpe)
    stock_weights_D.append(weights)

    # Printing the weights of each stock and the corresponding expected annual return and risk of portfolio
    print('Weight in GLD, PG, WMT, S&P500:', weights)
    print('Expected annual return of portfolio:', returns)
    print('Annual volatility of portfolio:', std)
```
<img width="608" height="818" alt="image" src="https://github.com/user-attachments/assets/acf1e1bc-693d-4b73-955e-86cc3885c2f9" />

### Create a DataFrame
``` Python
ticker = ['GLD', 'PG','WMT', '^GSPC']
# A dictionary for Returns, Risk, Sharpe Ratio values of each portfolio
portfolio = {'Returns': port_returns_D,
             'Volatility': port_volatility_D,
             'Sharpe Ratio': port_sharpe_D}

# Extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(ticker):
    portfolio[symbol+' weight'] = [weight[counter] for weight in stock_weights_D]
#   print(counter, symbol)

# Make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# Get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' weight' for stock in ticker]
print(df)
```
<img width="605" height="474" alt="image" src="https://github.com/user-attachments/assets/3e85423b-a765-41c6-864f-13ce4d5fee09" />

### Minimum Variance Portfolio & Optimal Portfolio
``` Python
# Find minimum variance portfolio & optimal portfolio with max Sharpe ratio in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# Use the min, max values to locate and create the two special portfolios
min_variance_port = df.loc[df['Volatility'] == min_volatility]
max_sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]

print('Minimum Variance Portfolio:')
print(min_variance_port)
print('\n')
print('Optimal Portfolio:')
print(max_sharpe_portfolio)
```
<img width="608" height="257" alt="image" src="https://github.com/user-attachments/assets/a8a64d9d-f248-44b1-9250-b90c84c3b266" />

### Plotting the Resulting Frontier
``` Python
# Plot frontier, max sharpe & min Volatility values with a scatterplot
df.plot.scatter(x = 'Volatility', y = 'Returns', figsize = (10, 8))

plt.scatter(x = max_sharpe_portfolio['Volatility'], y = max_sharpe_portfolio['Returns'], c = 'red', marker = 'o', s = 200)
plt.scatter(x = min_variance_port['Volatility'], y = min_variance_port['Returns'], c = 'yellow', edgecolors = 'red', marker = '*', s = 300)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Resulting Frontier')
plt.show();
```
<img width="870" height="697" alt="image" src="https://github.com/user-attachments/assets/483cb836-5c37-4cfe-a8bc-858f682284cd" />

### Y maximum
``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

# Calculate y max
y_star = (expected_return - rf) / (A * (volatility ** 2))
print("Optimal proportion to invest in the risky portfolio (y*):", y_star)
```
<img width="593" height="26" alt="image" src="https://github.com/user-attachments/assets/466ce4d1-3a06-4289-a06f-8a29ba595773" />

``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

max_sharpe_complete_portfolio = pd.DataFrame()

y_star = min(y_star, 1.0)
max_sharpe_complete_portfolio['Returns'] = y_star * max_sharpe_portfolio['Returns'] + (1-y_star) * rf
max_sharpe_complete_portfolio['Volatility'] = y_star * max_sharpe_portfolio['Volatility']
max_sharpe_complete_portfolio['Sharpe Ratio'] = max_sharpe_portfolio['Sharpe Ratio']
max_sharpe_complete_portfolio['risk_free_weight'] = 1- y_star
max_sharpe_complete_portfolio['GLD weight'] = y_star * max_sharpe_portfolio['GLD weight']
max_sharpe_complete_portfolio['PG weight'] = y_star * max_sharpe_portfolio['PG weight']
max_sharpe_complete_portfolio['WMT weight'] = y_star * max_sharpe_portfolio['WMT weight']
max_sharpe_complete_portfolio['^GSPC weight'] = y_star * max_sharpe_portfolio['^GSPC weight']

print('Optimal Risky Portfolio wtih Market Portfolio:')
print(max_sharpe_portfolio)
print('\n')
print('Optimal Complete Portfolio with Market Portfolio:')
print(max_sharpe_complete_portfolio)
```
<img width="598" height="252" alt="image" src="https://github.com/user-attachments/assets/917c1185-b96d-4c52-9b97-08a1ce207b11" />

``` Python
import matplotlib.pyplot as plt

#weight, labels
weights = max_sharpe_complete_portfolio.loc[:, ['risk_free_weight', 'GLD weight', 'PG weight', 'WMT weight', '^GSPC weight']].iloc[0]
labels = ['Risk-Free Asset', 'GLD', 'PG', 'WMT', 'S&P 500']
colors = ['#2CA6A4', '#FFD66B', '#F89FA1', '#7EC8E3', '#C8B7E0']
explode = [0.05 if w == max(weights) else 0 for w in weights]

# pie chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(weights,
                                  labels=labels,
                                  colors=colors,
                                  explode=explode,
                                  startangle=120,
                                  autopct='%1.1f%%',
                                  pctdistance=0.8,
                                  labeldistance=1.1,
                                  shadow= False,
                                  textprops={'fontsize': 12},
                                  wedgeprops={'edgecolor': 'white'})

# make the label bold
for autotext in autotexts:
    autotext.set_weight('bold')

ax.legend(wedges, labels,
          title='Assets',
          loc='center left',
          bbox_to_anchor=(1, 0.5),
          fontsize=12,
          title_fontsize=13)

ax.set_title("Optimal Complete Portfolio Allocation", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```
<img width="785" height="568" alt="image" src="https://github.com/user-attachments/assets/b822c317-3640-4cb8-aa95-d59d020f19cb" />

``` Python
# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate
A = 10        # risk aversion coefficient (you can change this)
num_portfolios = 1800

# Extract expected return and volatility from the optimal portfolio row
exp_ret_p = max_sharpe_portfolio['Returns'].values[0]
std_dev_p = max_sharpe_portfolio['Volatility'].values[0]

# Empty lists to store weight in the risky portfolio
# as well as returns, volatility, and utility value for the complete portfolios
port_returns = []
port_volatility = []
port_utility = []
riskyport_weights = []
# We start with 0% investment in risky portfolio
weight_p = 0.000

# for loop to iterate through number of complete portfolios required; round each relevant value to 4 d.p.
for single_portfolio in range(1000, num_portfolios):
    weight_p = round(weight_p, 4)
    weight_rf = round((1-weight_p), 4)
    returns = round(weight_p*exp_ret_p + weight_rf*rf, 4)
    std_dev = round(weight_p*std_dev_p, 4)
    utility = round(returns - 0.5*A*std_dev*std_dev, 4)

# We increase the weight by 0.001 and append the resulting values to the respective lists each time
    weight_p = weight_p + 0.001
    port_returns.append(returns)
    port_volatility.append(std_dev)
    port_utility.append(utility)
    riskyport_weights.append(weight_p)

 # A dictionary for weight in risky portfolio, returns, risk, and Utility values of each complete portfolio
# every increase in risk (0.1%), how would it affect returns, volatility, utility!!!
portfolio = {'Weight_Risky': riskyport_weights,
             'Returns': port_returns,
             'Volatility': port_volatility,
             'Utility': port_utility}

# Make a nice dataframe
df1 = pd.DataFrame(portfolio)

print(df1)
```
<img width="375" height="252" alt="image" src="https://github.com/user-attachments/assets/ae6f772a-722e-469a-9088-161436cbc69d" />

### Plotting The Efficient Frontier and Capital Allocation Line
``` Python
# Plot the Capital Allocation Line
plt.figure(figsize=(10, 8))
plt.scatter(df['Volatility'], df['Returns'], s=10, alpha=0.5, label='Efficient Frontier')
plt.plot(df1['Volatility'], df1['Returns'],label='CAL', color='orange')
plt.scatter(x = max_sharpe_portfolio['Volatility'], y = max_sharpe_portfolio['Returns'], c = 'red', marker = 'o', s = 50,
            label='Tangency Portfolio (P)')

plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier with Capital Allocation Line')
plt.show();
```
<img width="861" height="702" alt="image" src="https://github.com/user-attachments/assets/167b3b4f-fdc6-4e9b-bda9-fd8889456e35" />

### Sensitivity Analysis
``` Python
import numpy as np
import pandas as pd

# Assume risk-free rate and risk aversion coefficient
rf = 0.0446  # risk-free rate

# Extract expected return and volatility from the optimal portfolio row
expected_return = max_sharpe_portfolio['Returns'].values[0]
volatility = max_sharpe_portfolio['Volatility'].values[0]

A_range = np.arange(1, 11)  # A从1到10

results = []
for A in A_range:
    y_star = (expected_return - rf) / (A * volatility ** 2)

    y_star = min(y_star, 1.0)
    # port_return、volatility、utility
    portfolio_return = rf + y_star * (expected_return - rf)
    portfolio_volatility = y_star * volatility
    utility = portfolio_return - 0.5 * A * (portfolio_volatility ** 2)
    results.append({
        'A': A,
        'Y*': y_star,
        'Risk Free Weight': 1-y_star,
        'Portfolio Return': portfolio_return,
        'Portfolio Volatility': portfolio_volatility,
        'Utility': utility
    })

df_sensitivity = pd.DataFrame(results)
print(df_sensitivity)
```
<img width="613" height="198" alt="image" src="https://github.com/user-attachments/assets/9d23a546-5244-481d-a3b3-7e49e3c7239c" />

``` Python
styled_df = df_sensitivity.style\
    .format({
        'Y*': '{:.1%}',
        'Risk Free Weight': '{:.1%}',
        'Portfolio Return': '{:.2%}',
        'Portfolio Volatility': '{:.2%}',
        'Utility': '{:.3f}'
    })\
    .background_gradient(cmap='Blues', subset=['Utility'])\
    .set_caption("Table: Optimal Complete Portfolio Metrics for Different A")\
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])\
    .set_properties(**{'text-align': 'center'})

styled_df
```
<img width="629" height="367" alt="image" src="https://github.com/user-attachments/assets/9fd819c8-c2e0-4fbd-81a0-38ab8b24bd35" />

**Explanation:**
The objective of this sensitivity analysis is to examine how the portfolio’s utility score responds to changes in the investor’s degree of risk aversion (A). The utility model determines the optimal allocation between risky assets, represented here by defensive stocks and risk-free assets, such as Treasury bonds. In general, a portfolio becomes more attractive to an investor as its expected return increases but less attractive as its level of risk rises. The utility score reflects the investor’s trade-off between risk and return. The utility score consistently decreases as the investor’s risk aversion (A) increases, reaching its highest value of 0.2441 when A = 1 and its lowest value of 0.1618 when A = 10. This pattern reflects that more risk-averse investors penalize risk more heavily and require higher returns to maintain the same level of utility. This outcome is consistent with theoretical expectations and reinforces the inverse relationship between risk aversion and portfolio attractiveness.

## Comparison Analysis: 2 Asset Allocation and 3 Asset Allocation
Both optimal portfolios – with or without the market index (S&P 500) deliver very similar performance, with expected returns of approximately 25.3% and volatility around 13.4%. The portfolio without the market index achieved a slightly higher Sharpe ratio of 1.56, indicating a marginally more efficient risk-return trade-off. However, the portfolio including the S&P 500 offers more exposure to the market and enhances diversification, allocating 5.6% to the S&P 500 while slightly reducing the weight in P&G stock. Although this results in a slightly lower Sharpe ratio of 1.54, the difference is minimal. In both cases, the complete portfolio allocates 100% to risky assets, reflecting the strong performance of the underlying mix.

## Limitations and Conclusions
While the findings of this study offered valuable insights for portfolio construction during periods of heightened economic uncertainty, specifically under the influence of ongoing tariff 
wars, several limitations were noted. 

First, the analysis was based on a small and specific selection of six stocks—three aggressive and three defensive stocks. While this allowed for a focused comparison, it may not fully reflect the diversity and risk-return characteristics of their respective sectors. Moreover, we assumed a constant risk-free rate of 4.46% based on the U.S. T-bill rates, which may not be realistic in actual market conditions due to varying borrowing constraints and credit risks.Additionally, portfolio optimization relied heavily on the Sharpe ratio. While this is a widely accepted risk-adjusted return metric, it overlooks other risk factors such as skewness or kurtosis in return distributions. As a result, the portfolio may appear optimal under the Sharpe ratio yet remain exposed to asymmetric risks or extreme outcomes that the Sharpe ratio fails to capture. The model also presumes investors behave rationally and only differ by degrees of risk aversion, which overlooks real-world constraints such as taxation, transaction costs, liquidity needs, and behavioral biases. Finally, the data used, spanning from 2022 to 2025, reflects the unique post-COVID tariff 
war environment, but this timeframe may not represent longer-term market conditions or different 
economic cycles. 

Despite these limitations, our findings indicated that defensive stocks provide superior riskadjusted returns under volatile market conditions. The optimal defensive portfolio for 2-asset allocation achieved a higher Sharpe ratio of 1.56 and lower volatility of 0.133 compared to the aggressive portfolio. When combined with the market index (S&P 500), the diversified portfolio maintained strong performance while enhancing exposure to broader market trends. Sensitivity analysis showed that increasing risk aversion leads to lower utility, reinforcing the need to tailor asset allocation based on individual risk preferences. To summarize, defensive stocks offer a resilient investment strategy during times of uncertainty, and the incorporation of strategic diversification with market indices and risk-free rates can further improve portfolio performance for risk-conscious investors.


