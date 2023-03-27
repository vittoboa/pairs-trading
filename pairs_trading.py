import os
import warnings
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from itertools import combinations, chain
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS


def get_stock_price(ticker):
    """ Use Nasdaq API to retrieve the closing prices of a stock.

    Parameters:
    ticker -- the symbol of the stock to retrieve
    """

    # Define the endpoint to retrieve the closing prices
    endpoint = f'https://data.nasdaq.com/api/v3/datasets/WIKI/{ticker}?column_index=4&order=asc'
    # Send a GET request to the endpoint
    response = requests.get(endpoint)

    # Print an error message if the request was unsuccessful
    if response.status_code != 200:
        print(f'Failed to retrieve the data form {ticker} with error code {response.status_code}')
        return dict()

    # Convert the response data to json format
    data = response.json()

    # Return the data as a dictionary with the closing_price for each day
    return dict(data['dataset']['data'])


def get_stocks_prices(tickers):
    """ Create a dataframe with the closing price of the stocks for each day.

    Parameters:
    tickers -- all symbols of the stocks to retrieve
    """

    # Create the dataframe with all closing prices
    prices = pd.concat(
        [pd.DataFrame.from_dict(get_stock_price(ticker), 'index', columns=[ticker]) for ticker in tickers],
        axis=1
    ).sort_index()

    # Store the prices in a csv file
    prices.to_csv('prices.csv')

    return prices


def select_pairs(prices):
    """
    Select pairs of stocks that are cointegrated using a combination of PCA and DBSCAN clustering.

    Parameters:
    prices -- stock prices where each column of the dataframe represents a different stock
    """

    # Normalize prices
    prices = (prices - prices.mean()) / prices.std()
    # Drop tickers with missing values
    prices = prices.dropna(axis='columns')

    # Apply PCA to get the principal components
    pca = PCA(n_components=5)
    pca.fit(prices)
    components = pca.components_

    # Apply DBSCAN to the principal components
    dbscan = DBSCAN(min_samples=4, eps=0.005)
    dbscan.fit(components.T)

    # Create a DataFrame with tickers and their corresponding clusters
    clusters = pd.DataFrame({'ticker': prices.columns, 'cluster': dbscan.labels_})

    # Filter out tickers that don't belong to any cluster
    clusters = clusters[clusters.cluster != -1]

    # Group tickers by cluster to generate pairs of tickers
    groups = clusters.groupby('cluster')
    pairs = chain.from_iterable(groups['ticker'].apply(combinations, 2))

    # Calculate the p-value of each pair of tickers using the cointegration test
    p_value = lambda pair: coint(prices[pair[0]], prices[pair[1]], maxlag=5)[1]

    # Filter out pairs with p-values above the significance level (0.05)
    selected_pairs = [pair for pair in pairs if p_value(pair) < 0.05]

    return selected_pairs


def pairs_trading(prices):
    """
    Implements a pairs trading strategy based on PCA and OLS regression, selecting pairs of
    stocks with similar price movements and trading their spread based on z-scores.

    Parameters:
    prices -- stock prices where each column of the dataframe represents a different stock
    """

    # Filter out tickers with many missing values
    non_na_pct = prices.count() / len(prices)
    prices = prices.loc[:, non_na_pct > 0.90]

    # Drop rows with missing values
    prices = prices.dropna()

    # Split data into training and test sets based on month of year
    train = prices[prices.index.month < 7]
    test = prices[prices.index.month >= 7]

    # Select pairs of cointegrated stocks
    pairs = select_pairs(train)
    pairs = np.array(pairs)
    pairs_a, pairs_b = pairs[:,0], pairs[:,1]

    # Compute the regression coefficients for each pair of stocks
    b = [OLS(train[b], train[a]).fit().params[a] for a, b in zip(pairs_a, pairs_b)]

    # Calculate the spread between each pair of stocks
    train_spreads = train[pairs_b] - b * train[pairs_a].values
    test_spreads = test[pairs_b] - b * test[pairs_a].values

    # Standardize the test set spreads using the mean and standard deviation of the training set spreads
    test_z_scores = (test_spreads - train_spreads.mean()) / train_spreads.std()

    # Set the threshold for entering a long or short position
    threshold = 1.0

    # Create a DataFrame of long and short positions
    long = (test_z_scores < -threshold).astype(int)
    short = (test_z_scores > threshold).astype(int)
    # Create a DataFrame of net positions
    positions = long - short

    # Calculate the returns of each stock in the pairs for the test set
    returns = test.pct_change().dropna()

    # Add a row of zeros to the end of the returns DataFrame to close all positions
    returns.loc[returns.index[-1] + pd.DateOffset(1)] = 0

    # Compute the returns of the portfolio for each stock in the pairs
    portfolio_a = positions * returns[pairs_a].values
    portfolio_b = positions * returns[pairs_b].values

    # Compute the returns of the overall portfolio
    portfolio = portfolio_b - portfolio_a

    # Compute the mean return of the portfolio for the test set
    mean_portfolio_returns = (portfolio + 1.0).cumprod().iloc[-1].mean()
    print(mean_portfolio_returns)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Load the list of tickers
    tickers = pd.read_csv('tickers.csv')['tickers']

    # Load the prices from file or download them
    prices = pd.read_csv('prices.csv', index_col=0) if os.path.exists('prices.csv') else get_stocks_prices(tickers)

    # Convert the date to datetime format
    prices.index = pd.to_datetime(prices.index)

    # Select prices from 2001 to 2017
    prices = prices.loc[(prices.index.year >= 2001) & (prices.index.year <= 2017)]

    # Apply pairs trading to each year
    prices.groupby(prices.index.year).apply(pairs_trading)
