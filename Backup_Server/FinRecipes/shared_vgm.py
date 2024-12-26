import os
import pandas as pd
import progressbar
from time import sleep
import datetime as dt
from datetime import date
from datetime import timedelta
import yfinance as yf
import numpy as np
import matplotlib.dates as mdates

import math
from scipy.stats import zscore
import logging

import empyrical as ep
import progressbar
import time

import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas_ta as ta
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

import warnings
import requests
import concurrent.futures
import os
import pickle
import time
import statistics

from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import datetime as dt
from datetime import timedelta
import scipy.stats as stats

from concurrent.futures import ThreadPoolExecutor, as_completed


import sys
sys.path.append('FinRecipes/')
from config_po_vgm import (
    COUNTRY,
    ROOT_DIR,
    MOM_VOLATILITY_PERIOD,
    RISK_VOLATILITY_PERIOD,
    MOMENTUM_PERIOD_LIST,
    RISK_VOLATILITY_PERIOD,
    RISK_PERIOD_LIST,
    INDUSTRY_REGRESSION_DIR,
    INDUSTRY_DATA_FILE,
    CAT1_RATIOS,
    CAT2_RATIOS,
    SAVE_EXCEL,
    PATH_DATA,
    VALUE_METRICES,
    GROWTH_METRICES,
    SAVE_INDUSTRIAL_MOMENTUM_SCORE,
    SCALING_METHOD,
)

warnings.filterwarnings('ignore')

def send_email(name, email, message, formspree_email):

    # Construct the form data
    form_data = {
        'name': name,
        'email': email,
        'message': message
    }

    # Send POST request to formspree.io
    response = requests.post(f'https://formspree.io/{formspree_email}', data=form_data)

    # Check if the request was successful
    if response.status_code == 200:
        print("Form submission successful!")
    else:
        print("Form submission failed.")

def save_dataframe(df, filename, save_as = 'h5'):
    filename = filename + '.' + save_as

    if save_as == 'h5':
        df.to_hdf(filename,"df", mode = 'w')

    elif save_as == 'parquet':
        df.to_parquet(filename)

    elif save_as == 'csv':
        df.to_csv(filename, index = False)

    elif save_as == 'excel':
        df.to_excel(filename, index = False)

def weekday_dates_list(start_date, end_date, weekday = 4):
    date = start_date

    date_list = []
    while date <= end_date:
        if date.weekday() == weekday:  # Check if the current date is a Friday (0=Monday, 1=Tuesday, ..., 6=Sunday)
            # yield date
            date_list.append(date.date())
        date += pd.DateOffset(days=1)  # Move to the next day

    return date_list

def computation_time(start, message):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{message} {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
    return hours, minutes, seconds

def check_and_make_directories(directories: list):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

def first_day_month_list(start_date, end_date):
    date = start_date.replace(day=1)  # Set the day to 1st of the start month
    date_list = []
    while date <= end_date:
        # yield date
        date_list.append(date.date())
        date += pd.DateOffset(months=1)  # Move to the 1st day of the next month

    return date_list

class Load_n_Preprocess:
    def __init__(self, tickers_list, start_date, end_date, feature_lookback = 0, drl_lookback = 0, path_daily_data = None, path_intraday_data=None):
        if isinstance(tickers_list, list):
            self.tickers_list = tickers_list
        elif isinstance(tickers_list, str):
            self.tickers_list = [tickers_list]
        else:
            print("Not a valid list")
        
        self.removed_tickers_list = []
        self.start_date = pd.to_datetime(start_date)
        self.feature_lookback = feature_lookback
        self.drl_lookback = drl_lookback
        self.end_date = pd.to_datetime(end_date)
        self.path_daily_data = path_daily_data
        self.path_intraday_data = path_intraday_data

        self.market_open = pd.Timestamp('09:30:00').time()              # Regular market open time
        self.market_close = pd.Timestamp('15:30:00').time()             # Regular market close time

    # def download_yfinance(self, is_live=True, proxy=None) -> pd.DataFrame:
    #     """Fetches data from Yahoo API
    #     Parameters
    #     ----------
    #     Returns
    #     -------
    #     `pd.DataFrame`
    #         7 columns: A date, open, high, low, close, volume and tick symbol
    #         for the specified stock ticker
    #     """
    #     # Function to download data for a single ticker
    #     def fetch_data(tic):
    #         if not ('.NS' in tic or '.BO' in tic):
    #             if '.' in tic:
    #                 tic = tic.replace('.', '-')
            
    #         if is_live:
    #             return yf.download(tic, start=self.start_date, proxy=proxy)
    #         else:
    #             return yf.download(tic, start=self.start_date, end=self.end_date, proxy=proxy)

    #     data_df = pd.DataFrame()
    #     num_failures = 0

    #     with ThreadPoolExecutor() as executor:
    #         future_to_ticker = {executor.submit(fetch_data, tic): tic for tic in self.tickers_list}
    #         for future in as_completed(future_to_ticker):
    #             tic = future_to_ticker[future]
    #             try:
    #                 temp_df = future.result()
    #                 temp_df["tic"] = tic
    #                 if len(temp_df) > 0:
    #                     data_df = pd.concat([data_df, temp_df], axis=0)
    #                 else:
    #                     num_failures += 1
    #             except Exception as e:
    #                 print(f"Error downloading {tic}: {e}")
    #                 num_failures += 1

    #     if num_failures == len(self.tickers_list):
    #         raise ValueError("No data is fetched.")

    #     # Reset the index, we want to use numbers as index instead of dates
    #     data_df = data_df.reset_index()
    #     try:
    #         # Convert the column names to standardized names
    #         data_df.columns = [
    #             "date",
    #             "open",
    #             "high",
    #             "low",
    #             "close",
    #             "adjcp",
    #             "volume",
    #             "tic",
    #         ]
    #         # Drop the adjusted close price column
    #         data_df = data_df.drop(labels="adjcp", axis=1)
    #     except NotImplementedError:
    #         print("The features are not supported currently")

    #     # Create day of the week column (Monday = 0)
    #     data_df["day"] = data_df["date"].dt.dayofweek
    #     # Convert date to standard string format, easy to filter
    #     data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    #     # Drop missing data
    #     data_df = data_df.dropna()
    #     data_df = data_df.reset_index(drop=True)
    #     print("Shape of DataFrame: ", data_df.shape)
    #     print(f"Number of Tickers: {len(data_df['tic'].unique())}")

    #     data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    #     data_df['date'] = pd.to_datetime(data_df['date'])

    #     return data_df


    def download_yfinance(self, is_live = True, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.tickers_list:
            tic = str(tic)
            if not ('.NS' in tic or '.BO' in tic):
                if '.' in tic:
                    tic = tic.replace('.', '-')

            if is_live:
                temp_df = yf.download(tic, start=self.start_date, proxy=proxy)
            else:
                temp_df = yf.download(tic, start=self.start_date, end=self.end_date, proxy=proxy)
           
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.tickers_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            # data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        print(f"Number of Tickers: {len(data_df['tic'].unique())}")
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        data_df['date'] = pd.to_datetime(data_df['date'])

        return data_df

    def load_daily_data(self):
        file_ext = os.path.splitext(self.path_daily_data)[-1]
        if file_ext == '.csv':
            df_tics_daily = pd.read_csv(self.path_daily_data)
        elif file_ext == '.h5':
            df_tics_daily = pd.read_hdf(self.path_daily_data, "df",mode = 'r')

        ######## this part is written for temproary  ###########################
        # df_tics['tic'] = TICKERS_LIST[0]
        # df_tics = df_tics[['Date','tic','Open','High','Low','Close']]
        # df_tics = df_tics.rename(columns = {'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
        # df_tics['tic'] = TICKERS_LIST[0]
        # df_tics = df_tics[['time','tic','open','high','low','close']]
        # df_tics = df_tics.rename(columns = {'time':'date'})
        ########################################################################
        if len(self.tickers_list) == 0:
            self.tickers_list = list(df_tics_daily['tic'].unique())
        else:
            df_tics_daily = df_tics_daily.loc[df_tics_daily['tic'].isin(self.tickers_list)]
        
        # df_tics_daily['date'] = pd.to_datetime(df_tics_daily['date']).dt.date               #,utc=True)              # convert date column to datetime
        # df_tics_daily['date'] = pd.to_datetime(df_tics_daily['date'])
        # df_tics_daily = df_tics_daily.sort_values(by=['date', 'tic'],ignore_index=True)
        # df_tics_daily.index = df_tics_daily['date'].factorize()[0]

        # start_date_idx = df_tics_daily[(df_tics_daily['date'] >= self.start_date)].index[0] - (self.drl_lookback + self.feature_lookback)
        # end_date_idx = df_tics_daily[(df_tics_daily['date'] <= self.end_date)].index[-1] 
        # df_tics_daily = df_tics_daily.loc[start_date_idx:end_date_idx]
        # df_tics_daily = df_tics_daily.reset_index(drop=True)
        # df_tics_daily.index = df_tics_daily['date'].factorize()[0]

        df_tics_daily['date'] = pd.to_datetime(df_tics_daily['date'])
        df_tics_daily['date'] = df_tics_daily['date'].dt.floor('D')
        df_tics_daily = df_tics_daily[(df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] <= self.end_date)]
        df_tics_daily.index = df_tics_daily['date'].factorize()[0]

        return df_tics_daily

    # def clean_daily_data(self, df, missing_values_allowed = 0.01, print_missing_values = False):
    #     uniqueDates = df['date'].unique()
    #     # if len(self.tickers_list) == 0:
    #     #     self.tickers_list = df['tic'].unique()
    #     # print("===================================================")
    #     # print(f'Number of Unique dates in between {self.start_date.date()} and {self.end_date.date()} is {len(uniqueDates)}')
    #     # print("===================================================")
    #     df_dates = pd.DataFrame(uniqueDates, columns=['date'])

    #     df_tics_daily_list = []
    #     updated_tickers_list = []
    #     for i, tic in enumerate(self.tickers_list):
    #         df_tic = df[df['tic'] == tic]
    #         df_tic = df_dates.merge(df_tic, on='date', how='left')
    #         df_tic['tic'] = tic

    #         if print_missing_values == True:
    #             # print('No. of missing values for', tic, ' =',df_tic['close'].isna().sum())
    #             # print("No. of missing values for %5s = %5d"%(tic,df_tic['close'].isna().sum()))
    #             print("No. of missing values before imputation for %5s = %5d"%(tic,df_tic['close'].isna().sum()))

    #         if df_tic['close'].isna().sum() <= missing_values_allowed * len(df_dates): 
    #             # df_tic = df_tic.fillna(method='ffill').fillna(method='bfill')   # can be commented.
    #             df_tic = df_tic.ffill().bfill()   
    #             df_tics_daily_list.append(df_tic)
    #             updated_tickers_list.append(tic)
    #         else:
    #             # print(tic,' is removed based on missing values')
    #             self.removed_tickers_list.append(tic)

    #     self.tickers_list = updated_tickers_list
    #     df_tics_daily = pd.concat(df_tics_daily_list)
    #     df_tics_daily = df_tics_daily.sort_values(by=['date', 'tic'],ignore_index=True)
    #     df_tics_daily.index = df_tics_daily.date.factorize()[0]
    #     print(f"{self.removed_tickers_list} are removed due to missing data.")\
        
    #     start_date_idx = df_tics_daily[(df_tics_daily['date'] >= self.start_date)].index[0]
    #     end_date_idx = df_tics_daily[(df_tics_daily['date'] <= self.end_date)].index[-1] 
    #     df_tics_daily = df_tics_daily.reset_index(drop=True)
    #     df_tics_daily.index = df_tics_daily['date'].factorize()[0]

    #     if len(df_tics_daily[df_tics_daily.duplicated(keep=False)]) == 0:
    #         print('Duplicate test: there is no duplicate rows.')

    #     return df_tics_daily

    def clean_daily_data(self, df, missing_values_allowed = 0.01, print_missing_values = False):
        uniqueDates = df['date'].unique()
        df_dates = pd.DataFrame(uniqueDates, columns=['date'])
        # if len(self.tickers_list) == 0:
        #     self.tickers_list = df['tic'].unique()
        # print("===================================================")
        # print(f'Number of Unique dates in between {self.start_date.date()} and {self.end_date.date()} is {len(uniqueDates)}')
        # print("===================================================")
        # df_dates = pd.DataFrame(uniqueDates, columns=['date'])

        # Merge the dates with the entire DataFrame at once
        df_merged = df_dates.merge(df, on='date', how='left')

        # Create a pivot table to align all tickers' data by date
        df_pivot = df_merged.pivot(index='date', columns='tic')

        # Count missing values for each ticker
        missing_values_count = df_pivot['close'].isna().sum()

        # Filter tickers based on allowed missing values
        valid_tickers = missing_values_count[missing_values_count <= missing_values_allowed * len(df_dates)].index
        self.removed_tickers_list = list(missing_values_count[missing_values_count > missing_values_allowed * len(df_dates)].index)

        # Keep only valid tickers and fill missing data
        # valid_columns = pd.MultiIndex.from_product([['open','high','close'], valid_tickers])
        # df_pivot = df_pivot.loc[:, valid_columns]
        # df_filled = df_pivot.ffill().bfill()
        
        # Keep only valid tickers and fill missing data for all columns
        df_pivot = df_pivot.loc[:, pd.IndexSlice[:, valid_tickers]]
        df_filled = df_pivot.ffill().bfill()

        # # Convert pivoted DataFrame back to long format
        # df_tics_daily = df_filled.stack().reset_index()

        # Convert pivoted DataFrame back to long format
        df_tics_daily = df_filled.stack(level='tic').reset_index()

        if print_missing_values:
            for tic in valid_tickers:
                print("No. of missing values before imputation for %5s = %5d" % (tic, missing_values_count[tic]))

        df_tics_daily = df_tics_daily.sort_values(by=['date', 'tic'], ignore_index=True)
        df_tics_daily.index = df_tics_daily.date.factorize()[0]
        # print(f"{self.removed_tickers_list} are removed due to missing data.")

        # # Filter rows based on the date range
        # df_tics_daily = df_tics_daily[
        #     (df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] <= self.end_date)
        # ]

        # df_tics_daily = df_tics_daily.reset_index(drop=True)
        # df_tics_daily.index = df_tics_daily['date'].factorize()[0]

        if len(df_tics_daily[df_tics_daily.duplicated(keep=False)]) != 0:
            print('Duplicate found')

        return df_tics_daily


def buy_hold_portfolio_return(df_tics, df_indices = None, with_indices = False,dropna = True,fillna = False):

    ''' Created on Date: 2023-11-07
    This function return the porfolio return of equal weight portfolio
    also compare the returns with indices'''

    n_tickers = len(df_tics.tic.unique())
    df_tics_close = pd.pivot_table(df_tics, values='close',index='date',columns='tic')
    if dropna:
        df_tics_returns = df_tics_close.pct_change().dropna()
    if fillna:
        df_tics_returns = df_tics_close.pct_change().fillna(0)
    equal_weights = np.full(n_tickers, 1 / n_tickers)
    portfolio_returns = df_tics_returns.dot(equal_weights)
    portfolio_returns = pd.DataFrame(portfolio_returns,columns=['Buy_Hold_returns'])

    if with_indices:
        df_index_close = pd.pivot_table(df_indices, values='close',index='date',columns='tic')
        if dropna:
            df_index_returns = df_index_close.pct_change().dropna()
        if fillna:
            df_index_returns = df_index_close.pct_change().fillna(0)

        # df_index_returns = df_index_close.pct_change()    #.fillna(0)
        portfolio_returns = portfolio_returns.merge(df_index_returns,how='left',on = 'date')
        # cumulative_returns = (portfolio_returns + 1).cumprod()
    
    return portfolio_returns

def plot_returns_drawdown(df_returns, tickers_list=[], filename='results/Strategy', period='daily', name_stock_world=None, pos='lower right'):
    df_returns['date'] = mdates.date2num(df_returns.index)
    # df_returns = df_returns.rename(columns={"^GSPC": "S&P 500", "^NDX": "NASDAQ 100"})

    # Create a figure with space for the table, cumulative returns, and drawdown plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # Adjust the cumulative return axis (ax1)
    ax1.set_position([0.1, 0.55, 0.8, 0.35])  # Moving this up a bit for space

    # Adjust the drawdown axis (ax2)
    ax2.set_position([0.1, 0.1, 0.8, 0.4])

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list[:-1]:  # Exclude the last column if it's a date
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{max_drawdown:.2f}"])

    # Create a new axis for the table above both plots
    ax_table = fig.add_subplot(111, frame_on=False)
    ax_table.xaxis.set_visible(False)
    ax_table.yaxis.set_visible(False)

    # Create the table
    table = plt.table(cellText=data,
                      colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                      cellLoc='center', loc='top', bbox=[0, 1.02, 1, 0.2],  # Adjusted position
                      colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])

    # Adjust column alignment for numeric values
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # First column (labels)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center')

    table.auto_set_font_size(False)
    table.set_fontsize(18)  # Increase font size of the table

    # Plotting Cumulative Returns
    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [2, 1.5, 1.5, 1.5, 1.5]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    for idx in np.arange(len(df_returns.columns.to_list()) - 1):
        ax1.plot(df_returns['date'].values, 100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1).values,
                color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx], label=f'{df_returns.columns.to_list()[idx]} Returns')


    ax1.set_ylabel("Cumulative Return (%)", fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=16)



    # Plotting Drawdowns
    for idx in np.arange(len(df_returns.columns.to_list()) - 1):
        cum_return = (df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod()
        drawdown = -(1 - cum_return / cum_return.cummax()) * 100
        ax2.fill_between(df_returns['date'], drawdown, color=color_val[idx], alpha=0.3, label=f'{df_returns.columns.to_list()[idx]} Drawdown')

    ax2.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('Drawdown (%)', fontsize=16)
    ax2.legend(loc=pos,fontsize=16)
    ax2.xaxis_date()
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(linestyle='dotted', linewidth=1)

    # # Adding title
    # suptitle_text = ""
    # if name_stock_world is not None:
    #     if len(tickers_list) == 0:
    #         suptitle_text += "Stock world: " + name_stock_world + '\n'


        # Add suptitle if tickers_list is empty or not
    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n'
    else:
        suptitle_text = ""
    
    if len(tickers_list) != 0:
        suptitle_text += str(tickers_list)

    plt.suptitle(suptitle_text, fontsize=16)

    # Adjust the plot to make room for the table and title
    # plt.subplots_adjust(top=0.83)  # Adjust to create space at the top for the table
    plt.subplots_adjust(top=0.83, hspace=0.15)

    # Move the title higher above the table
    plt.suptitle(suptitle_text, fontsize=20, y=1.02)  # Adjust y for title position

    # Save the figure
    plt.savefig(filename + '.jpeg', bbox_inches='tight')
    plt.close()

def plot_returns_drawdown_v2(df_returns, tickers_list=[], filename='results/Strategy', period='daily', name_stock_world=None, pos='lower right'):
    df_returns['date'] = mdates.date2num(df_returns.index)
    # df_returns = df_returns.rename(columns={"^GSPC": "S&P 500", "^NDX": "NASDAQ 100"})

    # Create a figure with space for the table, cumulative returns, and drawdown plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Adjust the cumulative return axis (ax1)
    ax1.set_position([0.1, 0.55, 0.8, 0.35])  # Moving this up a bit for space
    # Adjust the drawdown axis (ax2)
    ax2.set_position([0.1, 0.1, 0.4, 0.4])

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list[:-1]:  # Exclude the last column if it's a date
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        sortino_ratio = ep.sortino_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])
        std_dev = 100*df_returns[col].std()  # Standard deviation as a percentage

        # Append each row to data
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", 
                     f"{max_drawdown:.2f}", f"{std_dev:.2f}"])

    # Create a new axis for the table above both plots
    ax_table = fig.add_subplot(111, frame_on=False)
    ax_table.xaxis.set_visible(False)
    ax_table.yaxis.set_visible(False)

    # Create the table
    table = plt.table(cellText=data,
                      colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Std Dev (%)'],
                      cellLoc='center', loc='top', bbox=[0, 1.025, 1, 0.2],  # Adjusted position
                      colWidths=[0.185, 0.115, 0.115, 0.115, 0.115, 0.16, 0.115])

    # Adjust column alignment for numeric values
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # First column (labels)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center')

    table.auto_set_font_size(False)
    table.set_fontsize(18)  # Increase font size of the table

    # Plotting Cumulative Returns
    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [2, 1.5, 1.5, 1.5, 1.5]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    for idx in np.arange(len(df_returns.columns.to_list()) - 1):
        ax1.plot(df_returns['date'].values, 100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1).values,
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx], label=f'{df_returns.columns.to_list()[idx]} Returns')

    ax1.set_ylabel("Cumulative Return (%)", fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=16)

    # Plotting Drawdowns
    for idx in np.arange(len(df_returns.columns.to_list()) - 1):
        cum_return = (df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod()
        drawdown = -(1 - cum_return / cum_return.cummax()) * 100
        ax2.fill_between(df_returns['date'], drawdown, color=color_val[idx], alpha=0.3, label=f'{df_returns.columns.to_list()[idx]} Drawdown')

    ax2.set_xlabel('Date', fontsize=16)
    ax2.set_ylabel('Drawdown (%)', fontsize=16)
    ax2.legend(loc=pos, fontsize=16)
    ax2.xaxis_date()
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(linestyle='dotted', linewidth=1)

    plt.subplots_adjust(top=0.83, hspace=0.05)

    # # Add suptitle based on tickers_list
    # suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n' if len(tickers_list) == 0 else ""
    # suptitle_text += str(tickers_list) if len(tickers_list) != 0 else ""

    # # Move the title higher above the table
    # plt.suptitle(suptitle_text, fontsize=20, y=1.02)  # Adjust y for title position

    # Save the figure
    plt.savefig(filename + '.jpeg', bbox_inches='tight')
    plt.close()



def plot_returns_range(df_returns, tickers_list=[], filename='results/Strategy', period='daily', 
                 name_stock_world=None, start_date=None, end_date=None):
    # Filter the DataFrame for the given date range
    # if start_date is not None and end_date is not None:
    #     df_returns = df_returns.loc[start_date:end_date]

    if start_date is not None and end_date is not None:
        # Filter the DataFrame for the specified date range
        df_returns = df_returns.loc[start_date:end_date]

        # Append a row for the day before start_date with return 0
        previous_day = pd.to_datetime(start_date) - pd.DateOffset(days=1)
        
        # Create a Series filled with 0s for each column, with the index matching df_returns
        zero_return_series = pd.Series([0] * len(df_returns.columns), index=df_returns.columns)
        
        # Concatenate the new row to the DataFrame
        df_returns = pd.concat([df_returns, zero_return_series.rename(previous_day).to_frame().T])

        # Re-sort the DataFrame by index (date)
        df_returns = df_returns.sort_index()


    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(21, 10), dpi=120)

    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [3, 2, 2, 2, 2]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    # Plot the cumulative returns
    for idx in np.arange(len(df_returns.columns.to_list())):
        ax1.plot(100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1),
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx])

    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.xaxis_date()
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=20)

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list:
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])

        # Append each row to data
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{max_drawdown:.2f}"])

    # Create a table above the plot
    ax_table = plt.table(cellText=data, colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                         cellLoc='center', loc='top', bbox=[0, 1.02, 1, 0.25], colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])

    # Adjust column alignment (cell by cell)
    for (i, j), cell in ax_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # First column (labels)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center') 

    ax_table.auto_set_font_size(False)
    ax_table.set_fontsize(18)  # Increase the font size of the table

    # Adjust the plot to make room for the table and title
    plt.subplots_adjust(top=0.83)

    # Add suptitle if tickers_list is empty or not
    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n'
    else:
        suptitle_text = ""

    if len(tickers_list) != 0:
        suptitle_text += str(tickers_list)

    # Adjust suptitle to be above the table
    plt.suptitle(suptitle_text, fontsize=20, y=1.06)

    # Save and show the figure
    plt.savefig(filename + f'_{start_date}_{end_date}.jpeg', bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_returns_range_v2(df_returns, tickers_list=[], filename='results/Strategy', period='daily', 
                       name_stock_world=None, start_date=None, end_date=None):
    # Filter the DataFrame for the specified date range
    if start_date is not None and end_date is not None:
        df_returns = df_returns.loc[start_date:end_date]

        # Append a row for the day before start_date with return 0
        previous_day = pd.to_datetime(start_date) - pd.DateOffset(days=1)
        
        # Create a Series filled with 0s for each column, with the index matching df_returns
        zero_return_series = pd.Series([0] * len(df_returns.columns), index=df_returns.columns)
        
        # Concatenate the new row to the DataFrame
        df_returns = pd.concat([df_returns, zero_return_series.rename(previous_day).to_frame().T])

        # Re-sort the DataFrame by index (date)
        df_returns = df_returns.sort_index()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(21, 10), dpi=120)

    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [3, 2, 2, 2, 2]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    # Plot the cumulative returns
    for idx in np.arange(len(df_returns.columns.to_list())):
        ax1.plot(100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1),
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx])

    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.xaxis_date()
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=20)

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list:
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        sortino_ratio = ep.sortino_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])
        std_dev = 100 * df_returns[col].std()  # Standard deviation in percentage terms

        # Append each row to data
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}",
                     f"{max_drawdown:.2f}", f"{std_dev:.2f}"])

    # Create a table above the plot
    ax_table = plt.table(cellText=data, 
                         colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Std Dev (%)'],
                         cellLoc='center', loc='top', bbox=[0, 1.02, 1, 0.25], colWidths=[0.185, 0.115, 0.115, 0.115, 0.115, 0.16, 0.115])

    # Adjust column alignment (cell by cell)
    for (i, j), cell in ax_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # First column (labels)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center') 

    ax_table.auto_set_font_size(False)
    ax_table.set_fontsize(18)  # Increase the font size of the table

    # Adjust the plot to make room for the table and title
    plt.subplots_adjust(top=0.83)

    # Add suptitle if tickers_list is empty or not
    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n'
    else:
        suptitle_text = ""

    if len(tickers_list) != 0:
        suptitle_text += str(tickers_list)

    # Adjust suptitle to be above the table
    plt.suptitle(suptitle_text, fontsize=20, y=1.06)

    # Save and show the figure
    plt.savefig(filename + f'_{start_date}_{end_date}.jpeg', bbox_inches='tight')
    # plt.show()
    plt.close()



def plot_returns(df_returns, tickers_list=[], filename='results/Strategy', period='daily', name_stock_world=None):
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(21, 10), dpi=120)

    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [3, 2, 2, 2, 2]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    # Plot the cumulative returns
    for idx in np.arange(len(df_returns.columns.to_list())):
        ax1.plot(100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1),
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx])

    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.xaxis_date()
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=20)


  

    # # Format the x-axis to display only the year
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # plt.gcf().autofmt_xdate()

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list:
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])

        # Append each row to data
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{max_drawdown:.2f}"])

    # Create a table above the plot
    ax_table = plt.table(cellText=data, colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                         cellLoc='center', loc='top', bbox=[0, 1.02, 1, 0.25], colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])

    # Adjust column alignment (cell by cell)
    for (i, j), cell in ax_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # Second column (numeric values)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center') 

    ax_table.auto_set_font_size(False)
    ax_table.set_fontsize(18)  # Increase the font size of the table

    # Adjust the plot to make room for the table and title
    plt.subplots_adjust(top=0.83)

    # Add suptitle if tickers_list is empty or not
    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n'
    else:
        suptitle_text = ""

    if len(tickers_list) != 0:
        suptitle_text += str(tickers_list)

    # Adjust suptitle to be above the table
    plt.suptitle(suptitle_text, fontsize=20, y=1.06)

    # Save and show the figure
    plt.savefig(filename + '.jpeg', bbox_inches='tight')
    # plt.show()
    plt.close()

# Example call to function (assuming df_returns, ep, and other variables are defined)
# plot_returns(df_returns, tickers_list=['AAPL', 'GOOG'], filename='results/Strategy', period='daily', name_stock_world='Global Stocks')

def plot_returns_v2(df_returns, tickers_list=[], filename='results/Strategy', period='daily', name_stock_world=None):
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(21, 10), dpi=120)

    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [3, 2, 2, 2, 2]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']

    # Plot the cumulative returns
    for idx in np.arange(len(df_returns.columns.to_list())):
        ax1.plot(100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1),
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx])

    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.xaxis_date()
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=20)

    # Collect data for the table
    data = []
    column_list = df_returns.columns.to_list()
    for col in column_list:
        final_return = 100 * ep.cum_returns_final(df_returns[col])
        cagr_value = 100 * ep.cagr(df_returns[col], period=period)
        sharpe_ratio = ep.sharpe_ratio(df_returns[col], period=period)
        sortino_ratio = ep.sortino_ratio(df_returns[col], period=period)
        max_drawdown = -100 * ep.max_drawdown(df_returns[col])
        std_dev = 100 * df_returns[col].std()  # Standard deviation as a percentage

        # Append each row to data
        data.append([col, f"{final_return:.2f}", f"{cagr_value:.2f}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", 
                     f"{max_drawdown:.2f}", f"{std_dev:.2f}"])

    # Create a table above the plot
    ax_table = plt.table(cellText=data,
                         colLabels=[' ', 'Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Std Dev (%)'],
                         cellLoc='center', loc='top', bbox=[0, 1.02, 1, 0.25], colWidths=[0.185, 0.115, 0.115, 0.115, 0.115, 0.16, 0.115])

    # Adjust column alignment (cell by cell)
    for (i, j), cell in ax_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(horizontalalignment='center')
        else:
            if j == 0:  # First column (labels)
                cell.set_text_props(horizontalalignment='left')
            else:
                cell.set_text_props(horizontalalignment='center')

    ax_table.auto_set_font_size(False)
    ax_table.set_fontsize(18)  # Increase the font size of the table

    # Adjust the plot to make room for the table and title
    plt.subplots_adjust(top=0.83)

    # Add suptitle if tickers_list is empty or not
    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + (name_stock_world if name_stock_world else "") + '\n'
    else:
        suptitle_text = ""

    if len(tickers_list) != 0:
        suptitle_text += str(tickers_list)

    # Adjust suptitle to be above the table
    plt.suptitle(suptitle_text, fontsize=20, y=1.06)

    # Save and show the figure
    plt.savefig(filename + '.jpeg', bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_returns_US(df_returns, tickers_list = [], filename = 'results/Strategy', period = 'daily', name_stock_world = None):
    
    plt.figure(figsize=(21,10), dpi=120)
    linestyle_val = ['-','-.','-.','-.','-.']
    linewidth_val = [3,1.5,1.5,1.5,1.5]
    color_val = ['royalblue','red','darkorange','green','purple']
    plt.figure(figsize=(21,10), dpi=120)
    for idx in np.arange(len(df_returns.columns.to_list())):
        plt.plot(100*((df_returns[df_returns.columns.to_list()[idx]]+1).cumprod()-1),color = color_val[idx],linestyle = linestyle_val[idx],linewidth=linewidth_val[idx])
       
    plt.gcf().autofmt_xdate()
    # plt.xlabel("Date",fontsize=20)
    plt.legend(df_returns.columns.to_list(),fontsize=20)
    plt.ylabel("Cumulative Return (%)",fontsize=20)
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)

    if len(tickers_list) == 0:
        suptitle_text = "Stock world: " + name_stock_world +'\n'
    else:
        suptitle_text = ""

    column_list = df_returns.columns.to_list()
    for i in range(len( column_list )):
        # suptitle_text = suptitle_text + column_list[i] + "    CAGR: %.2f" % (100*ep.cagr(df_returns[df_returns.columns.to_list()[i]],period=period)) + ' ~ ' +\
        #                 "Sharpe Ratio: %.2f" % ep.sharpe_ratio(df_returns[df_returns.columns.to_list()[i]],period=period) + ' ~ ' +\
        #                 "Max Drawdown: %.2f" % (-100*ep.max_drawdown(df_returns[df_returns.columns.to_list()[i]]))+ "\n"
        suptitle_text = suptitle_text + column_list[i] + "    Return: %.2f" % (100 * ep.cum_returns_final(df_returns[df_returns.columns.to_list()[i]])) + ' ~ ' + \
                "CAGR: %.2f" % (100 * ep.cagr(df_returns[df_returns.columns.to_list()[i]], period=period)) + ' ~ ' + \
                "Sharpe Ratio: %.2f" % ep.sharpe_ratio(df_returns[df_returns.columns.to_list()[i]], period=period) + ' ~ ' + \
                "Max Drawdown: %.2f" % (-100 * ep.max_drawdown(df_returns[df_returns.columns.to_list()[i]])) + "\n"

    if len(tickers_list) != 0:
        suptitle_text = suptitle_text + str(tickers_list)
    plt.suptitle(suptitle_text, fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '.jpeg')
    # plt.show()
    plt.close()

def process_industry_dataframe(df):
    df['date'] = pd.to_datetime(df['date'])
    latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()
    required_columns = ['symbol', 'date'] + CAT1_RATIOS + CAT2_RATIOS
    # logging.info(f"Columns in required for V and G scores = {required_columns}")
    
    df_filtered = latest_data[required_columns]

    for col in df_filtered.columns:
        if col not in ['symbol', 'date']:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    df_filtered.fillna(df_filtered.median(numeric_only=True), inplace=True)               # questionable (we may drop the na row)
    df_score = df_filtered.copy()

    for col in CAT1_RATIOS:
        if col in df_score.columns:
            df_score[f'{col}_rank'] = df_score[col].apply(lambda x: score_factor(df_score[col], x, 1))

    for col in CAT2_RATIOS:
        if col in df_score.columns:
            df_score[f'{col}_rank'] = df_score[col].apply(lambda x: score_factor(df_score[col], x, 2))

    for col in CAT1_RATIOS:
        if col in df_score.columns:
            df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 1))

    for col in CAT2_RATIOS:
        if col in df_score.columns:
            df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 2))


    # # Check for duplicates in the 'symbol' column
    # duplicates = df_score[df_score.duplicated('symbol', keep=False)]

    # # Print the duplicates
    # if not duplicates.empty:
    #     print("Duplicate symbols found:")
    #     print(duplicates[['symbol']].drop_duplicates())
    # else:
    #     print("No duplicate symbols found.")

    logging.info(f"Columns after process industry factors = {df_score.columns.to_list()}")
    return df_score

def score(values, value, cat) -> int:
    try:
        std = statistics.stdev(values)
        mean = statistics.mean(values)
    except statistics.StatisticsError:
        # Handle cases where stdev or mean calculation fails (e.g., insufficient data points)
        return 0
    
    # if mean < 0:
    #    return 0
    if cat == 1:
        if value < 0:
            return 0
        if (mean + (-1 * std)) < value <= mean:
            return 1
        elif (mean + (-2 * std)) < value <= (mean + (-1 * std)):
            return 2
        elif value <= (mean + (-2 * std)):
            return 3
        elif mean < value <= (mean + (1 * std)):
            return -1
        elif (mean + (1 * std)) < value <= (mean + (2 * std)):
            return -2
        else:
            return -3
    else:
        if value < 0:
            return 0
        if mean <= value < (mean + (1 * std)):
            return 1
        elif (mean + (1 * std)) <= value < (mean + (2  *std)):
            return 2
        elif value >= (mean + (2 * std)):
            return 3
        elif (mean + (-1 * std)) <= value < mean:
            return -1
        elif (mean + (-2 * std)) <= value < (mean + (-1 * std)):
            return -2
        else:
            return -3

def score_factor(values, value, cat) -> int:
    try:
        std = statistics.stdev(values)
        mean = statistics.mean(values)
    except statistics.StatisticsError:
        # Handle cases where stdev or mean calculation fails (e.g., insufficient data points)
        return 0
    
    # if mean < 0:
    #    return 0
    if cat == 1:
        if value < 0:
            return 0
        if (mean - (0.6 * std)) < value <= mean:
            return 3
        elif ((mean - (1.2 * std))) <= value < (mean - (0.6 * std)):
            return 3.5
        elif ((mean - (1.8 * std))) <= value < (mean - (1.2 * std)):
            return 4
        elif ((mean - (2.4 * std))) <= value < (mean - (1.8 * std)):
            return 4.5
        elif value < (mean - (2.4 * std)):
            return 5

        elif mean < value <= (mean + (0.6 * std)):
            return 2.5
        elif (mean + (0.6 * std)) < value <= (mean + (1.2 * std)):
            return 2
        elif (mean + (1.2 * std)) < value <= (mean + (1.8 * std)):
            return 1.5
        elif (mean + (1.8 * std)) < value <= (mean + (2.4 * std)):
            return 1
        elif (mean + (2.4 * std)) < value:                    #<= (mean + (-1 * std)):
            return 0.5

    else:
        if value < 0:
            return 0
        if (mean - (0.6 * std)) < value <= mean:
            return 2.5
        elif ((mean - (1.2 * std))) <= value < (mean - (0.6 * std)):
            return 2.0
        elif ((mean - (1.8 * std))) <= value < (mean - (1.2 * std)):
            return 1.5
        elif ((mean - (2.4 * std))) <= value < (mean - (1.8 * std)):
            return 1.0
        elif value < (mean - (2.4 * std)):
            return 0.5

        elif mean < value <= (mean + (0.6 * std)):
            return 3.0
        elif (mean + (0.6 * std)) < value <= (mean + (1.2 * std)):
            return 3.5
        elif (mean + (1.2 * std)) < value <= (mean + (1.8 * std)):
            return 4
        elif (mean + (1.8 * std)) < value <= (mean + (2.4 * std)):
            return 4.5
        elif (mean + (2.4 * std)) < value:                    #<= (mean + (-1 * std)):
            return 5.0


            

# Define the existing functions
def calculate_value_scores(df_score, transformed_coefficients, industry):
    coefficients = transformed_coefficients.get(industry, None)
    if coefficients is None:
        print(f"Industry {industry} not found in coefficients.")
        logging.info(f"Industry {industry} not found in coefficients.")
        return pd.Series(np.zeros(len(df_score)), index=df_score.index)
    value_score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
    for metric, weight in coefficients.items():
        if metric in df_score.columns:
            value_score += df_score[metric] * weight
        else:
            print(f"Metric {metric} not found in DataFrame for industry {industry}.")
            logging.info(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
    return value_score

def rank_scores(df, score_column):
    try:
        # Adjust the quantile labels so that the highest scores get rank 10
        df[f'{score_column}_rank'] = pd.qcut(df[score_column], 10, labels=range(1, 11), duplicates='drop')
    except ValueError as e:
        # Handle the error by notifying and setting qcut rank to None
        # print(f"Error in ranking scores for {score_column} using qcut: {e}")
        df[f'{score_column}_rank'] = df[score_column].rank(method='min', ascending=True).astype(int)
        df[f'{score_column}_rank']  = ((df[f'{score_column}_rank']  - 1) / (df[f'{score_column}_rank'].max() - 1) * 9 + 1).astype(int)
    
    return df

def rank_scores_factor_cat1(df, score_column):
    try:
        # Adjust the quantile labels so that the lowest scores get rank 10
        df[f'{score_column}_rank'] = pd.qcut(df[score_column], 10, labels=range(10, 0, -1), duplicates='drop')
    except ValueError as e:
        # Handle the error by notifying and setting qcut rank to None
        print(f"Error in ranking scores for {score_column} using qcut: {e}")
        df[f'{score_column}_rank'] = df[score_column].rank(method='min', ascending=False).astype(int)
        df[f'{score_column}_rank'] = ((df[f'{score_column}_rank'] - 1) / (df[f'{score_column}_rank'].max() - 1) * 9 + 1).astype(int)
    
    return df

def rank_scores_factor_cat2(df, score_column):
    try:
        # Adjust the quantile labels so that the highest scores get rank 10
        df[f'{score_column}_rank'] = pd.qcut(df[score_column], 10, labels=range(1, 11), duplicates='drop')
    except ValueError as e:
        # Handle the error by notifying and setting qcut rank to None
        # print(f"Error in ranking scores for {score_column} using qcut: {e}")
        df[f'{score_column}_rank'] = df[score_column].rank(method='min', ascending=True).astype(int)
        df[f'{score_column}_rank']  = ((df[f'{score_column}_rank']  - 1) / (df[f'{score_column}_rank'].max() - 1) * 9 + 1).astype(int)
    
    return df

def rank_risk_scores(df, score_column):
    try:
        # Adjust the quantile labels so that the lowest scores get rank 10
        df[f'{score_column}_rank'] = pd.qcut(df[score_column], 10, labels=range(10, 0, -1), duplicates='drop')
    except ValueError as e:
        # Handle the error by notifying and setting qcut rank to None
        print(f"Error in ranking scores for {score_column} using qcut: {e}")
        df[f'{score_column}_rank'] = df[score_column].rank(method='min', ascending=False).astype(int)
        df[f'{score_column}_rank'] = ((df[f'{score_column}_rank'] - 1) / (df[f'{score_column}_rank'].max() - 1) * 9 + 1).astype(int)
    
    return df


def normalize_column(column):
    col_min = column.min()
    col_max = column.max()
    return 2 * (column - col_min) / (col_max - col_min) - 1

def calculate_growth_scores(df_score, transformed_coefficients, industry):
    coefficients = transformed_coefficients.get(industry, None)
    if coefficients is None:
        print(f"Industry {industry} not found in coefficients.")
        logging.info(f"Industry {industry} not found in coefficients.")
        return pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
    growth_score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
    for metric, weight in coefficients.items():
        if metric in df_score.columns:
            growth_score += df_score[metric] * weight
        else:
            print(f"Metric {metric} not found in DataFrame for industry {industry}.")
            logging.info(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
    return growth_score

def excel_to_pickle(industry_data_path):
    start_process_a = time.time()
    # industry_data_path = 'examples/Quant_Rating/data/industry_data_rs3000'#change the path here for quarter and annual, currently annual
    industry_files = os.listdir(industry_data_path)
    industry_dfs = {}

    for file in industry_files:
        if file.endswith('.xlsx'):
            industry_name = file.split('.')[0]
            file_path = os.path.join(industry_data_path, file)
            df = pd.read_excel(file_path)

            # =========== included to take care of earning dates ====================
            df = df.dropna(subset=['dateAccepted'])
            df['dateAccepted'] = pd.to_datetime(df['dateAccepted'])
            df['time'] = df['time'].fillna('amc')
            df.loc[df['time'].str.contains('amc', case=False), 'dateAccepted'] += pd.Timedelta(days=1)   # Increment 'dateAccepted' by one day if 'time' contains 'amc'
            df = df.drop(columns=['date'])
            df = df.rename(columns={'dateAccepted': 'date'})
            # ==============  end ===================================================

            industry_dfs[industry_name] = df

    # Write to Pickle
    with open(INDUSTRY_DATA_FILE, 'wb') as f:
        pickle.dump(industry_dfs, f)


    computation_time(start_process_a, message = "Total execution time: ")

def transform_coefficients(coeffs_dict):
    transformed_coefficients = {}
    for metric, industries in coeffs_dict.items():
        for industry, value in industries.items():
            if industry not in transformed_coefficients:
                transformed_coefficients[industry] = {}
            transformed_coefficients[industry][metric] = value
    return transformed_coefficients


def compute_risk_score(df_tics_daily, formation_date, tickers_list):
    # Calculate risk scores
    START_DATE = formation_date - dt.timedelta(days=365*RISK_VOLATILITY_PERIOD)
    END_DATE = formation_date

    # LP = Load_n_Preprocess(tickers_list, START_DATE, END_DATE)
    # df_tics_daily = LP.clean_daily_data(df_tics_daily)

    df_risk = pd.DataFrame(columns=['tic'])

    for period in RISK_PERIOD_LIST:
        RS = Risk_Score(END_DATE, period)
        key = f"risk_{period}"
        df_risk_score = RS.kk_risk_score(df_tics_daily)
        df_risk_score = df_risk_score.reset_index(drop=True)
        df_risk_score = df_risk_score.rename(columns={"Risk_Score": key})
        df_risk = pd.merge(df_risk, df_risk_score, on='tic', how='outer')

    # Calculate the overall risk score
    risk_columns = [f'risk_{period}' for period in RISK_PERIOD_LIST]
    df_risk['risk_score'] = df_risk[risk_columns].mean(axis=1)
    df_risk['risk_score'] = stats.zscore(df_risk['risk_score'])
    df_risk = rank_risk_scores(df_risk, 'risk_score')

    df_risk = df_risk.sort_values(by='risk_score', ascending=False)
    
    return df_risk

class Risk_Score:
    def __init__(self, formation_date, period):
        self.formation_date = formation_date
        self.end_date = pd.to_datetime(self.formation_date)
        self.start_date = self.end_date - timedelta(days=365*period)

    def kk_risk_score(self, df_tics_daily):
        df_tics_daily = df_tics_daily[(df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] <= self.end_date)].reset_index(drop=True)
        df_tics_daily_pivot = df_tics_daily.pivot(index='date', columns='tic', values='close')
        df_tics_daily_pivot = df_tics_daily_pivot.fillna(method='ffill').fillna(method='bfill')
        all_daily_return = df_tics_daily_pivot.pct_change()
        all_daily_return = all_daily_return.dropna()
        

        # Filter negative returns
        negative_returns = all_daily_return[all_daily_return < 0]

        # Compute the standard deviation of negative returns
        risk_score = negative_returns.std().fillna(0)
        # risk_score = risk_score.fillna(0)


        # risk_score = all_daily_return.std()  # Standard deviation of daily returns
        risk_zscore = stats.zscore(risk_score)


        df_tics_risk_zscore = pd.DataFrame({'tic': df_tics_daily.tic.unique(), 'Risk_Score': risk_zscore})
        # print('Risk scores are computed.')
        return df_tics_risk_zscore


def compute_vgm_score_v1_v2(df_tics_daily, formation_date, tickers_list, rank_based_on = None):
    value_coeff, growth_coeff = load_factors_coefficients_US()
    df_value_growth = compute_value_growth_score_US(formation_date, value_coeff, growth_coeff)
    df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

    df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]
    df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)

    df_vgm_scores = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')

    if rank_based_on == 'mom-value':
        df_vgm_scores= df_vgm_scores.sort_values(by=['momentum_score_rank','value_score_rank'], ascending=False)

    elif rank_based_on == 'mom-growth':
        df_vgm_scores = df_vgm_scores.sort_values(by=['momentum_score_rank','growth_score_rank'], ascending=False)
    
    else:
        df_vgm_scores = df_vgm_scores.sort_values(by=['momentum_score_rank','growth_score_rank','value_score_rank'], ascending=False)

    return df_vgm_scores

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def calculate_percentile(series):
    return (series.rank(method='min') - 1) / (len(series) - 1)

# def compute_vgm_score_IN(df_tics_daily, formation_date, tickers_list, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR = 'no'):
#     if RATING_OR_SCORE == 'score':
#         if VGM_METHOD == 'only-momentum':
#             df_vgm_scores = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
            
#             if CONSIDER_RISK_FACTOR == 'yes':
#                 df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
#                 df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

#                 df_vgm_scores['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_scores['momentum_score'])
#                 df_vgm_scores['norm_risk_score'] = 0.5 * (1 - min_max_normalize(df_vgm_scores['risk_score']))
#                 df_vgm_scores['norm_score_avg'] = df_vgm_scores[['norm_momentum_score', 'norm_risk_score']].sum(axis=1)

#                 # df_vgm_scores['z_momentum_score'] = 0.5 * zscore(df_vgm_scores['momentum_score'])
#                 # df_vgm_scores['z_risk_score'] = 0.5 * (- zscore(df_vgm_scores['risk_score']))
#                 # df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_momentum_score', 'z_risk_score']].sum(axis=1)

#                 df_vgm_scores = df_vgm_scores.sort_values(by='norm_score_avg', ascending=False)            

#             return df_vgm_scores
#         else:
#             value_coeff, growth_coeff = load_factors_coefficients_IN()
#             df_value_growth = compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff)
#             df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

#             df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]
#             df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
#             df_vgm_scores = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')
            
#             if SAVE_INDUSTRIAL_MOMENTUM_SCORE:
#                 df_momentum_ind = compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date)
#                 df_vgm_scores = pd.merge(df_vgm_scores, df_momentum_ind, on='tic', how='inner')

#             if CONSIDER_RISK_FACTOR == 'yes':
#                 # df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_vgm_scores['tic'])]
#                 df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
#                 df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

#             df_vgm_scores = pd.merge(df_vgm_scores, df_rank_marketcap, on='tic', how='left')
                       
#             if VGM_METHOD == 'z-score_avg':
#                 if CONSIDER_RISK_FACTOR == 'yes':  
#                     print(f"This combination is not implemented yet")
#                     # logger.warning(f"This combination is not implemented yet")
#                     return None

#                 elif CONSIDER_RISK_FACTOR == 'no':  
#                     df_vgm_scores['z_value_score'] = 0.25 * zscore(df_vgm_scores['value_score'])
#                     df_vgm_scores['z_growth_score'] = 0.25 * zscore(df_vgm_scores['growth_score'])
#                     df_vgm_scores['z_momentum_score'] = 0.5 * zscore(df_vgm_scores['momentum_score'])
#                     df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_value_score', 'z_growth_score', 'z_momentum_score']].sum(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='z_score_avg', ascending=False)

#             elif VGM_METHOD == 'min-max_avg': 
#                 if CONSIDER_RISK_FACTOR == 'yes':  
#                     df_vgm_scores['norm_value_score'] = 0.25 * min_max_normalize(df_vgm_scores['value_score'])
#                     df_vgm_scores['norm_growth_score'] = 0.25 * min_max_normalize(df_vgm_scores['growth_score'])
#                     df_vgm_scores['norm_momentum_score'] = 0.25 * min_max_normalize(df_vgm_scores['momentum_score'])
#                     df_vgm_scores['norm_risk_score'] = 0.25 * (1-min_max_normalize(df_vgm_scores['risk_score'])) 

#                     df_vgm_scores['min-max_avg'] = df_vgm_scores[['norm_value_score', 'norm_growth_score', 'norm_momentum_score','norm_risk_score']].sum(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='min-max_avg', ascending=False)
#                     df_vgm_scores = rank_scores(df_vgm_scores, 'min-max_avg')
#                     df_vgm_scores = df_vgm_scores.rename(columns = {'min-max_avg_rank':'vgm_score_rank','min-max_avg':'vgm_score'})
#                     # logger.warning(f"This combination is not implemented yet")

#                 elif CONSIDER_RISK_FACTOR == 'no':  
#                     df_vgm_scores['norm_value_score'] = 0.25 * min_max_normalize(df_vgm_scores['value_score'])
#                     df_vgm_scores['norm_growth_score'] = 0.25 * min_max_normalize(df_vgm_scores['growth_score'])
#                     df_vgm_scores['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_scores['momentum_score'])
#                     df_vgm_scores['min-max_avg'] = df_vgm_scores[['norm_value_score', 'norm_growth_score', 'norm_momentum_score']].sum(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='min-max_avg', ascending=False)
#                     df_vgm_scores = rank_scores(df_vgm_scores, 'min-max_avg')
#                     df_vgm_scores = df_vgm_scores.rename(columns = {'min-max_avg_rank':'vgm_score_rank','min-max_avg':'vgm_score'})

#             elif VGM_METHOD == 'percentile_avg':
#                 if CONSIDER_RISK_FACTOR == 'yes':
#                     df_vgm_scores['percentile_value_score'] = calculate_percentile(df_vgm_scores['value_score'])
#                     df_vgm_scores['percentile_growth_score'] = calculate_percentile(df_vgm_scores['growth_score'])
#                     df_vgm_scores['percentile_momentum_score'] = calculate_percentile(df_vgm_scores['momentum_score'])
#                     df_vgm_scores['percentile_risk_score'] = calculate_percentile(df_vgm_scores['risk_score'])
#                     df_vgm_scores['percentile_avg'] = df_vgm_scores[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score','percentile_risk_score']].mean(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='percentile_avg', ascending=False)

#                 elif CONSIDER_RISK_FACTOR == 'no':
#                     df_vgm_scores['percentile_value_score'] = 0.25 * calculate_percentile(df_vgm_scores['value_score'])
#                     df_vgm_scores['percentile_growth_score'] = 0.25 * calculate_percentile(df_vgm_scores['growth_score'])
#                     df_vgm_scores['percentile_momentum_score'] = 0.5 * calculate_percentile(df_vgm_scores['momentum_score'])
#                     df_vgm_scores['percentile_avg'] = df_vgm_scores[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score']].sum(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='percentile_avg', ascending=False)

#             elif VGM_METHOD == 'rank_avg':
#                 if CONSIDER_RISK_FACTOR == 'yes':  
#                     print(f"This combination is not implemented yet")
#                     # logger.warning(f"This combination is not implemented yet")
#                     return None

#                 elif CONSIDER_RISK_FACTOR == 'no': 
#                     df_vgm_scores['rank_value_score'] = 0.25 * df_vgm_scores['value_score'].rank(ascending=False, method='min')
#                     df_vgm_scores['rank_growth_score'] = 0.25 * df_vgm_scores['growth_score'].rank(ascending=False, method='min')
#                     df_vgm_scores['rank_momentum_score'] = 0.5 * df_vgm_scores['momentum_score'].rank(ascending=False, method='min')
#                     df_vgm_scores['average_rank'] = df_vgm_scores[['rank_value_score', 'rank_growth_score', 'rank_momentum_score']].mean(axis=1)
#                     df_vgm_scores = df_vgm_scores.sort_values(by='average_rank', ascending=True)

#             else:
#                 return None
#             df_vgm_scores = df_vgm_scores.reset_index(drop = True)
#             return df_vgm_scores
    
#     elif RATING_OR_SCORE =='rating':
        
#         if VGM_METHOD == 'value':
#             df_vgm_scores= df_vgm_scores.sort_values(by=['value_score_rank','marketcap'], ascending=False)

#         if VGM_METHOD == 'growth':
#             df_vgm_scores= df_vgm_scores.sort_values(by=['growth_score_rank','marketcap'], ascending=False)
        
#         elif VGM_METHOD == 'mom-value':
#             df_vgm_scores= df_vgm_scores.sort_values(by=['value_score_rank','momentum_score_rank','marketcap'], ascending=False)

#         elif VGM_METHOD == 'mom-growth':
#             df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','momentum_score_rank','marketcap'], ascending=False)
        
#         elif VGM_METHOD == 'growth_value':
#             df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','value_score_rank','momentum_score_rank','marketcap'], ascending=False)
        
#         elif VGM_METHOD == 'Avg_VGM':
#             df_vgm_scores['Avg_VGM'] = df_vgm_scores[['value_score_rank','growth_score_rank','momentum_score_rank']].astype(float).mean(axis=1)
#             df_vgm_scores = df_vgm_scores.sort_values(by=['Avg_VGM'], ascending=False)

#         elif VGM_METHOD == 'no-momentum':
#             df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','value_score_rank','marketcap'], ascending=False)
        
#         else:
#             return None
        
#         return df_vgm_scores

def compute_collinearity(df, col1, col2):
    """
    Computes the correlation coefficient between two columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    
    Returns:
    float: The correlation coefficient between the two columns.
    """
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("One or both specified columns do not exist in the DataFrame.")
    
    # Calculate correlation coefficient
    correlation = df[col1].corr(df[col2])
    return correlation
       
def compute_vgm_score(df_tics_daily, formation_date, tickers_list, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR = 'no'):
    if RATING_OR_SCORE == 'score':
        if VGM_METHOD == 'only-momentum':
            df_vgm_scores = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
            
            if CONSIDER_RISK_FACTOR == 'yes':
                df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
                df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

                if SCALING_METHOD == 'min-max':
                    df_vgm_scores['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_scores['momentum_score'])
                    df_vgm_scores['norm_risk_score'] = 0.5 * (1 - min_max_normalize(df_vgm_scores['risk_score']))
                    df_vgm_scores['norm_score_avg'] = df_vgm_scores[['norm_momentum_score', 'norm_risk_score']].sum(axis=1)

                    logging.info(f"Check for Multicollinearity between (i) momentum_score and (ii) risk_score = {round(compute_collinearity(df_vgm_scores, 'norm_momentum_score', 'norm_risk_score'),4)}")

                elif SCALING_METHOD == 'z-score':
                    df_vgm_scores['z_momentum_score'] = 0.5 * zscore(df_vgm_scores['momentum_score'])
                    df_vgm_scores['z_risk_score'] = 0.5 * (-zscore(df_vgm_scores['risk_score']))
                    df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_momentum_score', 'z_risk_score']].sum(axis=1)
                    
                    logging.info(f"Check for Multicollinearity between (i) momentum_score and (ii) risk_score = {round(compute_collinearity(df_vgm_scores, 'norm_momentum_score', 'norm_risk_score'),4)}")

                df_vgm_scores = df_vgm_scores.sort_values(by='norm_score_avg', ascending=False)            

            return df_vgm_scores
        else:
            if COUNTRY == 'IN':
                value_coeff, growth_coeff = load_factors_coefficients_IN()
                df_value_growth = compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff)

            elif COUNTRY == 'US':
                value_coeff, growth_coeff = load_factors_coefficients_US()
                df_value_growth = compute_value_growth_score_US(formation_date, value_coeff, growth_coeff)

            else:
                print(f"Other country do not support")

            df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

            df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]        # select only those tickers which are scored (V and G)
            df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
            df_vgm_scores = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')

            if SAVE_INDUSTRIAL_MOMENTUM_SCORE:
                df_momentum_ind = compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date)
                df_vgm_scores = pd.merge(df_vgm_scores, df_momentum_ind, on='tic', how='inner')

            if CONSIDER_RISK_FACTOR == 'yes':
                df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
                df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

            df_vgm_scores = pd.merge(df_vgm_scores, df_rank_marketcap, on='tic', how='left')

        if VGM_METHOD == 'z-score_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_scores['z_value_score'] = 0.25 * zscore(df_vgm_scores['value_score'])
                df_vgm_scores['z_growth_score'] = 0.25 * zscore(df_vgm_scores['growth_score'])
                df_vgm_scores['z_momentum_score'] = 0.25 * zscore(df_vgm_scores['momentum_score'])
                df_vgm_scores['z_risk_score'] = 0.25 * (- zscore(df_vgm_scores['risk_score'])) 
                
                df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_value_score', 'z_growth_score', 'z_momentum_score','z_risk_score']].sum(axis=1)
                df_vgm_scores = df_vgm_scores.sort_values(by='z_score_avg', ascending=False)
                
            elif CONSIDER_RISK_FACTOR == 'no':
                df_vgm_scores['z_value_score'] = 0.25 * zscore(df_vgm_scores['value_score'])
                df_vgm_scores['z_growth_score'] = 0.25 * zscore(df_vgm_scores['growth_score'])
                df_vgm_scores['z_momentum_score'] = 0.5 * zscore(df_vgm_scores['momentum_score'])
                df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_value_score', 'z_growth_score', 'z_momentum_score']].sum(axis=1)
                df_vgm_scores = df_vgm_scores.sort_values(by='z_score_avg', ascending=False)

            df_vgm_scores = rank_scores(df_vgm_scores, 'z_score_avg')
            df_vgm_scores = df_vgm_scores.rename(columns = {'z_score_avg_rank':'vgm_score_rank','z_score_avg':'vgm_score'})
            return df_vgm_scores

        if VGM_METHOD == 'min-max_avg': 
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_scores['norm_value_score'] = 0.25 * min_max_normalize(df_vgm_scores['value_score'])
                df_vgm_scores['norm_growth_score'] = 0.25 * min_max_normalize(df_vgm_scores['growth_score'])
                df_vgm_scores['norm_momentum_score'] = 0.25 * min_max_normalize(df_vgm_scores['momentum_score'])
                df_vgm_scores['norm_risk_score'] = 0.25 * (1 - min_max_normalize(df_vgm_scores['risk_score'])) 
                df_vgm_scores['min-max_avg'] = df_vgm_scores[['norm_value_score', 'norm_growth_score', 'norm_momentum_score','norm_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_scores['norm_value_score'] = 0.25 * min_max_normalize(df_vgm_scores['value_score'])
                df_vgm_scores['norm_growth_score'] = 0.25 * min_max_normalize(df_vgm_scores['growth_score'])
                df_vgm_scores['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_scores['momentum_score'])
                df_vgm_scores['min-max_avg'] = df_vgm_scores[['norm_value_score', 'norm_growth_score', 'norm_momentum_score']].sum(axis=1)
            
            df_vgm_scores = df_vgm_scores.sort_values(by='min-max_avg', ascending=False)  
            df_vgm_scores = rank_scores(df_vgm_scores, 'min-max_avg')
            df_vgm_scores = df_vgm_scores.rename(columns = {'min-max_avg_rank':'vgm_score_rank','min-max_avg':'vgm_score'})
            
            return df_vgm_scores   

        if VGM_METHOD == 'percentile_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_scores['percentile_value_score'] = 0.25 * calculate_percentile(df_vgm_scores['value_score'])
                df_vgm_scores['percentile_growth_score'] = 0.25 * calculate_percentile(df_vgm_scores['growth_score'])
                df_vgm_scores['percentile_momentum_score'] = 0.25 * calculate_percentile(df_vgm_scores['momentum_score'])
                df_vgm_scores['percentile_risk_score'] = 0.25 * (1 - calculate_percentile(df_vgm_scores['risk_score'])) 
                df_vgm_scores['percentile_avg'] = df_vgm_scores[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score','percentile_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_scores['percentile_value_score'] = 0.25 * calculate_percentile(df_vgm_scores['value_score'])
                df_vgm_scores['percentile_growth_score'] = 0.25 * calculate_percentile(df_vgm_scores['growth_score'])
                df_vgm_scores['percentile_momentum_score'] = 0.5 * calculate_percentile(df_vgm_scores['momentum_score'])
                df_vgm_scores['percentile_avg'] = df_vgm_scores[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score']].sum(axis=1)
            
            df_vgm_scores = df_vgm_scores.sort_values(by='percentile_avg', ascending=False)
            df_vgm_scores = rank_scores(df_vgm_scores, 'percentile_avg')
            df_vgm_scores = df_vgm_scores.rename(columns = {'percentile_avg_rank':'vgm_score_rank','percentile_avg':'vgm_score'})
            
            return df_vgm_scores

        if VGM_METHOD == 'rank_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_scores['rank_value_score'] = 0.25 * df_vgm_scores['value_score_rank']
                df_vgm_scores['rank_growth_score'] = 0.25 * df_vgm_scores['growth_score_rank']
                df_vgm_scores['rank_momentum_score'] = 0.25 * df_vgm_scores['momentum_score_rank']
                df_vgm_scores['rank_risk_score'] = 0.25 * df_vgm_scores['risk_score_rank']
                df_vgm_scores['rank_avg'] = df_vgm_scores[['rank_value_score', 'rank_growth_score', 'rank_momentum_score','rank_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_scores['rank_value_score'] = df_vgm_scores['value_score_rank']
                df_vgm_scores['rank_growth_score'] = df_vgm_scores['growth_score_rank']
                df_vgm_scores['rank_momentum_score'] = df_vgm_scores['momentum_score_rank']
                df_vgm_scores['rank_avg'] = df_vgm_scores[['rank_value_score', 'rank_growth_score', 'rank_momentum_score']].astype(float).sum(axis=1)
            
            df_vgm_scores = df_vgm_scores.sort_values(by='rank_avg', ascending=False)
            df_vgm_scores = rank_scores(df_vgm_scores, 'rank_avg')
            df_vgm_scores = df_vgm_scores.rename(columns = {'rank_avg_rank':'vgm_score_rank','rank_avg':'vgm_score'})
            
            return df_vgm_scores

    elif RATING_OR_SCORE =='rating':
       
        if COUNTRY == 'IN':
            value_coeff, growth_coeff = load_factors_coefficients_IN()
            df_value_growth = compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff)

        elif COUNTRY == 'US':
            value_coeff, growth_coeff = load_factors_coefficients_US()
            df_value_growth = compute_value_growth_score_US(formation_date, value_coeff, growth_coeff)

        else:
            print(f"Other country do not support")
            
        df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

        df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]        # select only those tickers which are scored (V and G)
        df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
        df_vgm_scores = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')

        if SAVE_INDUSTRIAL_MOMENTUM_SCORE:
            df_momentum_ind = compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date)
            df_vgm_scores = pd.merge(df_vgm_scores, df_momentum_ind, on='tic', how='inner')

        if CONSIDER_RISK_FACTOR == 'yes':
            df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
            df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

        df_vgm_scores = pd.merge(df_vgm_scores, df_rank_marketcap, on='tic', how='left')

        if VGM_METHOD == 'value':
            df_vgm_scores= df_vgm_scores.sort_values(by=['value_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'growth':
            df_vgm_scores= df_vgm_scores.sort_values(by=['growth_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'value-mom':
            df_vgm_scores= df_vgm_scores.sort_values(by=['value_score_rank','momentum_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'growth-mom':
            df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','momentum_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'mom-value':
            df_vgm_scores= df_vgm_scores.sort_values(by=['momentum_score_rank','value_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'mom-growth':
            df_vgm_scores = df_vgm_scores.sort_values(by=['momentum_score_rank','growth_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'growth-value-mom':
            df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','value_score_rank','momentum_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'growth-value':
            df_vgm_scores = df_vgm_scores.sort_values(by=['growth_score_rank','value_score_rank','marketcap'], ascending=False)
        
        else:
            return None
        
        return df_vgm_scores

    

    # if not (VGM_METHOD == 'only-momentum'):
       
        
    #     if CONSIDER_RISK_FACTOR == 'no':
           

    #         if RATING_OR_SCORE == 'score':
                

    #     if CONSIDER_RISK_FACTOR == 'yes':
    #         df_vgm_scores['percentile_value_score'] = calculate_percentile(df_vgm_scores['value_score'])
    #         df_vgm_scores['percentile_growth_score'] = calculate_percentile(df_vgm_scores['growth_score'])
    #         df_vgm_scores['percentile_momentum_score'] = calculate_percentile(df_vgm_scores['momentum_score'])
    #         df_vgm_scores['percentile_risk_score'] = calculate_percentile(df_vgm_scores['risk_score'])
    #         df_vgm_scores['percentile_avg'] = df_vgm_scores[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score','percentile_risk_score']].mean(axis=1)
    #         df_vgm_scores = df_vgm_scores.sort_values(by='percentile_avg', ascending=False)

    #     df_vgm_scores = df_vgm_scores.reset_index(drop = True)
    #     return df_vgm_scores
    
    # else:
        # df_vgm_scores = compute_momentum_score(df_tics_daily, formation_date, tickers_list)

        # if CONSIDER_RISK_FACTOR == 'yes':
        #     df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
        #     df_vgm_scores = pd.merge(df_vgm_scores, df_risk, on='tic', how='inner')

        #     df_vgm_scores['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_scores['momentum_score'])
        #     df_vgm_scores['norm_risk_score'] = 0.5 * (1 - min_max_normalize(df_vgm_scores['risk_score']))
        #     df_vgm_scores['norm_score_avg'] = df_vgm_scores[['norm_momentum_score', 'norm_risk_score']].sum(axis=1)

        #     # df_vgm_scores['z_momentum_score'] = 0.5 * zscore(df_vgm_scores['momentum_score'])
        #     # df_vgm_scores['z_risk_score'] = 0.5 * (- zscore(df_vgm_scores['risk_score']))
        #     # df_vgm_scores['z_score_avg'] = df_vgm_scores[['z_momentum_score', 'z_risk_score']].sum(axis=1)

        #     df_vgm_scores = df_vgm_scores.sort_values(by='norm_score_avg', ascending=False)            

        # return df_vgm_scores

    

class Momentum_Score:
    def __init__(self, formation_date, risk_free_rate=0):
        self.formation_date = formation_date
        self.end_date = pd.to_datetime(self.formation_date)
        self.start_date = self.end_date - timedelta(days = 365*MOM_VOLATILITY_PERIOD)
        self.risk_free_rate = risk_free_rate

    def kk_momentum_score(self, df_tics_daily, momentum_period=3, gap=1):
        T1 = momentum_period
        df_tics_daily = df_tics_daily[(df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] <= self.end_date)].reset_index(drop=True)
        df_tics_daily_pivot = df_tics_daily.pivot(index='date', columns='tic', values='close')
        all_daily_return = df_tics_daily_pivot.pct_change()
        std_daily_return = math.sqrt(252) * all_daily_return.std()
        risk_adj_mom_score = (((df_tics_daily_pivot.iloc[-gap] / df_tics_daily_pivot.iloc[-21 * T1 - gap]) - 1) / std_daily_return) - self.risk_free_rate
        
        risk_adj_mom_score = risk_adj_mom_score.dropna()
        
        momentum_zscore = stats.zscore(risk_adj_mom_score)
        df_tics_momentum_zscore = pd.DataFrame(momentum_zscore).reset_index()
        df_tics_momentum_zscore.columns = ['tic','Momentum_Score'] 
        # df_tics_momentum_zscore = pd.DataFrame({'tic': df_tics_daily.tic.unique(), 'Momentum_Score': momentum_zscore})
        # print('Momentum scores are computed.')
        return df_tics_momentum_zscore

def compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date):
    df_industry_mom = pd.DataFrame([])
    for ind in df_value_growth.industry.unique():
        if ind != 'Specialty Retail':
            tickers_list = df_value_growth[df_value_growth['industry'] == ind].tic.to_list()
            df_tics_daily_ind = df_tics_daily[df_tics_daily.tic.isin(tickers_list)]
            print(f'{ind} = {len(tickers_list)} and {len(df_tics_daily_ind)}')
            df_tics_mom = compute_momentum_score(df_tics_daily_ind, formation_date, tickers_list)
            df_industry_mom = pd.concat([df_industry_mom, df_tics_mom], ignore_index= True)

    df_industry_mom = df_industry_mom.rename(columns = {'momentum_3_rank': 'ind_momentum_3_rank',
                                                        'momentum_6_rank': 'ind_momentum_6_rank',
                                                        'momentum_12_rank': 'ind_momentum_12_rank'})
    return df_industry_mom[['tic','ind_momentum_3_rank','ind_momentum_6_rank','ind_momentum_12_rank']]


def compute_momentum_score(df_tics_daily, formation_date, tickers_list):
    # Calculate momentum scores
    # tickers_list = df_new_score['symbol'].unique().tolist()
    # END_DATE = dt.datetime.now()#df['date'].max()
    # START_DATE = END_DATE - timedelta(days=400)
    
    START_DATE = formation_date - dt.timedelta(days = 365*MOM_VOLATILITY_PERIOD)
    END_DATE = formation_date

    # LP = Load_n_Preprocess(tickers_list, START_DATE, END_DATE)
    # # df_tics_daily = LP.download_yfinance(is_live=False)
    # df_tics_daily = LP.clean_daily_data(df_tics_daily)

    
    momentum_columns = [f'momentum_{period}' for period in MOMENTUM_PERIOD_LIST]
    df_momentum = pd.DataFrame(columns=['tic'])

    for period in MOMENTUM_PERIOD_LIST:
        MS = Momentum_Score(END_DATE)
        key = f"momentum_{period}"
        df_momentum_score = MS.kk_momentum_score(df_tics_daily, momentum_period=period, gap=5)
        # df_momentum_score['Momentum_Score'] = normalize_column(df_momentum_score['Momentum_Score'])
        df_momentum_score = df_momentum_score.reset_index(drop=True)
        df_momentum_score = df_momentum_score.rename(columns={"Momentum_Score": key})
        
        if df_momentum.empty:
            df_momentum = df_momentum_score.copy()  # Copy the df_momentum_score to df_momentum
        else:
            df_momentum = pd.merge(df_momentum, df_momentum_score, on='tic', how='inner')
        
        # df_momentum = pd.merge(df_momentum, df_momentum_score, on='tic', how='outer')     

    # Calculate the overall momentum score
    df_momentum['momentum_score'] = df_momentum[momentum_columns].mean(axis=1)#/len(momentum_columns)
    df_momentum['momentum_score'] = stats.zscore(df_momentum['momentum_score'])
    # df_momentum['momentum_score'] = (df_momentum['momentum_score'] - df_momentum['momentum_score'].min()) / (df_momentum['momentum_score'].max() - df_momentum['momentum_score'].min())
    # df_momentum['momentum_score'] = (df_momentum['momentum_score']-df_momentum['momentum_score'].min())/df_momentum['momentum_score'].std()    #stats.zscore(df_momentum['score'])
    df_momentum = rank_scores(df_momentum, 'momentum_score')

    for col in momentum_columns:
        df_momentum = rank_scores(df_momentum, col)

    df_momentum = df_momentum.sort_values(by='momentum_score',ascending=False)

    df_tics_daily_temp = df_tics_daily[df_tics_daily['date'] <= pd.to_datetime(END_DATE)].reset_index(drop=True)
    df_tics_daily_temp = pd.DataFrame(df_tics_daily_temp.pivot(index='date', columns='tic', values='close').iloc[-1]).reset_index()
    df_tics_daily_temp.columns = ['tic','close']    
    df_momentum = pd.merge(df_momentum, df_tics_daily_temp, on='tic', how='left')
    df_momentum = df_momentum.sort_values(by='momentum_score',ascending=False)
    return df_momentum

def compute_value_growth_score_US(formation_date, value_coeff, growth_coeff):
    with open(INDUSTRY_DATA_FILE, 'rb') as f:
            industry_dfs = pickle.load(f)
            logging.info(f"Columns in fundamental data = {industry_dfs[list(industry_dfs.keys())[1]].columns.to_list()}")

    if SAVE_EXCEL:
        combined_df = pd.concat(industry_dfs.values(), ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')
        combined_df.to_excel('./examples/Quant_Rating/datasets/industry_dfs_US.xlsx', index=False)

    # Update the main loop to include momentum score calculation
    df_industry_scores = pd.DataFrame([])
    # df_industry_list = []
    sum_i = 0
    for industry, df in industry_dfs.items():
        if industry not in value_coeff or industry not in growth_coeff:
            # print(f"Skipping industry: {industry} as it does not have regression coefficients.")
            logging.info(f"Skipping industry: {industry} as it does not have regression coefficients.")
            continue

        if (len(df['symbol'].unique()) >= 7):
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] < pd.to_datetime(formation_date)]    # changed <= to <    on 13/08/2024
            # print(f"Industry: {industry}, Number of Symbols: {len(df['symbol'].unique())}, Data Shape: {df.shape}")
            logging.info(f"Industry: {industry}, Number of Symbols: {len(df['symbol'].unique())}, Data Shape: {df.shape}")

            df_score = process_industry_dataframe(df)
            
            # sum_i = sum_i + len(df_score)
            # print(f"Scores are computed for industry {industry}")
            df_score['industry'] = industry
            df_score['sector']  = df['sector'].unique()[0]

            # Calculate value and growth scores
            df_score['value_score'] = calculate_value_scores(df_score, value_coeff, industry)
            df_score['growth_score'] = calculate_growth_scores(df_score, growth_coeff, industry)      # normalize industry score between 0-1, standardize
            
            if SCALING_METHOD == 'z-score':
                df_score['value_score'] = (df_score['value_score'] - df_score['value_score'].mean()) / df_score['value_score'].std()
                df_score['growth_score'] = (df_score['growth_score'] - df_score['growth_score'].mean()) / df_score['growth_score'].std()
            
            elif SCALING_METHOD == 'min-max':
                df_score['value_score'] = (df_score['value_score'] - df_score['value_score'].min()) / (df_score['value_score'].max() - df_score['value_score'].min())
                df_score['growth_score'] = (df_score['growth_score'] - df_score['growth_score'].min()) / (df_score['growth_score'].max() - df_score['growth_score'].min())
            
            else:
                df_score['value_score'] = (df_score['value_score'] - df_score['value_score'].min()) / (df_score['value_score'].max() - df_score['value_score'].min())
                df_score['growth_score'] = (df_score['growth_score'] - df_score['growth_score'].min()) / (df_score['growth_score'].max() - df_score['growth_score'].min())
            

            # Rank the scores
            df_score = rank_scores(df_score, 'value_score')
            df_score = rank_scores(df_score, 'growth_score')


            # df_score = rank_scores_factor(df_score, CAT1_RATIOS)
            # df_score = rank_scores_factor(df_score, CAT2_RATIOS)



            # Keep only the specified columns
            # df_score = df_score[['symbol','sector','industry', 'value_score', 'growth_score', 'value_score_rank', 'growth_score_rank']]

            # if len(df_score['symbol'].unique()) != len(df_score['symbol']):
            #     print(industry)

            # # Check calculated scores and ranks
            # print(f"Value scores for {industry}: {df_score[['value_score', 'value_score_rank']].head()}")
            # print(f"Growth scores for {industry}: {df_score[['growth_score', 'growth_score_rank']].head()}")

            # df['date'] = pd.to_datetime(df['date'])
            # latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()

            # # Ensure the required metrics are present in the DataFrame
            # missing_metrices = [metric for metric in VALUE_METRICES + GROWTH_METRICES if metric not in latest_data.columns]
            # if missing_metrices:
            #     print(f"Missing metrics {missing_metrices} for industry {industry}. Skipping this industry.")
            #     continue

            # # Filter to include only the necessary columns
            # required_columns = ['symbol', 'date']
            # df_filtered = latest_data[required_columns]

            # Merge the filtered DataFrame with the score DataFrame
            # df_new_score = pd.merge(df_filtered, df_score, on='symbol', how='left')

            #UNCOMMENTED THIS LINES:  (Duplicate problem)
            # latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()
            # required_columns = ['symbol', 'date']
            # df_filtered = latest_data[required_columns]
            # df_new_score = pd.merge(df_filtered, df_score, on='symbol', how='left')

            # df_industry_list.append(df_score)
            df_industry_scores = pd.concat([df_industry_scores, df_score], ignore_index= True)
        
        else:
            logging.info(f"Industry: {industry}, Number of Symbols: {len(df['symbol'].unique())}, Data Shape: {df.shape}")


    df_industry_scores = df_industry_scores.drop_duplicates(subset='symbol', keep='first')

    return df_industry_scores

def impute_missing_values(df):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.replace(0, np.nan)
    all_nan_columns = numeric_df.columns[numeric_df.isna().all()].tolist()
    if all_nan_columns:
        print(f"Columns dropped due to all NaN values: {all_nan_columns}")

    numeric_df = numeric_df.dropna(axis=1, how='all')
    knn_imputer = KNNImputer(n_neighbors=5)
    numeric_df_imputed = pd.DataFrame(knn_imputer.fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)
    df_imputed = df.copy()
    df_imputed[numeric_df.columns] = numeric_df_imputed
    
    return df_imputed

def compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff):
    with open(INDUSTRY_DATA_FILE, 'rb') as f:
            industry_dfs = pickle.load(f)
            logging.info(f"Columns in fundamental data = {industry_dfs[list(industry_dfs.keys())[1]].columns.to_list()}")

    if SAVE_EXCEL:
        combined_df = pd.concat(industry_dfs.values(), ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')
        combined_df.to_excel('./examples/Quant_Rating/datasets/industry_dfs_IN.xlsx', index=False)

    # industry_data_path = PATH_DATA +'NSE_DATA'  # Change the path here for quarter and annual data
    # industry_files = os.listdir(industry_data_path)

    # industry_dfs = {}
    # for file in industry_files:
    #     if file.endswith('.xlsx'):
    #         industry_name = file.split('.')[0]
    #         file_path = os.path.join(industry_data_path, file)
    #         industry_dfs[industry_name] = pd.read_excel(file_path)

    # for file in industry_files:
    #     if file.endswith('.xlsx'):
    #         industry_name = file.split('.')[0]
    #         file_path = os.path.join(industry_data_path, file)
    #         df = pd.read_excel(file_path)

    #         # =========== included to take care of earning dates ====================
    #         df = df.dropna(subset=['dateAccepted'])
    #         df['dateAccepted'] = pd.to_datetime(df['dateAccepted'])
    #         df['time'] = df['time'].fillna('amc')
    #         df.loc[df['time'].str.contains('amc', case=False), 'dateAccepted'] += pd.Timedelta(days=1)   # Increment 'dateAccepted' by one day if 'time' contains 'amc'
    #         df = df.drop(columns=['date'])
    #         df = df.rename(columns={'dateAccepted': 'date'})
    #         # ==============  end ===================================================

    #         industry_dfs[industry_name] = df

    ## =================    Begin : New features and Data imputation ===================================

    for industry, df in list(industry_dfs.items()):
        if industry not in value_coeff or industry not in growth_coeff:
            # print(f"Skipping industry: {industry} as it does not have regression coefficients.")
            continue
        
        # ## ======= Begin: edited by Kundan  =============
        # df = df.drop('date', axis=1)
        # df = df.rename(columns={'broadCastDate': 'date'})
        # ## ======= End: edited by Kundan  =============

        if len(df['symbol'].unique()) >= 5:

            df = df.groupby('symbol', group_keys=False).apply(lambda x: x.sort_values('date', ascending=True)).reset_index(drop=True)
            metrics = ['revenue', 'eps', 'incomeBeforeTax']
            
            for metric in metrics:
                df[f'{metric} 1-Period Avg Growth'] = df.groupby('symbol')[metric].apply(lambda x: x.diff() / x.shift(1)).reset_index(drop=True)
                df[f'{metric} 3-Period Avg Growth'] = df.groupby('symbol')[f'{metric} 1-Period Avg Growth'].apply(lambda x: x.rolling(window=3).mean()).reset_index(drop=True)

            df['date'] = pd.to_datetime(df['date'])  
            df = df[df['date'] <= pd.to_datetime(formation_date)]
            df = df.groupby('symbol', group_keys=False).apply(lambda x: x.sort_values('date', ascending=False)).reset_index(drop=True)
            req_cols = VALUE_METRICES + GROWTH_METRICES + ['symbol', 'sector', 'industry', 'date', 'period', 'broadCastDate', 'time', 'er']
            df = df[req_cols]

            # === start: added by Kundan ==================
            df['broadCastDate'] = pd.to_datetime(df['broadCastDate'], format='%d-%b-%Y %H:%M:%S').dt.date
            # If you want to keep it as a datetime object but with only the date part:
            df['broadCastDate'] = pd.to_datetime(df['broadCastDate'])
            # === end: added by Kundan ==================

            industry_dfs[industry] = impute_missing_values(df)
            
            # print(f"Processed {industry}: {industry_dfs[industry].shape}")
            logging.info(f"Processed {industry}: {industry_dfs[industry].shape}")

    ## =================    End : New features and Data imputation ===================================



    # if SAVE_EXCEL:
    #     combined_df = pd.concat(industry_dfs.values(), ignore_index=True)
    #     combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')
    #     combined_df.to_excel('./examples/Quant_Rating/datasets/industry_dfs.xlsx', index=False)



    # Update the main loop to include momentum score calculation
    df_industry_scores = pd.DataFrame([])
    # df_industry_list = []
    sum_i = 0
    for industry, df in industry_dfs.items():
        if industry not in value_coeff or industry not in growth_coeff:
            # print(f"Skipping industry: {industry} as it does not have regression coefficients.")
            continue
        
        # ## ======= Begin: edited by Kundan  =============
        # df = df.drop('date', axis=1)
        # df = df.rename(columns={'broadCastDate': 'date'})
        # ## ======= End: edited by Kundan  =============

        if len(df['symbol'].unique()) >= 5:
            # df['date'] = pd.to_datetime(df['date'])
            df = df[df['broadCastDate'] < pd.to_datetime(formation_date)]    # changed <= to <    on 13/08/2024
            # print(f"Industry: {industry}, Number of Symbols: {len(df['symbol'].unique())}, Data Shape: {df.shape}")

            df_score = process_industry_dataframe(df)
            # sum_i = sum_i + len(df_score)
            # print(f"Scores are computed for industry {industry}")
            df_score['industry'] = industry
            df_score['sector']  = df['sector'].unique()[0]

            # Calculate value and growth scores
            df_score['value_score'] = calculate_value_scores(df_score, value_coeff, industry)
            df_score['growth_score'] = calculate_growth_scores(df_score, growth_coeff, industry)      # normalize industry score between 0-1, standardize
            
            df_score['value_score'] = (df_score['value_score'] - df_score['value_score'].min()) / (df_score['value_score'].max() - df_score['value_score'].min())
            df_score['growth_score'] = (df_score['growth_score'] - df_score['growth_score'].min()) / (df_score['growth_score'].max() - df_score['growth_score'].min())
            
            # df_score['value_score'] = (df_score['value_score'] - df_score['value_score'].mean()) / df_score['value_score'].std()
            # df_score['growth_score'] = (df_score['growth_score'] - df_score['growth_score'].mean()) / df_score['growth_score'].std()

            # Rank the scores
            df_score = rank_scores(df_score, 'value_score')
            df_score = rank_scores(df_score, 'growth_score')

            # Keep only the specified columns
            # df_score = df_score[['symbol', 'sector', 'industry', 'value_score', 'growth_score', 'value_score_rank', 'growth_score_rank']]

            # if len(df_score['symbol'].unique()) != len(df_score['symbol']):
            #     print(industry)

            # # Check calculated scores and ranks
            # print(f"Value scores for {industry}: {df_score[['value_score', 'value_score_rank']].head()}")
            # print(f"Growth scores for {industry}: {df_score[['growth_score', 'growth_score_rank']].head()}")

            # df['date'] = pd.to_datetime(df['date'])
            # latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()

            # # Ensure the required metrics are present in the DataFrame
            # missing_metrices = [metric for metric in VALUE_METRICES + GROWTH_METRICES if metric not in latest_data.columns]
            # if missing_metrices:
            #     print(f"Missing metrics {missing_metrices} for industry {industry}. Skipping this industry.")
            #     continue

            # # Filter to include only the necessary columns
            # required_columns = ['symbol', 'date']
            # df_filtered = latest_data[required_columns]

            # Merge the filtered DataFrame with the score DataFrame
            # df_new_score = pd.merge(df_filtered, df_score, on='symbol', how='left')

            #UNCOMMENTED THIS LINES:  (Duplicate problem)
            # latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()
            # required_columns = ['symbol', 'date']
            # df_filtered = latest_data[required_columns]
            # df_new_score = pd.merge(df_filtered, df_score, on='symbol', how='left')

            # df_industry_list.append(df_score)
            df_industry_scores = pd.concat([df_industry_scores, df_score], ignore_index= True)

    df_industry_scores = df_industry_scores.drop_duplicates(subset='symbol', keep='first')

    return df_industry_scores

def fmp_download_hist_marketcap(TICKERS_LIST, START_DATE_MKTCAP, END_DATE_MKTCAP, filename='data/df_hist_marketcap'):
    # start_time = time.time()
    # START_DATE_MKTCAP = (END_DATE - dt.timedelta(days=365*6)).strftime('%Y-%m-%d')
    # END_DATE_MKTCAP = END_DATE.strftime('%Y-%m-%d')

    def fetch_marketcap(tic):
        print(tic)
        search_url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{tic}?limit=2000&from={START_DATE_MKTCAP}&to={END_DATE_MKTCAP}&apikey=8481ed700f2bf3bb0575f4e9d88f8bbf"
        response = requests.get(search_url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to retrieve data for {tic}: {response.status_code}")
            return pd.DataFrame()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_marketcap, tic): tic for tic in TICKERS_LIST}
        temp_list = []
        for future in as_completed(futures):
            temp_list.append(future.result())

    df_marketcap = pd.concat(temp_list, ignore_index=True)
    save_dataframe(df_marketcap, filename, save_as='h5')
    
    # end_time = time.time()
    # computation_time(start_time, message="Total download time:")

    return df_marketcap

def StockReturnsComputing(prices):
    # Vectorized return computation
    return (prices[1:, :] - prices[:-1, :]) / prices[:-1, :]

def compute_portfolio_stats(weights, covReturns, meanReturns):
    # Compute portfolio risk and return
    portfolio_risk = np.sqrt(weights @ covReturns @ weights.T).item()
    portfolio_return = (weights @ meanReturns.T).item()
    return portfolio_risk, portfolio_return

def diversification_score(df_tics_daily, tickers_list, df_marketcap = [], method='equal-weight'):
    # Pivot the data for closing prices
    df_tics_daily_close = df_tics_daily.pivot(index='date', columns='tic', values='close').sort_index()
    df_tics_daily_close = df_tics_daily_close[tickers_list]
    
    # Extract stock prices as a NumPy array
    arStockPrices = df_tics_daily_close.to_numpy()

    # Compute asset returns
    arReturns = StockReturnsComputing(arStockPrices)
    
    # Calculate mean returns and covariance matrix
    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)
    
    # Set precision for NumPy prints
    np.set_printoptions(precision=5, suppress=True)

    # Compute weights based on method
    if method == 'equal-weight':
        # Equal weights: 1/N for each asset
        PortfolioSize = len(meanReturns)
        weights = np.full(PortfolioSize, 1.0 / PortfolioSize)
    
    elif method == 'inv-volatility':
        # Inverse volatility weights: 1/σ for each asset
        asset_risks = np.sqrt(np.diagonal(covReturns))
        weights = 1.0 / asset_risks
        weights /= weights.sum()  # Normalize to ensure sum(weights) = 1

    elif method == 'marketcap-weight':
        
        if len(df_marketcap) == 0:
            print("df_marketcap is not provided")
            return 0
        else:
            df_marketcap = df_marketcap[df_marketcap['symbol'].isin(df_tics_daily_close.columns.to_list())]
            df_marketcap = df_marketcap.sort_values(by = ['symbol']).reset_index(drop = True)
            df_marketcap['weights'] = df_marketcap['weights']/df_marketcap['weights'].sum()
            weights = np.array(df_marketcap['weights'])


        # ROOT_PATH = '../../datasets/'
        # df_marketcap = pd.read_hdf(ROOT_PATH + "df_hist_marketcap_NIFTY50_2024.h5", "df", mode = 'r')
        # df_marketcap = df_marketcap.pivot_table(index = 'date', columns = 'symbol',values = 'marketCap').reset_index()
        # df_marketcap['date'] = pd.to_datetime(df_marketcap['date'])
        # df_marketcap = df_marketcap.sort_values(by = 'date')
        # index_weights= pd.DataFrame(df_marketcap.iloc[-1])
        # index_weights =index_weights.iloc[1:,:].reset_index()
        # index_weights.columns = ['symbol','weights']
        # index_weights['weights'] = index_weights['weights']/index_weights['weights'].sum()

    
    # Compute risk, return, and diversification ratio
    portfolio_risk, portfolio_return = compute_portfolio_stats(weights, covReturns, meanReturns)
    ann_portfolio_risk = portfolio_risk * np.sqrt(251) * 100
    ann_portfolio_return = portfolio_return * 251 * 100

    # print(f"Annualized Portfolio Risk: {ann_portfolio_risk:.2f}%")
    # print(f"Annualized Expected Portfolio Return: {ann_portfolio_return:.2f}%")
    
    # Compute diversification ratio
    asset_std_dev = np.sqrt(np.diagonal(covReturns))
    portfolio_div_ratio = np.sum(asset_std_dev * weights) / portfolio_risk
    # print(f"{method.title()} Portfolio's Diversification Ratio: {portfolio_div_ratio:.2f}\n")

    return portfolio_div_ratio

def load_factors_coefficients_US():
    rf_value_coeffs_df = pd.read_csv(INDUSTRY_REGRESSION_DIR + 'rf_value_coefficients_industry_q_new.csv')#change her for quarter path
    rf_growth_coeffs_df = pd.read_csv(INDUSTRY_REGRESSION_DIR + 'rf_growth_coefficients_industry_q_new.csv')#change her for quarter path
    # Rename the Unnamed: 0 column to metrics
    rf_value_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    rf_value_coeffs_df_cleaned = rf_value_coeffs_df.dropna(how='all', subset=rf_value_coeffs_df.columns[1:])
    rf_value_coeffs_df_cleaned.set_index('metrics', inplace=True)

    # Rename the Unnamed: 0 column to metrics
    rf_growth_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    rf_growth_coeffs_df_cleaned = rf_growth_coeffs_df.dropna(how='all', subset=rf_growth_coeffs_df.columns[1:])
    rf_growth_coeffs_df_cleaned.set_index('metrics', inplace=True)

    value_coefficients =rf_value_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()
    growth_coefficients =rf_growth_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

    transformed_value_coefficients = transform_coefficients(value_coefficients)
    transformed_growth_coefficients = transform_coefficients(growth_coefficients)

    return transformed_value_coefficients, transformed_growth_coefficients

def load_factors_coefficients_IN():
    rf_value_coeffs_df = pd.read_csv(INDUSTRY_REGRESSION_DIR + 'rf_value_coefficients_industry_q_new.csv')#change her for quarter path
    rf_growth_coeffs_df = pd.read_csv(INDUSTRY_REGRESSION_DIR + 'rf_growth_coefficients_industry_q_new.csv')#change her for quarter path
    # Rename the Unnamed: 0 column to metrics
    rf_value_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    rf_value_coeffs_df_cleaned = rf_value_coeffs_df.dropna(how='all', subset=rf_value_coeffs_df.columns[1:])
    rf_value_coeffs_df_cleaned.set_index('metrics', inplace=True)

    # Rename the Unnamed: 0 column to metrics
    rf_growth_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    rf_growth_coeffs_df_cleaned = rf_growth_coeffs_df.dropna(how='all', subset=rf_growth_coeffs_df.columns[1:])
    rf_growth_coeffs_df_cleaned.set_index('metrics', inplace=True)

    value_coefficients =rf_value_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()
    growth_coefficients =rf_growth_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

    transformed_value_coefficients = transform_coefficients(value_coefficients)
    transformed_growth_coefficients = transform_coefficients(growth_coefficients)

    return transformed_value_coefficients, transformed_growth_coefficients


if __name__ == "__main__":
    # industry_data_path = './examples/Quant_Rating/datasets/industry_data_rs3000'
    industry_data_path = './examples/Quant_Rating/datasets/NSE_DATA'
    
    excel_to_pickle(industry_data_path)