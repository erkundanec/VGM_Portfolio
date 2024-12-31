'''===================== 20240726_Quant_Portfolio ===================== %          azure file "Stock_Factor_Rating"
Description                  : Single Code: Evaluate VGM Score using Quant Portfolio
Input parameter              : Indian Stock historical close price and VGM Score
Output parameter             : Cumulative return of Portfolio and Indices
Subroutine  called           : NA
Called by                    : NA
Reference                    :
Author of the code           : Dr.Kundan Kumar
Date of creation             : 16/12/2024
------------------------------------------------------------------------------------------ %
Modified on                  : 
Modification details         : All strateges are in a single file. All function and variables are defined in the same file.
Modified By                  : Dr.Kundan Kumar
Previous version             : 20241221_Stock_Ratings.py
========================================================================================== %
'''

# ===========   disable all warnings   ======================================
import warnings
warnings.filterwarnings("ignore")

# ===========     Import libraries   ==========================================
import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5.0'

import sys
import pandas as pd
from datetime import datetime, timedelta,datetime
import scipy.stats as stats
import math
from empyrical import max_drawdown, cagr, sharpe_ratio, annual_return, sortino_ratio
import numpy as np
import logging
from sklearn.cluster import KMeans
import random
import shutil
from typing import Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import zscore

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import re
import pickle
import statistics
import pdb

import empyrical as ep
from typing import Dict, Tuple
from pathlib import Path

# ==================   Import libraries for plotting   ========================
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates

# ==================    Define start time for computational time   ==================
import time
start_time = time.time()

# ================== Fundamental Factors ====================================
# VALUE_METRICES = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio',
#                 'dividendYield','priceToFreeCashFlowsRatio','enterpriseValueOverEBITDA','freeCashFlowYield']

# GROWTH_METRICES = ['revenueGrowth','epsgrowth','operatingIncomeGrowth','freeCashFlowGrowth','assetGrowth',
#                 'returnOnAssets','returnOnEquity','returnOnCapitalEmployed','operatingProfitMargin','netProfitMargin']

# CAT1_RATIOS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio','priceToFreeCashFlowsRatio',
#                 'enterpriseValueOverEBITDA','cashConversionCycle','debtToEquity','debtToAssets','netDebtToEBITDA']

# CAT2_RATIOS = ['dividendYield','freeCashFlowYield','revenueGrowth','epsgrowth',
#                 'operatingIncomeGrowth','freeCashFlowGrowth','assetGrowth','netProfitMargin','returnOnAssets','returnOnEquity',
#                 'returnOnCapitalEmployed','operatingProfitMargin','assetTurnover','inventoryTurnover','receivablesTurnover',
#                 'payablesTurnover','currentRatio','quickRatio','cashRatio','workingCapital','interestCoverage',]

GROUP1_FACTORS = ['priceToSalesRatio']       # Values are positive and lower value is preferable
GROUP2_FACTORS = ['priceToBookRatio','priceEarningsRatio','enterpriseValueOverEBITDA']    # Values can be negative and lower value is preferable 
GROUP3_FACTORS = ['dividendYield','revenueGrowth','epsgrowth','freeCashFlowGrowth']
GROUP4_FACTORS = ['returnOnEquity']
GROUP5_FACTORS = []

VALUE_METRICES = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio', 'dividendYield','enterpriseValueOverEBITDA']
GROWTH_METRICES = ['revenueGrowth','epsgrowth','freeCashFlowGrowth','returnOnEquity']
TARGET = ['er']

CAT1_RATIOS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio','enterpriseValueOverEBITDA']
CAT2_RATIOS = ['dividendYield','revenueGrowth','epsgrowth','freeCashFlowGrowth','returnOnEquity']

# ============================== Portfolio style ====================================
# Define the valid combinations
VALID_COMBINATIONS = {
    'score': ['min-max_avg', 'percentile_avg', 'z-score_avg', 'rank_avg', 'only-momentum'],
    'rating': ['value','growth','mom-value', 'mom-growth', 'growth_value', 'Avg_VGM','no-momentum']
}
# Define your valid options
RATING_OR_SCORE_OPTIONS = ['score', 'rating']
VGM_METHOD_OPTIONS = ['min-max_avg', 'percentile_avg', 'z-score_avg', 'only-momentum', 'rank_avg', 'value',
                        'growth', 'mom-value', 'mom-growth', 'growth_value', 'Avg_VGM', 'no-momentum']
# Define your portfolio style
RATING_OR_SCORE = 'score'
VGM_METHOD = 'z-score_avg'
SCALING_METHOD = 'no-scaling'                         # z-score   'min-max'  no-scaling

SCORE_FUNCTION = 'five_group'               # 'five_group'   or 'two_cat'
CONSIDER_RISK_FACTOR = 'no'
CONSIDER_DIVERSIFICATION = 'no'
DIV_METHOD = 'manual'               # 'KMeans' or 'manual'

# ===========================  Backtest time period definition  ==========================================
# global NAME_STOCK_WORLD
NAME_STOCK_WORLD = "RUSSELL-3000"
START_DATE =  '2020-01-01'    #   '2020-08-01'  a bug in this month in a perticular industry
END_DATE = '2024-12-31'
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE) 
FORMATION_DATE = pd.Timestamp.today().normalize()

# =========================== Define Directories: Data and Exp ========================================
READ_FROM_PICKLE = False
DEBUG = False
EXP_MODE = False
PRINT_FLAG = False
SAVE_EXCEL = False
SAVE_PLOT = True
PLOT_RANGE_FLAG = False
DO_PROCESS_FUNDA_DATA = True
PORTFOLIO_STOCK_WORLD = 'RUSSELL-3000'               #               'SP500, SP900, SP1500, RUSSELL-3000'
SP500_FILE = 's&p500_component_history.csv'
SP900_FILE = 's&p900_component_history.csv'
SP1500_FILE = 's&p1500_component_history.csv'
LARGE_CAP_THRESHOLD = 100
MID_CAP_THRESHOLD = 250
SMALL_CAP_THRESHOLD = 1000
MARKETCAP_THRESHOLD = 20000

# ============================   Portfolio variables   ==========================================
PRODUCTION_FLAG = False
EQUAL_WEIGHT_MOMENTUM = True
EQUAL_FACTOR_WEIGHT = False
CAP_FILTER = False
LARGE_CAP_FILTER = False
MID_CAP_FILTER = False
SMALL_CAP_FILTER =  False

PORTFOLIO_FREQ = 'Monthly'
N_TOP_TICKERS = 100
PORTFOLIO_SIZE = 30

COUNTRY = 'US'
TC = 0.001
INDICES_LIST = ['^NDX', '^GSPC', '^RUA']
MAJOR_INDEX = '^GSPC'
MARKETCAP_TH = 300000000

# ========================  Transaction Cost ================
WITH_TRANS_COST = True
if WITH_TRANS_COST:
    BROKER = 'other'                         # 'other'  or 'ibkr'
else:
    BROKER = 'NA'

# ========================== Define directories ========================================
DATA_DIR = '../../../Database/VGM_datasets/'
INDUSTRY_REGRESSION_DIR = "./Reg_Results/20241227_0533/"
PATH_DATA = "../FinRecipes/examples/Data_Download/data/Russell3000/"
PATH_DAILY_DATA = os.path.join(PATH_DATA, "df_tics_ohlcv_russell3000.h5")
PATH_MARKETCAP_DATA = os.path.join(PATH_DATA, "df_tics_marketcap_russell3000.h5")
PATH_SECTOR_DATA = os.path.join(PATH_DATA, "df_tics_sector_info_russell3000.h5")
PATH_FUNDA_DATA = os.path.join(PATH_DATA, "df_tics_funda_russell3000.h5")
REG_COEFF_VALUE = os.path.join(INDUSTRY_REGRESSION_DIR, 'value/rf_value_coeffs_df.xlsx')
REG_COEFF_GROWTH = os.path.join(INDUSTRY_REGRESSION_DIR, 'growth/rf_growth_coeffs_df.xlsx')
EQ_REG_COEFF_VALUE = os.path.join(INDUSTRY_REGRESSION_DIR, 'value/eq_rf_value_coeffs_df.xlsx')
EQ_REG_COEFF_GROWTH = os.path.join(INDUSTRY_REGRESSION_DIR, 'growth/eq_rf_growth_coeffs_df.xlsx')

ROOT_DIR = "./Exps"
HEAD_STR = datetime.now().strftime("%Y%m%d-%H%M")

if DEBUG:
    EXP_DIR = ROOT_DIR
elif EXP_MODE:
    EXP_DIR = ROOT_DIR
else:
    if RATING_OR_SCORE == 'rating':
        EXP_DIR = os.path.join(ROOT_DIR, f"{HEAD_STR}_{RATING_OR_SCORE}_{VGM_METHOD}_{CONSIDER_RISK_FACTOR}_{CONSIDER_DIVERSIFICATION}_{START_DATE.strftime('%Y-%m-%d')}_{N_TOP_TICKERS}_{PORTFOLIO_SIZE}_{PORTFOLIO_FREQ}_{COUNTRY}")

    if RATING_OR_SCORE == 'score':
        EXP_DIR = os.path.join(ROOT_DIR, f"{HEAD_STR}_{RATING_OR_SCORE}_{VGM_METHOD}_{CONSIDER_RISK_FACTOR}_{CONSIDER_DIVERSIFICATION}_{START_DATE.strftime('%Y-%m-%d')}_{N_TOP_TICKERS}_{PORTFOLIO_SIZE}_{PORTFOLIO_FREQ}_{COUNTRY}")

# create EXP_DIR if not exist
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

if SAVE_EXCEL:
    EXCEL_DIR = os.path.join(EXP_DIR, 'Excel')
    if not os.path.exists(EXCEL_DIR):
        os.makedirs(EXCEL_DIR)

# ============================ Momentum parameters ===============================================
MOMENTUM_PERIOD_LIST = [3, 6, 12]
GAP = 1
RISK_PERIOD_LIST = [1]
MOM_VOLATILITY_PERIOD = 1
RISK_VOLATILITY_PERIOD = 1
VOLATILITY_PERIOD = 1

# ============================ Configure logging ==============================
log_file = os.path.join(EXP_DIR, 'experiment.log')

# Check if log file exists, otherwise create one
if not os.path.exists(log_file):
    open(log_file, 'w').close()

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format without milliseconds

# ============== Validate the combination of rating_or_score and vgm_method ===========
def validate_combination(rating_or_score, vgm_method):
    # Check if the combination is valid
    if rating_or_score in VALID_COMBINATIONS and vgm_method in VALID_COMBINATIONS[rating_or_score]:
        if PRINT_FLAG:
            print(f"Valid combination: running '{vgm_method}' for '{rating_or_score}'.")
    else:
        print(f"Invalid combination: {rating_or_score} with {vgm_method}. Terminating...")
        sys.exit()  # Terminate the program if invalid combination

# ============================= Define all class and functions =======================
def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")
    except FileNotFoundError:
        print(f"Directory '{dir_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred: {e}")

def computation_time(start, message):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{message} {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
    return hours, minutes, seconds

def first_day_month_list(start_date, end_date):
    date = start_date.replace(day=1)  # Set the day to 1st of the start month
    date_list = []
    while date <= end_date:
        date_list.append(date)
        next_month = date.month + 1 if date.month < 12 else 1
        next_year = date.year if date.month < 12 else date.year + 1
        date = date.replace(year=next_year, month=next_month)
    return date_list

def weekday_dates_list(start_date, end_date, weekday = 4):
    date = start_date
    date_list = []
    while date <= end_date:
        if date.weekday() == weekday:
            date_list.append(date)
        date += timedelta(days=1)
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

    def filter_data(self, df_tics_daily):
        """Filter dataframe by date range and tickers.
        
        Args:
            df_tics_daily (pd.DataFrame): DataFrame with columns ['date', 'tic']
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Validate input
        required_cols = ['date', 'tic'] 
        if not all(col in df_tics_daily.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        # Update tickers list if empty
        if not self.tickers_list:
            self.tickers_list = df_tics_daily['tic'].unique().tolist()
            
        # Convert date once and create mask
        df_tics_daily.loc[:,'date'] = pd.to_datetime(df_tics_daily['date'])
        mask = ((df_tics_daily['date'] >= self.start_date) & 
                (df_tics_daily['date'] <= self.end_date) &
                (df_tics_daily['tic'].isin(self.tickers_list)))
                
        # Apply filter and sort in one operation
        return df_tics_daily[mask].sort_values(
            by=['date', 'tic']
        ).reset_index(drop=True)

    def clean_daily_data(self, df, missing_values_allowed=0.01, print_missing_values=False):
        # Create pivot table for close prices
        df_pivot = df.pivot(index='date', columns='tic', values='close')
        
        # Calculate missing values percentage for each ticker
        missing_pct = df_pivot.isnull().sum() / len(df_pivot)
        
        # Filter tickers based on missing value threshold
        valid_tickers = missing_pct[missing_pct <= missing_values_allowed].index
        self.removed_tickers_list = list(set(self.tickers_list) - set(valid_tickers))
        self.tickers_list = [str(ticker) for ticker in valid_tickers]   #list(valid_tickers)
        
        if print_missing_values:
            print("Missing values per ticker:")
            print(missing_pct[self.tickers_list])
        
        # Create multi-column pivot table for all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_pivot_all = df.pivot_table(
            index='date', 
            columns='tic',
            values=numeric_cols,
            aggfunc='first'
        )
        
        # Filter valid tickers and handle missing values
        df_pivot_all = df_pivot_all.loc[:, (slice(None), valid_tickers)]
        df_pivot_all = df_pivot_all.ffill().bfill()
        
        # Reshape back to long format
        df_clean = df_pivot_all.stack(level=1).reset_index()
        
        # Sort and filter date range
        df_clean = df_clean.sort_values(['date', 'tic'])
        mask = (df_clean['date'] >= self.start_date) & (df_clean['date'] <= self.end_date)
        df_clean = df_clean[mask].reset_index(drop=True)
        
        # Check for duplicates
        if df_clean.duplicated(['date', 'tic']).any():
            print('Duplicate test: there are duplicate rows.')
        
        return df_clean, self.tickers_list

    # def clean_daily_data(self, df, missing_values_allowed = 0.01, print_missing_values = False):
    #     uniqueDates = df['date'].unique()
    #     df_dates = pd.DataFrame(uniqueDates, columns=['date'])

    #     df_tics_daily_list = []
    #     updated_tickers_list = []
    #     for i, tic in enumerate(self.tickers_list):
    #         df_tic = df[df['tic'] == tic]
    #         df_tic = df_dates.merge(df_tic, on='date', how='left')
    #         df_tic['tic'] = tic

    #         if print_missing_values == True:
    #             print("No. of missing values before imputation for %5s = %5d"%(tic,df_tic['close'].isna().sum()))

    #         if df_tic['close'].isna().sum() <= missing_values_allowed * len(df_dates): 
    #             df_tic = df_tic.ffill().bfill()   
    #             df_tics_daily_list.append(df_tic)
    #             updated_tickers_list.append(tic)
    #         else:
    #             self.removed_tickers_list.append(tic)

    #     self.tickers_list = updated_tickers_list
    #     df_tics_daily = pd.concat(df_tics_daily_list)
    #     df_tics_daily = df_tics_daily.sort_values(by=['date', 'tic'],ignore_index=True)
    #     df_tics_daily.index = df_tics_daily.date.factorize()[0]
        
    #     start_date_idx = df_tics_daily[(df_tics_daily['date'] >= self.start_date)].index[0]
    #     end_date_idx = df_tics_daily[(df_tics_daily['date'] <= self.end_date)].index[-1] 
    #     df_tics_daily = df_tics_daily.reset_index(drop=True)
    #     df_tics_daily.index = df_tics_daily['date'].factorize()[0]

    #     if len(df_tics_daily[df_tics_daily.duplicated(keep=False)]) != 0:
    #         print('Duplicate test: there is duplicate rows.')

    #     return df_tics_daily

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
    portfolio_returns = pd.DataFrame(portfolio_returns,columns=['VGM Portfolio'])

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

def equal_factors_coefficients_US_v2():
    rf_value_coeffs_df = pd.read_excel(os.path.join(INDUSTRY_REGRESSION_DIR, EQ_REG_COEFF_VALUE))
    rf_value_coeffs_df_cleaned = rf_value_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'})
    rf_value_coeffs_df_cleaned.set_index('metrics', inplace=True)
    rf_value_coeffs_df_cleaned[:] = 1/len(rf_value_coeffs_df_cleaned)
    
    rf_growth_coeffs_df = pd.read_excel(os.path.join(INDUSTRY_REGRESSION_DIR, EQ_REG_COEFF_GROWTH))
    rf_growth_coeffs_df_cleaned = rf_growth_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'})
    rf_growth_coeffs_df_cleaned.set_index('metrics', inplace=True)
    rf_growth_coeffs_df_cleaned[:] = 1/len(rf_growth_coeffs_df_cleaned)

    # print(f'Number of Nan in rf_value_coeffs_df_cleaned = {rf_value_coeffs_df_cleaned.isna().sum().sum()}')   ## KK edited
    # print(f'Number of Nan in rf_growth_coeffs_df_cleaned = {rf_growth_coeffs_df_cleaned.isna().sum().sum()}')   ## KK edited

    value_coefficients = rf_value_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()
    growth_coefficients = rf_growth_coeffs_df_cleaned.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

    transformed_value_coefficients = transform_coefficients(value_coefficients)
    transformed_growth_coefficients = transform_coefficients(growth_coefficients)

    return transformed_value_coefficients, transformed_growth_coefficients

def transform_coefficients(coeffs_dict):
    transformed_coefficients = {}
    for metric, industries in coeffs_dict.items():
        for industry, value in industries.items():
            if industry not in transformed_coefficients:
                transformed_coefficients[industry] = {}
            transformed_coefficients[industry][metric] = value
    return transformed_coefficients

# def load_factors_coefficients_US_v2():
#     def clean_and_transform(file_path):
#         df = pd.read_excel(file_path).rename(columns={'Unnamed: 0': 'metrics'}).set_index('metrics')
#         return transform_coefficients(df.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict())

#     value_coefficients = clean_and_transform(os.path.join(INDUSTRY_REGRESSION_DIR, 'value/rf_value_coeffs_df.xlsx'))
#     growth_coefficients = clean_and_transform(os.path.join(INDUSTRY_REGRESSION_DIR, 'growth/rf_growth_coeffs_df.xlsx'))

#     return value_coefficients, growth_coefficients

# def load_factors_coefficients_US_v2():
#     rf_value_coeffs_df = pd.read_excel(REG_COEFF_VALUE)
#     rf_value_coeffs_df_cleaned = rf_value_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'})
#     rf_value_coeffs_df_cleaned.set_index('metrics', inplace=True)
#     # Transpose so industries are columns and metrics are rows
#     rf_value_coeffs_df_cleaned = rf_value_coeffs_df_cleaned.T
    
#     rf_growth_coeffs_df = pd.read_excel(REG_COEFF_GROWTH)
#     rf_growth_coeffs_df_cleaned = rf_growth_coeffs_df.rename(columns={'Unnamed: 0': 'metrics'})
#     rf_growth_coeffs_df_cleaned.set_index('metrics', inplace=True)
#     # Transpose so industries are columns and metrics are rows  
#     rf_growth_coeffs_df_cleaned = rf_growth_coeffs_df_cleaned.T

#     # Transform coefficients maintaining the same structure but with transposed data
#     value_coefficients = rf_value_coeffs_df_cleaned.apply(lambda col: col.dropna().to_dict(), axis=0).to_dict()
#     growth_coefficients = rf_growth_coeffs_df_cleaned.apply(lambda col: col.dropna().to_dict(), axis=0).to_dict()

#     return value_coefficients, growth_coefficients


def load_factors_coefficients_US_v2(
                                    value_path: str = REG_COEFF_VALUE,
                                    growth_path: str = REG_COEFF_GROWTH
                                    ) -> Tuple[Dict, Dict]:
    """Load and process value and growth factor coefficients efficiently.
    
    Returns:
        Tuple[Dict, Dict]: Value and growth coefficients dictionaries
    """
    logger = logging.getLogger(__name__)
    
    def process_coefficients(path: str) -> Dict:
        """Process coefficient file to dictionary."""
        try:
            return (pd.read_excel(path)
                   .rename(columns={'Unnamed: 0': 'metrics'})
                   .set_index('metrics')
                   .T
                   .apply(lambda col: col.dropna().to_dict(), axis=0)
                   .to_dict())
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
            raise
    
    try:
        logger.info("Loading factor coefficients")
        value_coefficients = process_coefficients(value_path)
        growth_coefficients = process_coefficients(growth_path)
        logger.info("Successfully loaded coefficients")
        return value_coefficients, growth_coefficients
        
    except Exception as e:
        logger.error(f"Failed to load coefficients: {str(e)}")
        raise

class Momentum_Score_v2:
    def __init__(self, formation_date: Union[str, datetime], risk_free_rate: float = 0) -> None:
        if not isinstance(risk_free_rate, (int, float)) or risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")
            
        try:
            self.end_date = pd.to_datetime(formation_date)
            self.start_date = self.end_date - pd.Timedelta(days=365 * MOM_VOLATILITY_PERIOD + 30)
            self.risk_free_rate = float(risk_free_rate)
            self.annualization_factor = np.sqrt(252)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid formation_date: {e}")

    def _calculate_momentum_for_period(self, df_pivot: pd.DataFrame, 
                                    daily_returns: pd.DataFrame,
                                    momentum_period: int,
                                    gap: int) -> Optional[pd.Series]:
        days_needed = 21 * momentum_period + gap
        lookback_days = 252 * MOM_VOLATILITY_PERIOD
        
        if len(df_pivot) < days_needed:
            return None
        
        volatility = self.annualization_factor * daily_returns.iloc[-lookback_days:].std()
        momentum = (df_pivot.iloc[-(1+gap)] / df_pivot.iloc[-days_needed] - 1)
        risk_adj_score = (momentum / volatility) - self.risk_free_rate
        
        return risk_adj_score.dropna()

    def compute_momentum_score(self, df_tics_daily: pd.DataFrame) -> pd.DataFrame:
        # Validate and filter data once
        required_cols = ['date', 'tic', 'close']
        if not all(col in df_tics_daily.columns for col in required_cols):
            raise ValueError(f"Missing columns: {required_cols}")
        
        # Create pivot table and daily returns once
        mask = (df_tics_daily['date'].between(self.start_date, self.end_date))
        df_pivot = pd.pivot_table(
            df_tics_daily[mask],
            index='date',
            columns='tic',
            values='close',
            aggfunc='first'
        ).sort_index()
        
        daily_returns = df_pivot.pct_change().dropna()
        
        # Calculate momentum scores in parallel
        momentum_data = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._calculate_momentum_for_period,
                    df_pivot,
                    daily_returns,
                    period,
                    GAP
                ): period for period in MOMENTUM_PERIOD_LIST
            }
            
            for future in as_completed(futures):
                period = futures[future]
                result = future.result()
                if result is not None:
                    momentum_data[f"momentum_{period}"] = stats.zscore(result)
        
        if not momentum_data:
            raise ValueError("No valid momentum scores calculated")
        
        # Create final DataFrame and calculate weighted score
        df_momentum = pd.DataFrame(momentum_data)
        weights = (np.ones(len(MOMENTUM_PERIOD_LIST)) / len(MOMENTUM_PERIOD_LIST) 
                  if EQUAL_WEIGHT_MOMENTUM 
                  else np.array([1/p for p in MOMENTUM_PERIOD_LIST]) / sum(1/p for p in MOMENTUM_PERIOD_LIST))
        
        df_momentum['momentum_score'] = stats.zscore(df_momentum.dot(weights))
        
        return df_momentum.sort_values('momentum_score', ascending=False)

# class Momentum_Score_v2:
#     """Calculate momentum scores for stocks with efficient date handling.
    
#     Args:
#         formation_date (Union[str, datetime]): The date for momentum calculation
#         risk_free_rate (float, optional): Risk-free rate for calculations. Defaults to 0.
#     """
#     def __init__(self, 
#                 formation_date: Union[str, datetime], 
#                 risk_free_rate: Optional[float] = 0) -> None:
#         # Validate risk_free_rate
#         if not isinstance(risk_free_rate, (int, float)) or risk_free_rate < 0:
#             raise ValueError("risk_free_rate must be a non-negative number")
            
#         # Efficient date handling
#         try:
#             self.formation_date = formation_date
#             self.end_date = pd.to_datetime(formation_date)
#             self.start_date = self.end_date - pd.Timedelta(days=365 * MOM_VOLATILITY_PERIOD)
#             self.risk_free_rate = float(risk_free_rate)
#         except (ValueError, TypeError) as e:
#             raise ValueError(f"Invalid formation_date format: {e}")

#     def kk_momentum_score(self, df_tics_daily, momentum_period=3, gap=1):
#         """Calculate momentum scores for stocks.
        
#         Args:
#             df_tics_daily (pd.DataFrame): DataFrame with columns ['date', 'tic', 'close']
#             momentum_period (int): Momentum calculation period
#             gap (int): Gap period for momentum calculation
        
#         Returns:
#             pd.DataFrame: DataFrame with momentum scores
#         """
#         # Input validation
#         required_cols = ['date', 'tic', 'close']
#         if not all(col in df_tics_daily.columns for col in required_cols):
#             raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
#         # Constants
#         T1 = momentum_period
#         days_needed = 21 * T1 + gap
#         lookback_days = 252 * MOM_VOLATILITY_PERIOD + 10
#         annualization_factor = np.sqrt(252)
        
#         # Date filtering - use loc for efficiency
#         date_mask = (df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] < self.end_date)
#         df_filtered = df_tics_daily.loc[date_mask].copy()
        
#         # Create pivot table efficiently
#         df_pivot = pd.pivot_table(
#             df_filtered,
#             index='date',
#             columns='tic',
#             values='close',
#             aggfunc='first'
#         ).sort_index()
        
#         # Check data sufficiency
#         if len(df_pivot) < days_needed:
#             print(f"Not enough data for momentum period {momentum_period}. Skipping this period.")
#             return None
        
#         # Calculate returns and volatility efficiently
#         daily_returns = df_pivot.pct_change()
#         volatility = annualization_factor * daily_returns.iloc[-lookback_days:].std()
        
#         # Calculate momentum score vectorially
#         end_prices = df_pivot.iloc[-(1+gap)]
#         start_prices = df_pivot.iloc[-days_needed]
#         momentum = (end_prices / start_prices) - 1
        
#         # Risk-adjusted momentum score
#         risk_adj_score = (momentum / volatility) - self.risk_free_rate
#         risk_adj_score = risk_adj_score.dropna()
        
#         # Calculate z-scores and create final DataFrame
#         momentum_zscore = stats.zscore(risk_adj_score)
#         result_df = pd.DataFrame({
#             'tic': risk_adj_score.index,
#             'Momentum_Score': momentum_zscore
#         })
        
#         return result_df

#     def compute_momentum_score(self, df_tics_daily):
#         """Compute momentum scores efficiently for multiple periods."""
        
#         # Pre-calculate momentum scores in parallel
#         with ThreadPoolExecutor() as executor:
#             futures = {
#                 executor.submit(
#                     self.kk_momentum_score, 
#                     df_tics_daily, 
#                     period, 
#                     GAP
#                 ): period for period in MOMENTUM_PERIOD_LIST
#             }
            
#             # Process results as they complete
#             momentum_data = {}
#             for future in as_completed(futures):
#                 period = futures[future]
#                 result = future.result()
#                 if result is not None:
#                     momentum_data[f"momentum_{period}"] = result.set_index('tic')['Momentum_Score']
    
#         if not momentum_data:
#             raise ValueError("No valid momentum scores calculated. Check your data and parameters.")
    
#         # Create DataFrame from collected data
#         df_momentum = pd.DataFrame(momentum_data)
        
#         # Calculate weights vectorially
#         if EQUAL_WEIGHT_MOMENTUM:
#             weights = np.ones(len(MOMENTUM_PERIOD_LIST)) / len(MOMENTUM_PERIOD_LIST)
#         else:
#             weights = np.array([1/period for period in MOMENTUM_PERIOD_LIST])
#             weights /= weights.sum()
        
#         # Calculate final momentum score efficiently
#         df_momentum['momentum_score'] = df_momentum.dot(weights)
        
#         # Normalize and sort in one operation
#         df_momentum['momentum_score'] = stats.zscore(df_momentum['momentum_score'])
        
#         return df_momentum.sort_values('momentum_score', ascending=False)


    # class Momentum_Score_updated:
    #     def __init__(self, formation_date, risk_free_rate=0):
    #         self.formation_date = formation_date
    #         self.end_date = pd.to_datetime(self.formation_date)
    #         self.start_date = self.end_date - timedelta(days = 365 * MOM_VOLATILITY_PERIOD)
    #         self.risk_free_rate = risk_free_rate

    # def kk_momentum_score(self, df_tics_daily, momentum_period=3, gap=1):
    #     T1 = momentum_period
    #     df_tics_daily = df_tics_daily[(df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] < self.end_date)].reset_index(drop=True)
    #     df_tics_daily_pivot = df_tics_daily.pivot(index='date', columns='tic', values='close')
        
    #     # Check if there are enough rows to perform the indexing operation
    #     if len(df_tics_daily_pivot) < (21 * T1 + gap):
    #         print(f"Not enough data for momentum period {momentum_period}. Skipping this period.")
    #         return None

    #     # sort df_tics_daily_pivot by date
    #     df_tics_daily_pivot = df_tics_daily_pivot.sort_index(ascending=True)

    #     all_daily_return = df_tics_daily_pivot.pct_change().dropna()
    #     std_daily_return = np.sqrt(252) * all_daily_return.iloc[-252*MOM_VOLATILITY_PERIOD:].std()
    #     risk_adj_mom_score = (((df_tics_daily_pivot.iloc[-(1+gap)] / df_tics_daily_pivot.iloc[-21 * T1 - gap]) - 1) / std_daily_return) - self.risk_free_rate
    #     risk_adj_mom_score = risk_adj_mom_score.dropna()
    #     momentum_zscore = stats.zscore(risk_adj_mom_score)
    #     df_tics_momentum_zscore = pd.DataFrame(momentum_zscore, columns=['Momentum_Score']).reset_index()
    #     return df_tics_momentum_zscore

    # def compute_momentum_score(self, df_tics_daily):
    #     momentum_columns = []
    #     momentum_scores = []

    #     for period in MOMENTUM_PERIOD_LIST:
    #         key = f"momentum_{period}"
    #         df_momentum_score = self.kk_momentum_score(df_tics_daily, momentum_period=period, gap=GAP)
    #         if df_momentum_score is not None:
    #             df_momentum_score = df_momentum_score.rename(columns={"Momentum_Score": key})
    #             momentum_scores.append(df_momentum_score[['tic', key]])
    #             momentum_columns.append(key)

    #     if not momentum_scores:
    #         raise ValueError("No valid momentum scores calculated. Check your data and parameters.")

    #     # Concatenate all momentum scores into a single DataFrame
    #     df_momentum = pd.concat(momentum_scores, axis=1)

    #     # Remove duplicate 'tic' columns
    #     df_momentum = df_momentum.loc[:, ~df_momentum.columns.duplicated()]

    #     momentum_weights = np.array([1 / period for period in MOMENTUM_PERIOD_LIST])
    #     momentum_weights = momentum_weights / momentum_weights.sum()

    #     # Calculate the mean momentum score
    #     if EQUAL_WEIGHT_MOMENTUM:
    #         df_momentum['momentum_score'] = df_momentum[momentum_columns].mean(axis=1)
    #     else:
    #         df_momentum['momentum_score'] = df_momentum[momentum_columns].dot(momentum_weights)

    #     # Normalize the momentum score using z-score
    #     df_momentum['momentum_score'] = stats.zscore(df_momentum['momentum_score'])

    #     # Sort by momentum score in descending order
    #     df_momentum = df_momentum.sort_values(by='momentum_score', ascending=False)

    #     return df_momentum

def process_funda_data(df_funda, df_sector):
    df_funda = df_funda.rename(columns= {'tic': 'symbol'})    #[['tic', 'quarterStartDate']]
    # Remove duplicated columns
    df_funda = df_funda.loc[:, ~df_funda.columns.duplicated()]

    # Convert string columns to categorical
    # str_columns = ['symbol', 'Industry', 'Sector', 'time']
    # for col in str_columns:
    #     if col in df_funda.columns:
    #         df_funda[col] = df_funda[col].astype('category')
    
    current_date = pd.Timestamp.now().date()
    two_years_ago = current_date - pd.DateOffset(years=6)

    # if 'date' in df_funda.columns:
    #     df_funda['date'] = pd.to_datetime(df_funda['date'], errors='coerce')
    #     date_mask = (df_funda['date'].dt.date.between(two_years_ago.date(), current_date))
    #     df_funda = df_funda.loc[date_mask]


    # Convert columns to numeric
    numeric_cols = ['enterpriseValueOverEBITDA','freeCashFlowYield','debtToEquity','debtToAssets','netDebtToEBITDA','interestCoverage','workingCapital',
                        'priceToBookRatio', 'priceToSalesRatio', 'priceEarningsRatio', 'netProfitMargin', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed',
                        'operatingProfitMargin', 'currentRatio', 'quickRatio', 'cashRatio', 'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
                        'payablesTurnover', 'cashConversionCycle', 'dividendPayoutRatio', 'dividendYield', 'priceToFreeCashFlowsRatio', 'eps', 'incomeBeforeTax',
                        'totalAssets', 'revenueGrowth', 'epsgrowth', 'operatingIncomeGrowth', 'freeCashFlowGrowth', 'assetGrowth',
                        'epsEstimated', 'revenue', 'revenueEstimated']
    
    # # Convert in chunks to reduce memory usage
    # chunk_size = 10
    # for i in range(0, len(numeric_cols), chunk_size):
    #     chunk_cols = numeric_cols[i:i + chunk_size]
    #     for col in chunk_cols:
    #         if col in df_funda.columns:
    #             df_funda[col] = pd.to_numeric(df_funda[col], errors='coerce', downcast='float')
    

    # 4. Calculate earnings ratio efficiently
    if {'incomeBeforeTax', 'totalAssets'}.issubset(df_funda.columns):
        mask = (df_funda['totalAssets'].notna() & (df_funda['totalAssets'] != 0))
        df_funda.loc[mask, 'er'] = np.divide(
            df_funda.loc[mask, 'incomeBeforeTax'],
            df_funda.loc[mask, 'totalAssets']
        )

    # 5. Merge with sector data efficiently
    df_sector_slim = df_sector[['Ticker', 'Sector', 'Industry']].copy()
    # df_sector_slim['Industry'] = df_sector_slim['Industry'].astype('category')
    df_funda = pd.merge(df_funda,df_sector_slim,left_on='symbol',right_on='Ticker',how='inner')
    df_funda.drop(columns=['Ticker'], inplace=True)

    # 6. Process release dates efficiently
    if 'releaseDate' in df_funda.columns:
        if 'date' in df_funda.columns:
            df_funda = df_funda.drop(columns=['date'])
        
        # Handle release dates
        mask = df_funda['releaseDate'].notna()
        df_funda.loc[mask, 'releaseDate'] = pd.to_datetime(df_funda.loc[mask, 'releaseDate'])
        
        # Handle AMC dates
        df_funda['time'] = df_funda['time'].fillna('amc')
        amc_mask = df_funda['time'].str.contains('amc', case=False, na=False)
        df_funda.loc[amc_mask, 'releaseDate'] += pd.Timedelta(days=1)
        
        # Rename to standard column
        df_funda = df_funda.rename(columns={'releaseDate': 'date'})
    
    # 7. Group by industry efficiently
    # industry_dfs = dict(tuple(df_funda.groupby('Industry')))
    # industry_dfs = {k: v for k, v in industry_dfs.items() if k != ''}

    industry_dfs = df_funda.copy()
    
    # 8. Save to pickle efficiently

    # Save to HDF5
    if READ_FROM_PICKLE:
        with open(os.path.join(DATA_DIR, 'industry_dfs_US.pkl'), 'wb') as f:
            pickle.dump(industry_dfs, f, protocol=4)  # Use protocol 4 for better performance

    else:
        industry_dfs.to_hdf(os.path.join(DATA_DIR, 'industry_dfs_US.h5'), key='df', mode='w')

        # with pd.HDFStore(os.path.join(DATA_DIR, 'industry_dfs_US.h5')) as store:
        #     for key, df in industry_dfs.items():
        #         store.put(f'industry_{key}', df, format='table')        
    

    return industry_dfs

# def process_funda_data(df_funda, df_sector):
#     # Remove duplicated columns
#     df_funda = df_funda.loc[:, ~df_funda.columns.duplicated()]
#     current_date = datetime.now().date()
#     if 'date' in df_funda.columns:
#         df_funda['date'] = pd.to_datetime(df_funda['date'], errors='coerce')
#         two_years_ago = current_date - pd.DateOffset(years=6)                     # filter last 2 years of data
#         df_funda = df_funda[(df_funda['date'].dt.date <= current_date) & (df_funda['date'].dt.date >= two_years_ago.date())]  # filter last 2 year data

#     # Calculate 'er' (incomeBeforeTax / totalAssets) if totalAssets is not None and not zero
#     if 'incomeBeforeTax' in df_funda.columns and 'totalAssets' in df_funda.columns:
#         mask = (df_funda['totalAssets'].notnull()) & (df_funda['totalAssets'] != 0)
#         df_funda.loc[mask, 'er'] = df_funda.loc[mask, 'incomeBeforeTax'] / df_funda.loc[mask, 'totalAssets']

#     # Convert columns to numeric
#     columns_to_convert = ['enterpriseValueOverEBITDA','freeCashFlowYield','debtToEquity','debtToAssets','netDebtToEBITDA','interestCoverage','workingCapital',
#                         'priceToBookRatio', 'priceToSalesRatio', 'priceEarningsRatio', 'netProfitMargin', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed',
#                         'operatingProfitMargin', 'currentRatio', 'quickRatio', 'cashRatio', 'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
#                         'payablesTurnover', 'cashConversionCycle', 'dividendPayoutRatio', 'dividendYield', 'priceToFreeCashFlowsRatio', 'eps', 'incomeBeforeTax',
#                         'totalAssets', 'revenueGrowth', 'epsgrowth', 'operatingIncomeGrowth', 'freeCashFlowGrowth', 'assetGrowth',
#                         'epsEstimated', 'revenue', 'revenueEstimated']

#     for column in columns_to_convert:
#         if column in df_funda.columns:
#             df_funda[column] = pd.to_numeric(df_funda[column], errors='coerce')

#     ##=========================================***==============================================================
#     # Merge combined data with industry information
#     df_funda = df_funda.merge(df_sector[['Ticker', 'Sector', 'Industry']], left_on='symbol', right_on='Ticker', how='left')

#     # Drop duplicate 'Symbol' column
#     df_funda = df_funda.drop(columns=['Ticker'])
#     if SAVE_EXCEL:
#         df_funda.to_excel(os.path.join(DATA_DIR, 'All_Fundamental_US.xlsx'), index=False)
    
#     # Group by industry and save to separate Excel files
#     grouped = df_funda.groupby('Industry')

#     # Replace invalid characters in the industry name with an underscore
#     def sanitize_filename(name):
#         if not name:  # Handle empty industry name
#             return "Unknown_Industry"
#         return re.sub(r'[\\/*?:"<>|]', "_", name)

#     # # Group by industry and save to separate Excel files
#     # output_dir = './drive/MyDrive/VGM_US/industry_data_rs3000/'
#     # os.makedirs(output_dir, exist_ok=True)

#     industry_dfs = {}
#     for industry, group in grouped:
#         if industry != '':
#             # sanitized_industry_name = sanitize_filename(industry)
#             # filename = f'{sanitized_industry_name}.xlsx'
#             # filepath = os.path.join(output_dir, filename)

#             try:

#                 #=========== included to take care of earning dates ====================
#                 # group = group.dropna(subset=['dateAccepted'])
#                 # group['dateAccepted'] = pd.to_datetime(group['dateAccepted'])
#                 # group['time'] = group['time'].fillna('amc')
#                 # group.loc[group['time'].str.contains('amc', case=False), 'dateAccepted'] += pd.Timedelta(days=1)   # Increment 'dateAccepted' by one day if 'time' contains 'amc'
#                 # group = group.drop(columns=['date'])
#                 # group = group.rename(columns={'dateAccepted': 'date'})

#                 group = group.dropna(subset=['releaseDate'])
#                 group['releaseDate'] = pd.to_datetime(group['releaseDate'])
#                 group['time'] = group['time'].fillna('amc')
#                 group.loc[group['time'].str.contains('amc', case=False), 'releaseDate'] += pd.Timedelta(days=1)   # Increment 'dateAccepted' by one day if 'time' contains 'amc'
#                 group = group.drop(columns=['date'])
#                 group = group.rename(columns={'releaseDate': 'date'})

#                 industry_dfs[industry] = group
#                 # group.to_excel(filepath, index=False)
#                 # print(f"Saved data for industry: {industry}")
#             except Exception as e:
#                 print(f"Failed to save data for industry {industry}: {e}")

#     with open(os.path.join(DATA_DIR, 'industry_dfs_US.pkl'), 'wb') as f:
#         pickle.dump(industry_dfs, f)
#     # print("Processing complete.")

#     return industry_dfs

def scale_group(x, df, group_number):
    """
    Rank values into deciles (1-10) based on group characteristics
    Args:
        x: value to rank
        df: series containing all values
        group_number: 1,2,3 determining ranking behavior
    Returns:
        int: decile rank 1-10 (10=best)
    """
    if pd.isna(x):
        return 0
        
    if group_number == 1:
        # Handle negative values - assign worst rank (10)
        if x <= 0:
            return 10
            
        # Get positive values only
        valid_values = df[df > 0]
        if len(valid_values) < 2:
            return 10
            
        try:
            # Calculate decile ranks (1-10)
            rank = pd.qcut(valid_values, q=10, labels=False, duplicates='drop')
            # Return rank directly (lower values get lower/better ranks)
            return rank[df == x].iloc[0] + 1
        except:
            return 10
            
    elif group_number == 2:
        try:
            # Calculate decile ranks (1-10)
            rank = pd.qcut(df, q=10, labels=False, duplicates='drop')
            # Return rank directly (lower values get lower/better ranks)
            return rank[df == x].iloc[0] + 1
        except:
            return 10
            
    elif group_number == 3:
        try:
            # Calculate decile ranks (1-10)
            rank = pd.qcut(df, q=10, labels=False, duplicates='drop')
            # Reverse ranks so higher values get better ranks (1)
            return 10 - rank[df == x].iloc[0]
        except:
            return 10

    elif group_number == 4:
        # Handle negative values - assign worst rank (10)
        if x <= 0:
            return 10
            
        # Get positive values only
        valid_values = df[df > 0]
        if len(valid_values) < 2:
            return 10
            
        try:
            # Calculate decile ranks for positive values (1-9)
            rank = pd.qcut(valid_values, q=9, labels=False, duplicates='drop')
            # Reverse ranks so higher positive values get better ranks (1)
            return 9 - rank[df == x].iloc[0]
        except:
            return 10
            
    else:
        raise ValueError("group_number must be between 1 and 3")

# def compute_group_ranks(df_score, group_factors, group_num):
#     """Compute ranks for an entire group at once"""
#     result = df_score.copy()
    
#     for col in group_factors:
#         if col not in df_score.columns:
#             continue
            
#         series = df_score[col]
        
#         if group_num == 1:
#             # Handle negative values
#             mask = series <= 0
#             valid_values = series[~mask]
            
#             if len(valid_values) >= 2:
#                 ranks = pd.qcut(valid_values, q=10, labels=False, duplicates='drop')
#                 result.loc[~mask, col] = ranks + 1
#                 result.loc[mask, col] = 10
            
#         elif group_num == 2:
#             # Direct ranking for all values
#             result[col] = pd.qcut(series, q=10, labels=False, duplicates='drop') + 1
            
#         elif group_num == 3:
#             # Reverse ranking for all values 
#             result[col] = 10 - pd.qcut(series, q=10, labels=False, duplicates='drop')
            
#         elif group_num == 4:
#             # Handle negative values with reverse ranking for positives
#             mask = series <= 0
#             valid_values = series[~mask]
            
#             if len(valid_values) >= 2:
#                 ranks = pd.qcut(valid_values, q=9, labels=False, duplicates='drop')
#                 result.loc[~mask, col] = 9 - ranks
#                 result.loc[mask, col] = 10
                
#     return result

# def scale_group_parallel(df_score):
#     """Process all groups in parallel"""
#     with ThreadPoolExecutor() as executor:
#         futures = [
#             executor.submit(compute_group_ranks, df_score, GROUP1_FACTORS, 1),
#             executor.submit(compute_group_ranks, df_score, GROUP2_FACTORS, 2),
#             executor.submit(compute_group_ranks, df_score, GROUP3_FACTORS, 3),
#             executor.submit(compute_group_ranks, df_score, GROUP4_FACTORS, 4)
#         ]
        
#         results = [f.result() for f in futures]
        
#     # Combine results
#     final_result = df_score.copy()
#     for result in results:
#         for col in result.columns:
#             if col in final_result.columns:
#                 final_result[col] = result[col]
                
#     return final_result

def compute_group_ranks(df_score, group_factors, group_num):
    """Compute ranks for group columns that exist in df_score"""
    result = pd.DataFrame(index=df_score.index)
    
    # Only process columns that exist in df_score
    valid_cols = [col for col in group_factors if col in df_score.columns]
    
    for col in valid_cols:
        series = df_score[col]
        result[col] = pd.Series(index=df_score.index)
        
        if group_num == 1:
            result[col] = pd.qcut(series, q=10, labels=False, duplicates='drop') + 1
                
        elif group_num == 2:
            mask = series <= 0
            valid_values = series[~mask]
            if len(valid_values) >= 2:
                ranks = pd.qcut(valid_values, q = 9, labels=False, duplicates='drop')
                result.loc[~mask, col] = ranks + 1
                result.loc[mask, col] = 10
           
        elif group_num == 3:
            result[col] = 10 - pd.qcut(series, q=10, labels=False, duplicates='drop')
            # result[col] = pd.qcut(series, 10, labels=range(10, 0, -1), duplicates='drop')
            
        elif group_num == 4:
            mask = series <= 0
            valid_values = series[~mask]
            if len(valid_values) >= 2:
                ranks = pd.qcut(valid_values, q=9, labels=False, duplicates='drop')
                result.loc[~mask, col] = 9 - ranks
                result.loc[mask, col] = 10
                
    return result

def scale_group_parallel(df_score):
    final_result = df_score.copy()
    
    # Process each group
    group_ranks = []
    for group_num, factors in enumerate([GROUP1_FACTORS, GROUP2_FACTORS, GROUP3_FACTORS, GROUP4_FACTORS], 1):
        ranks = compute_group_ranks(df_score, factors, group_num)
        group_ranks.append(ranks)
    
    # Combine all results
    for ranks in group_ranks:
        for col in ranks.columns:
            final_result[col] = ranks[col]
            
    return final_result

def scale_group_min_max_range(x, df, group_number):
    # Ensure group_number is between 1 and 5
    if group_number == 1:
        if x < 0:
            return -1  # For non-positive values
        else:
            min_val = df['priceToBookRatio'][df['priceToBookRatio'] > 0].min()
            max_val = df['priceToBookRatio'][df['priceToBookRatio'] > 0].max()
            normalized_value = (x - min_val) / (max_val - min_val)
            return -normalized_value + 1

    elif group_number == 2:
        min_val = df['priceToSalesRatio'].min()
        max_val = df['priceToSalesRatio'].max()
        normalized_value = (x - min_val) / (max_val - min_val)
        return -normalized_value + 1

    elif group_number == 3:
        min_val = df['priceToSalesRatio'].min()
        max_val = df['priceToSalesRatio'].max()
        normalized_value = (x - min_val) / (max_val - min_val)
        return normalized_value

    elif group_number == 4:
        if x <= 0:
            min_val = df['priceToSalesRatio'][df['priceToSalesRatio'] <= 0].min()
            max_val = df['priceToSalesRatio'][df['priceToSalesRatio'] <= 0].max()
            normalized_value = ((x - min_val) / (max_val - min_val)) * (-1)
            return normalized_value
        else:
            min_val = df['priceToSalesRatio'][df['priceToSalesRatio'] > 0].min()
            max_val = df['priceToSalesRatio'][df['priceToSalesRatio'] > 0].max()
            normalized_value = (x - min_val) / (max_val - min_val)
            return normalized_value

    elif group_number == 5:
        if x <= 0:
            return -1  # For non-positive values
        else:
            min_val = df['priceToBookRatio'][df['priceToBookRatio'] > 0].min()
            max_val = df['priceToBookRatio'][df['priceToBookRatio'] > 0].max()
            normalized_value = (x - min_val) / (max_val - min_val)
            return normalized_value

    else:
        raise ValueError("group_number must be between 1 and 5")

# def score(values, value, cat) -> int:
#     try:
#         std = statistics.stdev(values)
#         mean = statistics.mean(values)
#     except statistics.StatisticsError:
#         # Handle cases where stdev or mean calculation fails (e.g., insufficient data points)
#         return 0
    
#     # if mean < 0:
#     #    return 0
#     if cat == 1:
#         if value < 0:
#             return 0
#         if (mean + (-1 * std)) < value <= mean:
#             return 1
#         elif (mean + (-2 * std)) < value <= (mean + (-1 * std)):
#             return 2
#         elif value <= (mean + (-2 * std)):
#             return 3
#         elif mean < value <= (mean + (1 * std)):
#             return -1
#         elif (mean + (1 * std)) < value <= (mean + (2 * std)):
#             return -2
#         else:
#             return -3
#     else:
#         if value < 0:
#             return 0
#         if mean <= value < (mean + (1 * std)):
#             return 1
#         elif (mean + (1 * std)) <= value < (mean + (2  *std)):
#             return 2
#         elif value >= (mean + (2 * std)):
#             return 3
#         elif (mean + (-1 * std)) <= value < mean:
#             return -1
#         elif (mean + (-2 * std)) <= value < (mean + (-1 * std)):
#             return -2
#         else:
#             return -3

def score(series, value, category):
    """
    Score values within industry using deciles
    Args:
        series: Series containing factor values
        value: Current value to score
        category: 1 for lower-is-better, 2 for higher-is-better
    Returns:
        int: Score from 1-10
    """
    if value <= 0:
        return 0
        
    valid_values = series[series > 0]
    if len(valid_values) < 2:
        return 0
        
    try:
        # Calculate decile rank
        deciles = pd.qcut(valid_values, q=10, labels=False, duplicates='drop')
        rank = deciles[series == value].iloc[0] + 1
        
        # Reverse ranks for category 1
        if category == 1:
            rank = 11 - rank
            
        return rank
        
    except Exception:
        return 0


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


def process_industry_dataframe(df, formation_date, industry):
    # latest_data = df.sort_values(by='date', ascending=False).groupby('symbol').first().reset_index()
    # required_columns = ['symbol', 'date'] + VALUE_METRICES + GROWTH_METRICES
    # logging.info(f"Columns in required for V and G scores = {required_columns}")
    
    # df_filtered = latest_data[required_columns]
    # df_filtered = df_filtered.dropna()

    # for col in df_filtered.columns:
    #     if col not in ['symbol', 'date']:
    #         df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # print(f"{industry} NaN value count = {df.isna().sum().sum()}")

    if df.isna().any().any():
        raise ValueError("DataFrame contains NaN values. Please clean the data before proceeding.")

    
    # df.fillna(df.median(numeric_only=True), inplace=True)               # questionable (we may drop the na row)

    # # Check for NaN values
    # if df.isna().any().any():
    #     print("DataFrame contains NaN values. Pausing for debugging...")
    #     pdb.set_trace()  # Debugger starts here
    # else:
    #     print("No NaN values found. Continuing execution.")

    if SAVE_EXCEL:
        df.to_excel(os.path.join(DATA_DIR, f'Industry_Data_Recent/{industry}_{formation_date.strftime("%Y-%m-%d")}.xlsx'), index=False)

    df_score = df.copy()

    # for col in CAT1_RATIOS:
    #     if col in df_score.columns:
    #         df_score[f'{col}_rank'] = df_score[col].apply(lambda x: score_factor(df_score[col], x, 1))

    # for col in CAT2_RATIOS:
    #     if col in df_score.columns:
    #         df_score[f'{col}_rank'] = df_score[col].apply(lambda x: score_factor(df_score[col], x, 2))

    # for col in CAT1_RATIOS:
    #     if col in df_score.columns:
    #         df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 1))

    # for col in CAT2_RATIOS:
    #     if col in df_score.columns:
    #         df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 2))


    # if industry =='REIT - Hotel & Motel':
    #     print('weight')

    if SCORE_FUNCTION == 'two_cat':
        for col in CAT1_RATIOS:
            if col in df_score.columns:
                df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 1))

        for col in CAT2_RATIOS:
            if col in df_score.columns:
                df_score[col] = df_score[col].apply(lambda x: score(df_score[col], x, 2))

    elif SCORE_FUNCTION == 'five_group':
        df_score = scale_group_parallel(df)


        # for col in GROUP1_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df[col].apply(lambda x: scale_group(x, df[col], 1))

        # for col in GROUP2_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df[col].apply(lambda x: scale_group(x, df[col], 2))

        # for col in GROUP3_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df[col].apply(lambda x: scale_group(x, df[col], 3))

        # for col in GROUP4_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df[col].apply(lambda x: scale_group(x, df[col], 4))

        # for col in GROUP5_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df[col].apply(lambda x: scale_group(x, df[col], 5))

        # for col in GROUP1_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df_score[col].apply(lambda x: scale_group(x, df_score[col], 1))

        # for col in GROUP2_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df_score[col].apply(lambda x: scale_group(x, df_score[col], 2))

        # for col in GROUP3_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df_score[col].apply(lambda x: scale_group(x, df_score[col], 3))

        # for col in GROUP4_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df_score[col].apply(lambda x: scale_group(x, df_score[col], 4))

        # for col in GROUP5_FACTORS:
        #     if col in df.columns:
        #         df_score[col] = df_score[col].apply(lambda x: scale_group(x, df_score[col], 5))



    logging.info(f"Columns after process industry factors = {df_score.columns.to_list()}")
    return df_score

def calculate_scores(
                        df_score: pd.DataFrame, 
                        transformed_coefficients: Dict[str, Dict[str, float]], 
                        industry: str, 
                        score_type: str
                    ) -> pd.Series:
    """
    Generalized function to calculate scores (value or growth) for a given industry.
    
    Args:
        df_score (pd.DataFrame): DataFrame containing the scores.
        transformed_coefficients (Dict[str, Dict[str, float]]): Dictionary of coefficients.
        industry (str): Industry for which to calculate the scores.
        score_type (str): Type of score to calculate ('value' or 'growth').
    
    Returns:
        pd.Series: Series containing the calculated scores.
    """
    logger = logging.getLogger(__name__)
    coefficients = transformed_coefficients.get(industry, None)
    
    if coefficients is None:
        logger.error(f"Industry {industry} not found in coefficients.")
        return pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
    # Initialize score series with zeros
    score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
    # Filter only the relevant columns from df_score
    relevant_metrics = [metric for metric in coefficients if metric in df_score.columns]
    
    if not relevant_metrics:
        logger.warning(f"No relevant metrics found in DataFrame for industry {industry}.")
        return score
    
    # Calculate the score using vectorized operations
    score = df_score[relevant_metrics].mul(pd.Series(coefficients)).sum(axis=1)
    
    # Log missing metrics
    missing_metrics = set(coefficients) - set(relevant_metrics)
    for metric in missing_metrics:
        logger.warning(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
    logger.info(f"Successfully calculated {score_type} scores for industry {industry}.")
    return score

# Example usage:
# value_scores = calculate_scores(df_score, transformed_coefficients, industry, 'value')
# growth_scores = calculate_scores(df_score, transformed_coefficients, industry, 'growth')

# def calculate_scores(
#     df_score: pd.DataFrame, 
#     transformed_coefficients: Dict[str, Dict[str, float]], 
#     industry: str, 
#     score_type: str
# ) -> pd.Series:
#     """
#     Generalized function to calculate scores (value or growth) for a given industry.
    
#     Args:
#         df_score (pd.DataFrame): DataFrame containing the scores.
#         transformed_coefficients (Dict[str, Dict[str, float]]): Dictionary of coefficients.
#         industry (str): Industry for which to calculate the scores.
#         score_type (str): Type of score to calculate ('value' or 'growth').
    
#     Returns:
#         pd.Series: Series containing the calculated scores.
#     """
#     logger = logging.getLogger(__name__)
#     coefficients = transformed_coefficients.get(industry, None)
    
#     if coefficients is None:
#         logger.error(f"Industry {industry} not found in coefficients.")
#         return pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
#     score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
#     for metric, weight in coefficients.items():
#         if metric in df_score.columns:
#             score += df_score[metric] * weight
#         else:
#             logger.warning(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
#     logger.info(f"Successfully calculated {score_type} scores for industry {industry}.")
#     return score

# Example usage:
# value_scores = calculate_scores(df_score, transformed_coefficients, industry, 'value')
# growth_scores = calculate_scores(df_score, transformed_coefficients, industry, 'growth')

# def calculate_value_scores(df_score, transformed_coefficients, industry):
#     coefficients = transformed_coefficients.get(industry, {})
#     value_score = sum(df_score[metric] * weight for metric, weight in coefficients.items() if metric in df_score.columns)
#     return value_score

# Define the existing functions
# def calculate_value_scores(df_score, transformed_coefficients, industry):
#     coefficients = transformed_coefficients.get(industry, None)
#     if coefficients is None:
#         print(f"Industry {industry} not found in coefficients.")
#         logging.info(f"Industry {industry} not found in coefficients.")
#         return pd.Series(np.zeros(len(df_score)), index=df_score.index)
#     value_score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
#     for metric, weight in coefficients.items():
#         if metric in df_score.columns:
#             value_score += df_score[metric] * weight
#         else:
#             print(f"Metric {metric} not found in DataFrame for industry {industry}.")
#             logging.info(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
#     return value_score

# # def calculate_growth_scores(df_score, transformed_coefficients, industry):
# #     coefficients = transformed_coefficients.get(industry, {})
# #     growth_score = sum(df_score[metric] * weight for metric, weight in coefficients.items() if metric in df_score.columns)
# #     return growth_score

# def calculate_growth_scores(df_score, transformed_coefficients, industry):
#     coefficients = transformed_coefficients.get(industry, None)
#     if coefficients is None:
#         print(f"Industry {industry} not found in coefficients.")
#         logging.info(f"Industry {industry} not found in coefficients.")
#         return pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
#     growth_score = pd.Series(np.zeros(len(df_score)), index=df_score.index)
    
#     for metric, weight in coefficients.items():
#         if metric in df_score.columns:
#             growth_score += df_score[metric] * weight
#         else:
#             print(f"Metric {metric} not found in DataFrame for industry {industry}.")
#             logging.info(f"Metric {metric} not found in DataFrame for industry {industry}.")
    
#     return growth_score

def rank_scores(df, score_column):
    try:
        # Adjust the quantile labels so that the highest scores get rank 10
        df[f'{score_column}_rank'] = pd.qcut(df[score_column], 10, labels=False, duplicates='drop') + 1
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

def compute_value_growth_score_US(df_funda, df_sector, formation_date, value_coeff, growth_coeff):
    industry_dfs = df_funda.copy()
    # industry_dfs['date'] = pd.to_datetime(industry_dfs['date'])
    industry_dfs = industry_dfs[(industry_dfs['date'] > formation_date - timedelta(days=365)) & (industry_dfs['date'] < formation_date)]

    df_industry_scores = pd.DataFrame([])
    sum_i = 0

    industry_name_list = industry_dfs['Industry'].unique().tolist()
    industry_name_list = [name for name in industry_name_list if name and name.strip()]
    industry_name_list.sort()

    for industry in industry_name_list:
        df = industry_dfs[industry_dfs['Industry'] == industry]
        if industry not in value_coeff or industry not in growth_coeff:
            # print(f"Skipping industry: {industry} as it does not have regression coefficients.")
            logging.info(f"Skipping industry: {industry} as it does not have regression coefficients.")
            continue

        if industry == 'REIT - Hotel & Motel':
            continue

        latest_data = df.sort_values(by='date', ascending=False).groupby('tic').first().reset_index()
        required_columns = ['tic', 'date'] + VALUE_METRICES + GROWTH_METRICES
        # logging.info(f"Columns in required for V and G scores = {required_columns}")
        
        latest_data = latest_data[required_columns]
        latest_data = latest_data.dropna()

        if (len(latest_data['tic'].unique()) >= 7):
            # df['date'] = pd.to_datetime(df['date'])
            # df = df[df['date'] < pd.to_datetime(formation_date)]    # changed <= to <    on 13/08/2024
            if PRINT_FLAG:
                print(f"Industry: {industry}, Number of tickers: {len(df['tic'].unique())}, Data Shape: {df.shape}")
            logging.info(f"Industry: {industry}, Number of tickers: {len(df['tic'].unique())}, Data Shape: {df.shape}")

            # df_score = process_industry_dataframe(df, formation_date, industry)
            df_score = process_industry_dataframe(latest_data, formation_date, industry)
            
            # sum_i = sum_i + len(df_score)
            # print(f"Scores are computed for industry {industry}")
            df_score['industry'] = industry
            df_score['sector']  = df['Sector'].unique()[0]

            # Calculate value and growth scores
            df_score['value_score'] = calculate_scores(df_score, value_coeff, industry, 'value')
            df_score['growth_score'] = calculate_scores(df_score, growth_coeff, industry, 'growth')
                
            try:
                if SCALING_METHOD == 'z-score':
                    df_score['value_score'] = stats.zscore(df_score['value_score'])
                    df_score['growth_score'] = stats.zscore(df_score['growth_score'])
                    logging.info(f"Applied z-score scaling to value and growth scores for industry {industry}.")
                
                elif SCALING_METHOD == 'min-max':
                    df_score['value_score'] = min_max_normalize(df_score['value_score'])
                    df_score['growth_score'] = min_max_normalize(df_score['growth_score'])
                    logging.info(f"Applied min-max scaling to value and growth scores for industry {industry}.")
                
                elif SCALING_METHOD == 'no-scaling':
                    logging.info(f"No scaling applied to value and growth scores for industry {industry}.")
                
                else:
                    raise ValueError(f"Invalid SCALING_METHOD: {SCALING_METHOD}")
            
            except Exception as e:
                logging.error(f"Error applying scaling method {SCALING_METHOD} to industry {industry}: {str(e)}")
                raise
               
            

            # # Rank the scores
            # df_score = rank_scores(df_score, 'value_score')
            # df_score = rank_scores(df_score, 'growth_score')


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
            logging.info(f"Industry: {industry}, Number of tickers: {len(df['tic'].unique())}, Data Shape: {df.shape}")


    df_industry_scores = df_industry_scores.drop_duplicates(subset='tic', keep='first')

    return df_industry_scores

class Momentum_Score_updated:
    def __init__(self, formation_date, risk_free_rate=0):
        self.formation_date = formation_date
        self.end_date = pd.to_datetime(self.formation_date)
        self.start_date = self.end_date - timedelta(days = 365 * VOLATILITY_PERIOD + 30) 
        self.risk_free_rate = risk_free_rate

    def kk_momentum_score(self, df_tics_daily, momentum_period=3, gap=1):
        T1 = momentum_period
        df_tics_daily = df_tics_daily[(df_tics_daily['date'] >= self.start_date) & (df_tics_daily['date'] <= self.end_date)].reset_index(drop=True)
        df_tics_daily_pivot = df_tics_daily.pivot(index='date', columns='tic', values='close')
        
        # Check if there are enough rows to perform the indexing operation
        if len(df_tics_daily_pivot) < (21 * T1 + gap):
            print(f"Not enough data for momentum period {momentum_period}. Skipping this period.")
            return None

        all_daily_return = df_tics_daily_pivot.pct_change()
        std_daily_return = np.sqrt(252) * all_daily_return.iloc[-252*VOLATILITY_PERIOD:].std()
        risk_adj_mom_score = (((df_tics_daily_pivot.iloc[-gap] / df_tics_daily_pivot.iloc[-21 * T1 - gap]) - 1) / std_daily_return) - self.risk_free_rate
        risk_adj_mom_score = risk_adj_mom_score.dropna()
        momentum_zscore = stats.zscore(risk_adj_mom_score)
        df_tics_momentum_zscore = pd.DataFrame(momentum_zscore, columns=['Momentum_Score']).reset_index()
        return df_tics_momentum_zscore

    def compute_momentum_score(self, df_tics_daily):
        momentum_columns = []
        momentum_scores = []

        for period in MOMENTUM_PERIOD_LIST:
            key = f"momentum_{period}"
            df_momentum_score = self.kk_momentum_score(df_tics_daily, momentum_period=period, gap=GAP)
            if df_momentum_score is not None:
                df_momentum_score = df_momentum_score.rename(columns={"Momentum_Score": key})
                momentum_scores.append(df_momentum_score[['tic', key]])
                momentum_columns.append(key)

        if not momentum_scores:
            raise ValueError("No valid momentum scores calculated. Check your data and parameters.")

        # Concatenate all momentum scores into a single DataFrame
        df_momentum = pd.concat(momentum_scores, axis=1)

        # Remove duplicate 'tic' columns
        df_momentum = df_momentum.loc[:, ~df_momentum.columns.duplicated()]

        momentum_weights = np.array([1 / period for period in MOMENTUM_PERIOD_LIST])
        momentum_weights = momentum_weights / momentum_weights.sum()

        # Calculate the mean momentum score
        if EQUAL_WEIGHT_MOMENTUM:
            df_momentum['momentum_score'] = df_momentum[momentum_columns].mean(axis=1)
        else:
            df_momentum['momentum_score'] = df_momentum[momentum_columns].dot(momentum_weights)

        # Normalize the momentum score using z-score
        df_momentum['momentum_score'] = stats.zscore(df_momentum['momentum_score'])

        # Sort by momentum score in descending order
        df_momentum = df_momentum.sort_values(by='momentum_score', ascending=False)

        return df_momentum

def compute_vgm_score(df_tics_daily, formation_date, tickers_list, df_rank_marketcap, df_funda,df_sector, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR = 'no'):
    if RATING_OR_SCORE == 'score':
        if VGM_METHOD == 'only-momentum':
            # MS = Momentum_Score_v2(formation_date)
            # df_vgm_score = MS.compute_momentum_score(df_tics_daily)

            MS = Momentum_Score_updated(formation_date)
            df_vgm_score = MS.compute_momentum_score(df_tics_daily)

            # df_vgm_score = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
            
            if CONSIDER_RISK_FACTOR == 'yes':
                df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
                df_vgm_score = pd.merge(df_vgm_score, df_risk, on='tic', how='inner')

                if SCALING_METHOD == 'min-max':
                    df_vgm_score['norm_momentum_score'] = 0.5 * min_max_normalize(df_vgm_score['momentum_score'])
                    df_vgm_score['norm_risk_score'] = 0.5 * (1 - min_max_normalize(df_vgm_score['risk_score']))
                    df_vgm_score['norm_score_avg'] = df_vgm_score[['norm_momentum_score', 'norm_risk_score']].sum(axis=1)

                    logging.info(f"Check for Multicollinearity between (i) momentum_score and (ii) risk_score = {round(compute_collinearity(df_vgm_score, 'norm_momentum_score', 'norm_risk_score'),4)}")

                elif SCALING_METHOD == 'z-score':
                    df_vgm_score['z_momentum_score'] = 0.5 * zscore(df_vgm_score['momentum_score'])
                    df_vgm_score['z_risk_score'] = 0.5 * (-zscore(df_vgm_score['risk_score']))
                    df_vgm_score['z_score_avg'] = df_vgm_score[['z_momentum_score', 'z_risk_score']].sum(axis=1)
                    
                    logging.info(f"Check for Multicollinearity between (i) momentum_score and (ii) risk_score = {round(compute_collinearity(df_vgm_score, 'norm_momentum_score', 'norm_risk_score'),4)}")

                df_vgm_score = df_vgm_score.sort_values(by='norm_score_avg', ascending=False)            

            df_vgm_score = df_vgm_score.reset_index(drop = True)
            return df_vgm_score
        else:
            if COUNTRY == 'IN':
                value_coeff, growth_coeff = load_factors_coefficients_IN()
                df_value_growth = compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff)

            elif COUNTRY == 'US':
                value_coeff, growth_coeff = load_factors_coefficients_US_v2()
                df_value_growth = compute_value_growth_score_US(df_funda, df_sector, formation_date, value_coeff, growth_coeff)
            else:
                print(f"Other country do not support")

            # df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

            df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]        # select only those tickers which are scored (V and G)
            
            MS = Momentum_Score_updated(formation_date)
            df_momentum = MS.compute_momentum_score(df_tics_daily)
            
            # df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
            df_vgm_score = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')

            if PRODUCTION_FLAG:
                df_momentum_ind = compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date)
                df_vgm_score = pd.merge(df_vgm_score, df_momentum_ind, on='tic', how='inner')

            if CONSIDER_RISK_FACTOR == 'yes':
                df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
                df_vgm_score = pd.merge(df_vgm_score, df_risk, on='tic', how='inner')

            df_vgm_score = pd.merge(df_vgm_score, df_rank_marketcap, on='tic', how='left')

        if VGM_METHOD == 'z-score_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_score['z_value_score'] = 0.25 * zscore(df_vgm_score['value_score'])
                df_vgm_score['z_growth_score'] = 0.25 * zscore(df_vgm_score['growth_score'])
                df_vgm_score['z_momentum_score'] = 0.25 * zscore(df_vgm_score['momentum_score'])
                df_vgm_score['z_risk_score'] = 0.25 * (- zscore(df_vgm_score['risk_score'])) 
                
                df_vgm_score['z_score_avg'] = df_vgm_score[['z_value_score', 'z_growth_score', 'z_momentum_score','z_risk_score']].sum(axis=1)
                df_vgm_score = df_vgm_score.sort_values(by='z_score_avg', ascending=False)
                
            elif CONSIDER_RISK_FACTOR == 'no':
                df_vgm_score['z_value_score'] = 0.25 *zscore(df_vgm_score['value_score'])
                df_vgm_score['z_growth_score'] = 0.25 *zscore(df_vgm_score['growth_score'])
                df_vgm_score['z_momentum_score'] = 0.5 *zscore(df_vgm_score['momentum_score'])
                df_vgm_score['z_score_avg'] = df_vgm_score[['z_value_score', 'z_growth_score', 'z_momentum_score']].sum(axis=1)
                df_vgm_score = df_vgm_score.sort_values(by='z_score_avg', ascending=False)

            try:
                df_vgm_score = rank_scores(df_vgm_score, 'z_score_avg')
                df_vgm_score = df_vgm_score.rename(columns = {'z_score_avg_rank':'vgm_score_rank','z_score_avg':'vgm_score'})
            except:
                pdb.set_trace()

            df_vgm_score = df_vgm_score.reset_index(drop = True)
            return df_vgm_score

        if VGM_METHOD == 'min-max_avg': 
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_score['norm_value_score'] = 0.25 * min_max_normalize(df_vgm_score['value_score'])
                df_vgm_score['norm_growth_score'] = 0.25 * min_max_normalize(df_vgm_score['growth_score'])
                df_vgm_score['norm_momentum_score'] = 0.25 * min_max_normalize(df_vgm_score['momentum_score'])
                df_vgm_score['norm_risk_score'] = 0.25 * (1 - min_max_normalize(df_vgm_score['risk_score'])) 
                df_vgm_score['min-max_avg'] = df_vgm_score[['norm_value_score', 'norm_growth_score', 'norm_momentum_score','norm_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_score['norm_value_score'] = 0.25 *min_max_normalize(df_vgm_score['value_score'])
                df_vgm_score['norm_growth_score'] = 0.25 *min_max_normalize(df_vgm_score['growth_score'])
                df_vgm_score['norm_momentum_score'] = 0.5 *min_max_normalize(df_vgm_score['momentum_score'])
                df_vgm_score['min-max_avg'] = df_vgm_score[['norm_value_score', 'norm_growth_score', 'norm_momentum_score']].sum(axis=1)
            
            df_vgm_score = df_vgm_score.sort_values(by='min-max_avg', ascending=False)  
            df_vgm_score = rank_scores(df_vgm_score, 'min-max_avg')
            df_vgm_score = df_vgm_score.rename(columns = {'min-max_avg_rank':'vgm_score_rank','min-max_avg':'vgm_score'})
            df_vgm_score = df_vgm_score.reset_index(drop = True)
            return df_vgm_score   

        if VGM_METHOD == 'percentile_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_score['percentile_value_score'] = 0.25 * calculate_percentile(df_vgm_score['value_score'])
                df_vgm_score['percentile_growth_score'] = 0.25 * calculate_percentile(df_vgm_score['growth_score'])
                df_vgm_score['percentile_momentum_score'] = 0.25 * calculate_percentile(df_vgm_score['momentum_score'])
                df_vgm_score['percentile_risk_score'] = 0.25 * (1 - calculate_percentile(df_vgm_score['risk_score'])) 
                df_vgm_score['percentile_avg'] = df_vgm_score[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score','percentile_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_score['percentile_value_score'] = 0.25 * calculate_percentile(df_vgm_score['value_score'])
                df_vgm_score['percentile_growth_score'] = 0.25 * calculate_percentile(df_vgm_score['growth_score'])
                df_vgm_score['percentile_momentum_score'] = 0.5 * calculate_percentile(df_vgm_score['momentum_score'])
                df_vgm_score['percentile_avg'] = df_vgm_score[['percentile_value_score', 'percentile_growth_score', 'percentile_momentum_score']].sum(axis=1)
            
            df_vgm_score = df_vgm_score.sort_values(by='percentile_avg', ascending=False)
            df_vgm_score = rank_scores(df_vgm_score, 'percentile_avg')
            df_vgm_score = df_vgm_score.rename(columns = {'percentile_avg_rank':'vgm_score_rank','percentile_avg':'vgm_score'})
            df_vgm_score = df_vgm_score.reset_index(drop = True)
            return df_vgm_score

        if VGM_METHOD == 'rank_avg':
            if CONSIDER_RISK_FACTOR == 'yes':
                df_vgm_score['rank_value_score'] = 0.25 * df_vgm_score['value_score_rank']
                df_vgm_score['rank_growth_score'] = 0.25 * df_vgm_score['growth_score_rank']
                df_vgm_score['rank_momentum_score'] = 0.25 * df_vgm_score['momentum_score_rank']
                df_vgm_score['rank_risk_score'] = 0.25 * df_vgm_score['risk_score_rank']
                df_vgm_score['rank_avg'] = df_vgm_score[['rank_value_score', 'rank_growth_score', 'rank_momentum_score','rank_risk_score']].sum(axis=1)
             
            elif CONSIDER_RISK_FACTOR == 'no':  
                df_vgm_score['rank_value_score'] = df_vgm_score['value_score_rank']
                df_vgm_score['rank_growth_score'] = df_vgm_score['growth_score_rank']
                df_vgm_score['rank_momentum_score'] = df_vgm_score['momentum_score_rank']
                df_vgm_score['rank_avg'] = df_vgm_score[['rank_value_score', 'rank_growth_score', 'rank_momentum_score']].astype(float).sum(axis=1)
            
            df_vgm_score = df_vgm_score.sort_values(by='rank_avg', ascending=False)
            df_vgm_score = rank_scores(df_vgm_score, 'rank_avg')
            df_vgm_score = df_vgm_score.rename(columns = {'rank_avg_rank':'vgm_score_rank','rank_avg':'vgm_score'})
            df_vgm_score = df_vgm_score.reset_index(drop = True)
            return df_vgm_score

    elif RATING_OR_SCORE =='rating':
       
        if COUNTRY == 'IN':
            value_coeff, growth_coeff = load_factors_coefficients_IN()
            df_value_growth = compute_value_growth_score_IN(formation_date, value_coeff, growth_coeff)

        elif COUNTRY == 'US':
            value_coeff, growth_coeff = load_factors_coefficients_US_v2()
            df_value_growth = compute_value_growth_score_US(formation_date, value_coeff, growth_coeff)

        else:
            print(f"Other country do not support")

        df_value_growth = df_value_growth.rename(columns = {'symbol':'tic'})

        df_tics_daily = df_tics_daily[df_tics_daily['tic'].isin(df_value_growth['tic'])]        # select only those tickers which are scored (V and G)
        df_momentum = compute_momentum_score(df_tics_daily, formation_date, tickers_list)
        df_vgm_score = pd.merge(df_value_growth, df_momentum, on='tic', how='inner')

        if SAVE_INDUSTRIAL_MOMENTUM_SCORE:
            df_momentum_ind = compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date)
            df_vgm_score = pd.merge(df_vgm_score, df_momentum_ind, on='tic', how='inner')

        if CONSIDER_RISK_FACTOR == 'yes':
            df_risk = compute_risk_score(df_tics_daily, formation_date, tickers_list)
            df_vgm_score = pd.merge(df_vgm_score, df_risk, on='tic', how='inner')

        df_vgm_score = pd.merge(df_vgm_score, df_rank_marketcap, on='tic', how='left')

        if VGM_METHOD == 'value':
            df_vgm_score= df_vgm_score.sort_values(by=['value_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'growth':
            df_vgm_score= df_vgm_score.sort_values(by=['growth_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'value-mom':
            df_vgm_score= df_vgm_score.sort_values(by=['value_score_rank','momentum_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'growth-mom':
            df_vgm_score = df_vgm_score.sort_values(by=['growth_score_rank','momentum_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'mom-value':
            df_vgm_score= df_vgm_score.sort_values(by=['momentum_score_rank','value_score_rank','marketcap'], ascending=False)

        elif VGM_METHOD == 'mom-growth':
            df_vgm_score = df_vgm_score.sort_values(by=['momentum_score_rank','growth_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'growth-value-mom':
            df_vgm_score = df_vgm_score.sort_values(by=['growth_score_rank','value_score_rank','momentum_score_rank','marketcap'], ascending=False)
        
        elif VGM_METHOD == 'growth-value':
            df_vgm_score = df_vgm_score.sort_values(by=['growth_score_rank','value_score_rank','marketcap'], ascending=False)
        
        else:
            return None
        df_vgm_score = df_vgm_score.reset_index(drop = True)
        return df_vgm_score

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def calculate_percentile(series):
    return (series.rank(method='min') - 1) / (len(series) - 1)

def send_email(name, email, message, formspree_email):
    import requests
    form_data = {
        'name': name,
        'email': email,
        'message': message
    }
    response = requests.post(f'https://formspree.io/{formspree_email}', data=form_data)
    if response.status_code == 200:
        print("Email sent successfully.")
    else:
        print("Failed to send email.")

def plot_returns(df_returns, tickers_list=[], filename='results/Strategy', period='daily', name_stock_world=None):
    fig, ax1 = plt.subplots(figsize=(21, 10), dpi=120)
    linestyle_val = ['-', '-.', '--', ':', '-']
    linewidth_val = [3, 2, 2, 2, 2]
    color_val = ['royalblue', 'red', 'darkorange', 'green', 'purple']
    for idx in np.arange(len(df_returns.columns.to_list())):
        ax1.plot(df_returns.index, 100 * ep.cum_returns(df_returns.iloc[:, idx]), linestyle=linestyle_val[idx], linewidth=linewidth_val[idx], color=color_val[idx])
    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.xaxis_date()
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(linestyle='dotted', linewidth=1)
    ax1.legend(df_returns.columns.to_list(), fontsize=20)
    plt.savefig(filename + '.jpeg', bbox_inches='tight')
    plt.close()

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
        ax1.plot(df_returns.index, 100 * ((df_returns[df_returns.columns.to_list()[idx]] + 1).cumprod() - 1),
                 color=color_val[idx], linestyle=linestyle_val[idx], linewidth=linewidth_val[idx])

    min_date = df_returns.index.min()
    max_date = df_returns.index.max()
    ax1.set_xlim(min_date, max_date)

    # Format date axis
    plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))


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

def compute_metrics_for_ticker(group):
    # Ensure data is sorted by date for each group
    group = group.sort_index()

    # Calculate Previous Close (shift close by 1 day)
    group['prev_close'] = group['close'].shift(1)

    # Calculate Change and Change%
    group['change'] = group['close'] - group['prev_close']
    group['change%'] = (group['change'] / group['prev_close']) * 100

    # 52-week high and low
    group['52_week_high'] = group['high'].rolling(window=252, min_periods=1).max()
    group['52_week_low'] = group['low'].rolling(window=252, min_periods=1).min()

    # Average Volume (20-day rolling average)
    group['avg_volume_20d'] = group['volume'].rolling(window=20, min_periods=1).mean()

    # YTD_Performance (from start of the current year)
    current_year = pd.Timestamp.today().year
    start_of_year = group.index[group.index.year == current_year][0]
    ytd_performance = 100 * (group['close'].iloc[-1] - group['close'].loc[start_of_year]) / group['close'].loc[start_of_year]

    # 1-week (7-day), 1-month (30-day), 3-month (90-day), 6-month (180-day), and 1-year (365-day) returns
    def calculate_return(days_ago):
        # Ensure there's enough data
        if len(group) > days_ago:
            past_date = group.index[-days_ago]
            return 100 * (group['close'].iloc[-1] - group['close'].loc[past_date]) / group['close'].loc[past_date]
        else:
            return None  # Not enough data for this period

    one_week_return = calculate_return(7)
    one_month_return = calculate_return(30)
    three_month_return = calculate_return(90)
    six_month_return = calculate_return(180)
    one_year_return = calculate_return(365)

    # Get the latest row
    latest_data = group.iloc[-1]

    # Return the calculated metrics for the current ticker
    return {
        'Symbol': group['tic'].iloc[0],
        'Open': latest_data['open'],
        'High': latest_data['high'],
        'Low': latest_data['low'],
        'Close': latest_data['close'],
        'Prev_Close': latest_data['prev_close'],
        'Change': latest_data['change'],
        'Change%': latest_data['change%'],
        'Volume': latest_data['volume'],
        'Avg_Vol_20D': latest_data['avg_volume_20d'],
        '52_Week_High': latest_data['52_week_high'],
        '52_Week_Low': latest_data['52_week_low'],
        'YTD_Performance': ytd_performance,
        '1W_Return': one_week_return,
        '1M_Return': one_month_return,
        '3M_Return': three_month_return,
        '6M_Return': six_month_return,
        '1Y_Return': one_year_return,
        'Last_Available_Date': latest_data.name,  # Keep the date of the last available row
    }

def compute_stock_metrics(df):
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Group data by ticker
    grouped = df.groupby('tic')

    # Use multiprocessing to compute metrics for each ticker in parallel
    with Pool(cpu_count()) as pool:
        metrics = pool.map(compute_metrics_for_ticker, [group for _, group in grouped])

    return pd.DataFrame(metrics)

def compute_metrics_for_ticker(group):
    # Ensure data is sorted by date for each group
    group = group.sort_index()

    # Calculate Previous Close (shift close by 1 day)
    group['prev_close'] = group['close'].shift(1)

    # Calculate Change and Change%
    group['change'] = group['close'] - group['prev_close']
    group['change%'] = (group['change'] / group['prev_close']) * 100

    # 52-week high and low
    group['52_week_high'] = group['high'].rolling(window=252, min_periods=1).max()
    group['52_week_low'] = group['low'].rolling(window=252, min_periods=1).min()

    # Average Volume (20-day rolling average)
    group['avg_volume_20d'] = group['volume'].rolling(window=20, min_periods=1).mean()

    # YTD_Performance (from start of the current year)
    current_year = pd.Timestamp.today().year
    start_of_year = group.index[group.index.year == current_year][0]
    ytd_performance = 100 * (group['close'].iloc[-1] - group['close'].loc[start_of_year]) / group['close'].loc[start_of_year]

    # 1-week (7-day), 1-month (30-day), 3-month (90-day), 6-month (180-day), and 1-year (365-day) returns
    def calculate_return(days_ago):
        # Ensure there's enough data
        if len(group) > days_ago:
            past_date = group.index[-days_ago]
            return 100 * (group['close'].iloc[-1] - group['close'].loc[past_date]) / group['close'].loc[past_date]
        else:
            return None  # Not enough data for this period

    one_week_return = calculate_return(7)
    one_month_return = calculate_return(30)
    three_month_return = calculate_return(90)
    six_month_return = calculate_return(180)
    one_year_return = calculate_return(365)

    # Get the latest row
    latest_data = group.iloc[-1]

    # Return the calculated metrics for the current ticker
    return {
        'Symbol': group['tic'].iloc[0],
        'Open': latest_data['open'],
        'High': latest_data['high'],
        'Low': latest_data['low'],
        'Close': latest_data['close'],
        'Prev_Close': latest_data['prev_close'],
        'Change': latest_data['change'],
        'Change%': latest_data['change%'],
        'Volume': latest_data['volume'],
        'Avg_Vol_20D': latest_data['avg_volume_20d'],
        '52_Week_High': latest_data['52_week_high'],
        '52_Week_Low': latest_data['52_week_low'],
        'YTD_Performance': ytd_performance,
        '1W_Return': one_week_return,
        '1M_Return': one_month_return,
        '3M_Return': three_month_return,
        '6M_Return': six_month_return,
        '1Y_Return': one_year_return,
        'Last_Available_Date': latest_data.name,  # Keep the date of the last available row
    }

def compute_stock_metrics(df):
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Group data by ticker
    grouped = df.groupby('tic')

    # Use multiprocessing to compute metrics for each ticker in parallel
    with Pool(cpu_count()) as pool:
        metrics = pool.map(compute_metrics_for_ticker, [group for _, group in grouped])

    return pd.DataFrame(metrics)

def create_date_list(df_tics_daily, freq):
    if freq == 'Monthly':
        date_list = pd.date_range(start=START_DATE, end=END_DATE, freq='MS').tolist()
    elif freq == 'Weekly':
        date_list = pd.date_range(start=START_DATE, end=END_DATE, freq='W-FRI').tolist()
    elif freq == 'Fortnight':
        date_list = pd.date_range(start=START_DATE, end=END_DATE, freq='W-FRI').tolist()
        date_list = date_list[::2]
    else:
        raise ValueError("Invalid frequency. Choose from 'Monthly', 'Weekly', or 'Fortnight'.")

    valid_date_list = []
    unique_business_dates = df_tics_daily['date'].unique()
    for date in date_list:
        while sum(unique_business_dates == pd.to_datetime(date)) == 0:
            date += timedelta(days=1)
        valid_date_list.append(date)

    # Append END_DATE if not present in the list
    if END_DATE not in valid_date_list:
        valid_date_list.append(END_DATE)

    if DEBUG:
        start_point = -4                       # uncomment to evaluate last month only
        valid_date_list = valid_date_list[start_point:]
        print(f"List of dates: {[datu.strftime('%Y-%m-%d') for datu in valid_date_list]}")

    return valid_date_list

# def load_raw_data():
#     # Load raw data from the local directory
#     df_tics_daily = pd.read_hdf(PATH_DAILY_DATA)
#     df_marketcap = pd.read_hdf(PATH_MARKETCAP_DATA)
#     df_sector = pd.read_hdf(PATH_SECTOR_DATA)
#     df_funda = pd.read_hdf(PATH_FUNDA_DATA)

#     df_sector = df_sector.rename(columns= {'Ticker': 'tic'})
#     df_funda = df_funda.loc[:, ~df_funda.columns.duplicated()]

#     if {'incomeBeforeTax', 'totalAssets'}.issubset(df_funda.columns):
#         mask = (df_funda['totalAssets'].notna() & (df_funda['totalAssets'] != 0))
#         df_funda.loc[mask, 'er'] = np.divide(
#             df_funda.loc[mask, 'incomeBeforeTax'],
#             df_funda.loc[mask, 'totalAssets']
#         )

#     df_sector_slim = df_sector[['tic', 'Sector', 'Industry']].copy()
#     df_funda = pd.merge(df_funda,df_sector_slim, on='tic',how='inner')

#     if 'releaseDate' in df_funda.columns:
#         if 'date' in df_funda.columns:
#             df_funda = df_funda.drop(columns=['date'])
        
#         # Handle release dates
#         mask = df_funda['releaseDate'].notna()
#         df_funda.loc[mask, 'releaseDate'] = pd.to_datetime(df_funda.loc[mask, 'releaseDate'])
        
#         # Handle AMC dates
#         df_funda['time'] = df_funda['time'].fillna('amc')
#         amc_mask = df_funda['time'].str.contains('amc', case=False, na=False)
#         df_funda.loc[amc_mask, 'releaseDate'] += pd.Timedelta(days=1)
        
#         # Rename to standard column
#         df_funda = df_funda.rename(columns={'releaseDate': 'date'})

#     current_date = pd.Timestamp.now().date()
#     n_years_ago = current_date - pd.DateOffset(years=6)
#     date_mask = (df_funda['date'].dt.date.between(n_years_ago.date(), current_date))
#     df_funda = df_funda.loc[date_mask]
    
#     # Convert 'date' column to datetime format
#     df_marketcap['date'] = pd.to_datetime(df_marketcap['date'])

#     return df_tics_daily, df_marketcap, df_sector, df_funda

def load_raw_data():
    try:
        logging.info("Loading raw data from local directory.")
        
        df_tics_daily = pd.read_hdf(PATH_DAILY_DATA)
        df_marketcap = pd.read_hdf(PATH_MARKETCAP_DATA)
        df_sector = pd.read_hdf(PATH_SECTOR_DATA)
        df_funda = pd.read_hdf(PATH_FUNDA_DATA)
        
        logging.info("Raw data loaded successfully.")
        
        df_sector = df_sector.rename(columns={'Ticker': 'tic'})
        df_funda = df_funda.loc[:, ~df_funda.columns.duplicated()]
        
        if {'incomeBeforeTax', 'totalAssets'}.issubset(df_funda.columns):
            mask = (df_funda['totalAssets'].notna() & (df_funda['totalAssets'] != 0))
            df_funda.loc[mask, 'er'] = np.divide(
                df_funda.loc[mask, 'incomeBeforeTax'],
                df_funda.loc[mask, 'totalAssets']
            )
            logging.info("Calculated 'er' for non-null and non-zero 'totalAssets'.")
        
        df_sector_slim = df_sector[['tic', 'Sector', 'Industry']].copy()
        df_funda = pd.merge(df_funda, df_sector_slim, on='tic', how='inner')
        logging.info("Merged sector information into fundamental data.")
        
        if 'releaseDate' in df_funda.columns:
            if 'date' in df_funda.columns:
                df_funda = df_funda.drop(columns=['date'])
                logging.info("Dropped existing 'date' column from fundamental data.")
            
            mask = df_funda['releaseDate'].notna()
            df_funda.loc[mask, 'releaseDate'] = pd.to_datetime(df_funda.loc[mask, 'releaseDate'])
            df_funda['time'] = df_funda['time'].fillna('amc')
            amc_mask = df_funda['time'].str.contains('amc', case=False, na=False)
            df_funda.loc[amc_mask, 'releaseDate'] += pd.Timedelta(days=1)
            df_funda = df_funda.rename(columns={'releaseDate': 'date'})
            logging.info("Processed 'releaseDate' and renamed to 'date'.")
        
        current_date = pd.Timestamp.now().normalize()
        n_years_ago = current_date - pd.DateOffset(years=8)
        date_mask = (df_funda['date'].dt.date.between(n_years_ago.date(), current_date.date()))
        df_funda = df_funda.loc[date_mask]
        logging.info("Filtered fundamental data for the last 8 years.")

        # filter df_tics_daily for the last 8 years
        df_tics_daily['date'] = pd.to_datetime(df_tics_daily['date'])
        df_tics_daily = df_tics_daily[df_tics_daily['date'].dt.date.between(n_years_ago.date(), current_date.date())]
        logging.info("Filtered daily data for the last 8 years.")

        df_marketcap['date'] = pd.to_datetime(df_marketcap['date'])
        logging.info("Converted 'date' column in market cap data to datetime format.")
        
        return df_tics_daily, df_marketcap, df_sector, df_funda
    
    except Exception as e:
        logging.error(f"Error occurred while loading raw data: {e}")
        raise

class production_run:
    def __init__(self, df_tics_daily, date_list, tickers_list, df_metrics, df_vgm_score, df_marketcap, df_sector, INDICES_LIST, LP):
        self.df_tics_daily = df_tics_daily
        self.date_list = date_list
        self.tickers_list = tickers_list
        self.df_metrics = df_metrics
        self.df_vgm_score = df_vgm_score
        self.df_marketcap = df_marketcap
        self.df_sector = df_sector
        self.INDICES_LIST = INDICES_LIST
        self.LP = LP

def stock_filter_marketcap(df_marketcap: pd.DataFrame, 
                          formation_date: Union[str, datetime, pd.Timestamp], 
                          marketcap_th: float) -> tuple[list, pd.DataFrame]:
    """
    Filter stocks based on market capitalization threshold at a given formation date.
    
    Args:
        df_marketcap (pd.DataFrame): DataFrame containing market cap data with columns ['date', 'tic', 'marketCap']
        formation_date (Union[str, datetime, pd.Timestamp]): Date to filter market cap data
        marketcap_th (float): Market capitalization threshold value
    
    Returns:
        tuple[list, pd.DataFrame]: Filtered list of tickers and DataFrame with market cap rankings
        
    Raises:
        ValueError: If required columns are missing or inputs are invalid
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Filtering stocks by market cap threshold: {marketcap_th:,.0f}")
    
    try:
        # Validate inputs
        required_cols = ['date', 'tic', 'marketCap']
        if not all(col in df_marketcap.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        if not isinstance(marketcap_th, (int, float)) or marketcap_th <= 0:
            raise ValueError("Market cap threshold must be a positive number")
            
        # Convert formation_date to datetime
        formation_date = pd.to_datetime(formation_date)
        
        # Filter data
        temp_mktcap = df_marketcap[df_marketcap['date'] <= formation_date].copy()
        
        if temp_mktcap.empty:
            logger.warning(f"No market cap data found before {formation_date}")
            return [], pd.DataFrame()
            
        # Get latest market cap before formation date for each ticker
        temp_mktcap = temp_mktcap.loc[temp_mktcap.groupby('tic')['date'].idxmax()]
        
        # Apply market cap threshold
        temp_mktcap = temp_mktcap[temp_mktcap['marketCap'] >= marketcap_th]
        
        if temp_mktcap.empty:
            logger.warning(f"No stocks found above market cap threshold {marketcap_th:,.0f}")
            return [], pd.DataFrame()
            
        # Rank by market cap
        df_rank_marketcap = temp_mktcap.sort_values(
            by='marketCap', 
            ascending=False
        ).reset_index(drop=True)
        
        tickers_list = df_rank_marketcap['tic'].tolist()
        
        logger.info(f"Found {len(tickers_list)} stocks above market cap threshold")
        logger.debug(f"Market cap range: {df_rank_marketcap['marketCap'].min():,.0f} - {df_rank_marketcap['marketCap'].max():,.0f}")
        
        return tickers_list, df_rank_marketcap
        
    except Exception as e:
        logger.error(f"Error filtering stocks by market cap: {str(e)}")
        raise

def production_vgm_score_IN():
    rename_dict = {
            'priceToBookRatio_rank': 'PB_Ratio',
            'priceToSalesRatio_rank': 'PS_Ratio',
            'priceEarningsRatio_rank': 'PE_Ratio',
            'priceToFreeCashFlowsRatio_rank': 'Price-to-Free-Cash-Flow_Ratio',
            'enterpriseValueOverEBITDA_rank': 'EV-to-EBITDA',
            'cashConversionCycle_rank': 'Cash Conversion Cycle',
            'debtToEquity_rank': 'Debt-to-Equity',
            'debtToAssets_rank': 'Debt-to-Asset',
            'netDebtToEBITDA_rank': 'Net-Debt-to-EBITDA',
            'dividendYield_rank': 'Dividend_Yield',
            'freeCashFlowYield_rank': 'FCF_Yield',
            'revenueGrowth_rank': 'Revenue_Growth',
            'epsgrowth_rank': 'EPS_Growth',
            'operatingIncomeGrowth_rank': 'Operating_Income_Growth',
            'freeCashFlowGrowth_rank': 'FCF_Growth',
            'assetGrowth_rank': 'Asset_Growth',
            'netProfitMargin_rank': 'Net_Profit_Margin',
            'returnOnAssets_rank': 'ROA',
            'returnOnEquity_rank': 'ROE',
            'returnOnCapitalEmployed_rank': 'ROCE',
            'operatingProfitMargin_rank': 'Operating_Profit_Margin',
            'assetTurnover_rank': 'Asset_Turnover',
            'inventoryTurnover_rank': 'Inventory_Turnover',
            'receivablesTurnover_rank': 'Receivables_Turnover',
            'payablesTurnover_rank': 'Payables_Turnover',
            'currentRatio_rank': 'Current_Ratio',
            'quickRatio_rank': 'Quick_Ratio',
            'cashRatio_rank': 'Cash_Ratio',
            'workingCapital_rank': 'Working_Capital',
            'interestCoverage_rank': 'Interest _Coverage_Ratio'
    }

    # Rename columns in the dataframe
    df_vgm_score = df_vgm_score.rename(columns=rename_dict)

    value_factors_list = ['PB_Ratio','PS_Ratio','PE_Ratio','Price-to-Free-Cash-Flow_Ratio','FCF_Yield','EV-to-EBITDA']
    growth_factors_list = ['Revenue_Growth','EPS_Growth','Operating_Income_Growth','FCF_Growth']
    global_momentum_list = ['momentum_3_rank','momentum_6_rank','momentum_12_rank']
    local_momentum_list = ['ind_momentum_3_rank','ind_momentum_6_rank','ind_momentum_12_rank']

    df_fullname = pd.read_csv(PATH_DATA + '/NSE_Stocks.csv')
    df_fullname['SYMBOL'] = df_fullname['SYMBOL'] + ".NS"
    
    df_vgm_score1 = pd.merge(df_vgm_score, df_fullname,  left_on='tic', right_on='SYMBOL', how='left')
    df_vgm_score1 = df_vgm_score1[['tic','Name_of_Company','vgm_score_rank', 'value_score_rank',
                            'growth_score_rank','momentum_score_rank','risk_score_rank','sector','industry','marketcap']
                            + value_factors_list + growth_factors_list + global_momentum_list + local_momentum_list]
    
    df_vgm_score1 = pd.merge(df_vgm_score1, df_metrics,  left_on='tic', right_on='Symbol', how='left')
    df_vgm_score1 = df_vgm_score1.drop(columns=['tic'])
    df_vgm_score1 = df_vgm_score1[['Symbol',
                                    'Name_of_Company',
                                    'vgm_score_rank', 
                                    'value_score_rank',
                                    'growth_score_rank',
                                    'momentum_score_rank',
                                    'risk_score_rank',
                                    'sector',
                                    'industry',
                                    'Open',
                                    'High',	
                                    'Low',	
                                    'Close',	
                                    'Prev_Close',	
                                    'Change',	
                                    'Change%',	
                                    'Volume',
                                    'Avg_Vol_20D',	
                                    '52_Week_High',
                                    '52_Week_Low',
                                    '1W_Return',
                                    '1M_Return',
                                    '3M_Return',
                                    '6M_Return',
                                    'YTD_Performance',
                                    '1Y_Return',
                                    'marketcap',
                                    'Last_Available_Date'] + value_factors_list + growth_factors_list + global_momentum_list + local_momentum_list]

    df_vgm_score1 = df_vgm_score1.rename(columns={'vgm_score_rank':'VGM_Ratings',
                                'value_score_rank':'Value',
                                'growth_score_rank':'Growth',
                                'momentum_score_rank':'Momentum',
                                'risk_score_rank':'Low_Risk',
                                'sector':'Sector',
                                'industry':'Industry',
                                'Company':'Name_of_Company',
                                'marketcap': 'Marketcap',
                                'momentum_3_rank': 'Short_Term_Momentum',
                                'momentum_6_rank': 'Medium_Term_Momentum',
                                'momentum_12_rank':'Long_Term_Momentum',
                                'ind_momentum_3_rank': 'Short_Term_Industry_Momentum',
                                'ind_momentum_6_rank': 'Medium_Term_Industry_Momentum',
                                'ind_momentum_12_rank': 'Long_Term_Industry_Momentum',
                                })

    df_vgm_score1['Short_Term_Momentum'] = df_vgm_score1['Short_Term_Momentum'].astype(float)/2
    df_vgm_score1['Medium_Term_Momentum'] = df_vgm_score1['Medium_Term_Momentum'].astype(float)/2
    df_vgm_score1['Long_Term_Momentum'] = df_vgm_score1['Long_Term_Momentum'].astype(float)/2

    df_vgm_score1['Short_Term_Industry_Momentum'] = df_vgm_score1['Short_Term_Industry_Momentum'].astype(float)/2
    df_vgm_score1['Medium_Term_Industry_Momentum'] = df_vgm_score1['Medium_Term_Industry_Momentum'].astype(float)/2
    df_vgm_score1['Long_Term_Industry_Momentum'] = df_vgm_score1['Long_Term_Industry_Momentum'].astype(float)/2

    def round_dataframe_columns(df,columns_to_round):
        df[columns_to_round] = df[columns_to_round].round(2)
        return df

    round_dataframe_columns(df_vgm_score1,['Open', 'High', 'Low', 'Close', 'Prev_Close', 
                            'Change', 'Change%', '52_Week_High', '52_Week_Low', 
                            'YTD_Performance', '1W_Return','1M_Return','3M_Return','6M_Return','1Y_Return',])

    df_vgm_score1['Avg_Vol_20D'] = df_vgm_score1['Avg_Vol_20D'].round(0)
    df_vgm_score1['Last_Available_Date'] = df_vgm_score1['Last_Available_Date'].dt.date
    df_vgm_score1.to_excel(EXP_DIR + 'df_vgm_score' + '_' + FORMATION_DATE.strftime("%Y-%m-%d") + '_IN.xlsx', index = False)

def production_vgm_score_US(df_vgm_score,df_sector,df_marketcap):
    rename_dict = {
                'priceToBookRatio_rank': 'PB_Ratio',
                'priceToSalesRatio_rank': 'PS_Ratio',
                'priceEarningsRatio_rank': 'PE_Ratio',
                'priceToFreeCashFlowsRatio_rank': 'Price-to-Free-Cash-Flow_Ratio',
                'enterpriseValueOverEBITDA_rank': 'EV-to-EBITDA',
                'cashConversionCycle_rank': 'Cash Conversion Cycle',
                'debtToEquity_rank': 'Debt-to-Equity',
                'debtToAssets_rank': 'Debt-to-Asset',
                'netDebtToEBITDA_rank': 'Net-Debt-to-EBITDA',
                'dividendYield_rank': 'Dividend_Yield',
                'freeCashFlowYield_rank': 'FCF_Yield',
                'revenueGrowth_rank': 'Revenue_Growth',
                'epsgrowth_rank': 'EPS_Growth',
                'operatingIncomeGrowth_rank': 'Operating_Income_Growth',
                'freeCashFlowGrowth_rank': 'FCF_Growth',
                'assetGrowth_rank': 'Asset_Growth',
                'netProfitMargin_rank': 'Net_Profit_Margin',
                'returnOnAssets_rank': 'ROA',
                'returnOnEquity_rank': 'ROE',
                'returnOnCapitalEmployed_rank': 'ROCE',
                'operatingProfitMargin_rank': 'Operating_Profit_Margin',
                'assetTurnover_rank': 'Asset_Turnover',
                'inventoryTurnover_rank': 'Inventory_Turnover',
                'receivablesTurnover_rank': 'Receivables_Turnover',
                'payablesTurnover_rank': 'Payables_Turnover',
                'currentRatio_rank': 'Current_Ratio',
                'quickRatio_rank': 'Quick_Ratio',
                'cashRatio_rank': 'Cash_Ratio',
                'workingCapital_rank': 'Working_Capital',
                'interestCoverage_rank': 'Interest _Coverage_Ratio'
        }

    # Rename columns in the dataframe
    df_vgm_score = df_vgm_score.rename(columns=rename_dict)


    value_factors_list = ['PB_Ratio','PS_Ratio','PE_Ratio','Dividend_Yield','Price-to-Free-Cash-Flow_Ratio','FCF_Yield','EV-to-EBITDA']
    growth_factors_list = ['Revenue_Growth','EPS_Growth','Operating_Income_Growth','FCF_Growth','Asset_Growth','ROE','ROCE']
    global_momentum_list = ['momentum_3_rank','momentum_6_rank','momentum_12_rank']
    local_momentum_list = ['ind_momentum_3_rank','ind_momentum_6_rank','ind_momentum_12_rank']

    

    # Rename dictionary for CAT1_RATIOS and CAT2_RATIOS
    
    # df_fullname = pd.read_excel(PATH_DATA + '/russell_3000_gurufocus_10-18-2024.xlsx')
    df_vgm_score1 = pd.merge(df_vgm_score, df_sector,  left_on='tic', right_on='Ticker', how='left')
    df_vgm_score1 = df_vgm_score1[['tic','Company','vgm_score_rank', 'value_score_rank','growth_score_rank','momentum_score_rank',
                                        'risk_score_rank','sector','industry','marketcap'] 
                                        + value_factors_list + growth_factors_list + global_momentum_list + local_momentum_list]
    
    df_vgm_score1 = pd.merge(df_vgm_score1, df_metrics,  left_on='tic', right_on='Symbol', how='left')

    df_vgm_score1 = df_vgm_score1.drop(columns=['tic'])
    df_vgm_score1['Date'] = df_vgm_score1['Last_Available_Date'].dt.date

    df_vgm_score1 = df_vgm_score1[['Date', 
                                    'Symbol',
                                    'Company',
                                    'vgm_score_rank', 
                                    'value_score_rank',
                                    'growth_score_rank',
                                    'momentum_score_rank',
                                    'risk_score_rank',
                                    'sector',
                                    'industry',
                                    'Open',
                                    'High',	
                                    'Low',	
                                    'Close',	
                                    'Prev_Close',	
                                    'Change',	
                                    'Change%',	
                                    'Volume',
                                    'Avg_Vol_20D',	
                                    '52_Week_High',
                                    '52_Week_Low',
                                    '1W_Return',
                                    '1M_Return',
                                    '3M_Return',
                                    '6M_Return',
                                    'YTD_Performance',
                                    '1Y_Return',
                                    'marketcap'] + value_factors_list + growth_factors_list + global_momentum_list + local_momentum_list]
            
    df_vgm_score1 = df_vgm_score1.rename(columns={'vgm_score_rank':'VGM_Ratings',
                                'value_score_rank':'Value',
                                'growth_score_rank':'Growth',
                                'momentum_score_rank':'Momentum',
                                'risk_score_rank':'Low_Risk',
                                'sector':'Sector',
                                'industry':'Industry',
                                'Company':'Name_of_Company',
                                'marketcap': 'Marketcap',
                                'momentum_3_rank': 'Short_Term_Momentum',
                                'momentum_6_rank': 'Medium_Term_Momentum',
                                'momentum_12_rank':'Long_Term_Momentum',
                                'ind_momentum_3_rank': 'Short_Term_Industry_Momentum',
                                'ind_momentum_6_rank': 'Medium_Term_Industry_Momentum',
                                'ind_momentum_12_rank': 'Long_Term_Industry_Momentum',
                                })

    df_vgm_score1['Short_Term_Momentum'] = df_vgm_score1['Short_Term_Momentum'].astype(float)/2
    df_vgm_score1['Medium_Term_Momentum'] = df_vgm_score1['Medium_Term_Momentum'].astype(float)/2
    df_vgm_score1['Long_Term_Momentum'] = df_vgm_score1['Long_Term_Momentum'].astype(float)/2

    df_vgm_score1['Short_Term_Industry_Momentum'] = df_vgm_score1['Short_Term_Industry_Momentum'].astype(float)/2
    df_vgm_score1['Medium_Term_Industry_Momentum'] = df_vgm_score1['Medium_Term_Industry_Momentum'].astype(float)/2
    df_vgm_score1['Long_Term_Industry_Momentum'] = df_vgm_score1['Long_Term_Industry_Momentum'].astype(float)/2

    def round_dataframe_columns(df,columns_to_round):
        df[columns_to_round] = df[columns_to_round].round(2)
        return df

    round_dataframe_columns(df_vgm_score1,['Open', 'High', 'Low', 'Close', 'Prev_Close', 
                            'Change', 'Change%', '52_Week_High', '52_Week_Low', 
                            'YTD_Performance', '1W_Return','1M_Return','3M_Return','6M_Return','1Y_Return',])

    df_vgm_score1['Avg_Vol_20D'] = df_vgm_score1['Avg_Vol_20D'].round(0)

    
    # Add "-" as a new category and then fill missing values
    for col in df_vgm_score1.select_dtypes(include=["category"]).columns:
        df_vgm_score1[col] = df_vgm_score1[col].cat.add_categories("-")
        
    # Now fill missing values
    df_vgm_score1 = df_vgm_score1.fillna("-")


    # df_vgm_score1.to_csv(EXP_DIR + 'df_vgm_score' + '_' + FORMATION_DATE.strftime("%Y-%m-%d") + '_US.csv', index = False)
    df_vgm_score1.to_csv(EXP_DIR + 'vgm_score_US.csv', index = False)

def lower_bounded_allocation(df_momentum, df_sector, tech_pct=0, consum_pct=0, fin_pct=0, med_pct=0, indstr_pct=0, enrg_pct=0, othrs_pct=0):
    tech_industries = ['Internet Content & Information', 'Software - Application', 'Semiconductors', 'Computer Hardware',
                       'Information Technology Services', 'Software - Application', 'Semiconductor Equipment & Materials',
                       'Electronic Components', 'Communication Equipment', 'Software - Infrastructure', 'Consumer Electronics',
                       'Scientific & Technical Instruments', 'Hardware, Equipment & Parts', 'Technology Distributors', 'Electronic Gaming & Multimedia',
                       'Internet Software/Services', 'Internet Content & Information', 'Electronics & Computer Distribution']

    selected_stocks = []

    df_momentum = df_momentum.merge(df_sector, on='tic')

    selected_stocks.extend(df_momentum[df_momentum['Industry'].isin(tech_industries)].nlargest(round(PORTFOLIO_SIZE * tech_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'] == 'Consumer Cyclical'].nlargest(round(PORTFOLIO_SIZE * consum_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'] == 'Financial Services'].nlargest(round(PORTFOLIO_SIZE * fin_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'] == 'Healthcare'].nlargest(round(PORTFOLIO_SIZE * med_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'] == 'Industrials'].nlargest(round(PORTFOLIO_SIZE * indstr_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'] == 'Energy'].nlargest(round(PORTFOLIO_SIZE * enrg_pct), 'momentum_score')['tic'].tolist())
    selected_stocks.extend(df_momentum[df_momentum['Sector'].isin(['Basic Materials', 'Utilities', 'Real Estate'])].nlargest(round(PORTFOLIO_SIZE * othrs_pct), 'momentum_score')['tic'].tolist())

    selected_stocks.extend(df_momentum[~df_momentum['tic'].isin(selected_stocks)].nlargest(PORTFOLIO_SIZE - len(selected_stocks), 'momentum_score')['tic'].tolist())

    return selected_stocks


def StockReturnsComputing(prices):
    # Vectorized return computation
    return (prices[1:, :] - prices[:-1, :]) / prices[:-1, :]

def compute_portfolio_stats(weights, covReturns, meanReturns):
    # Compute portfolio risk and return
    portfolio_risk = np.sqrt(weights @ covReturns @ weights.T).item()
    portfolio_return = (weights @ meanReturns.T).item()
    return portfolio_risk, portfolio_return

def create_random_portfolios(cluster_dict, num_portfolios=20):
    clusters_tickers = [tickers for tickers in cluster_dict.values()]
    
    # Generate 20 random portfolios by selecting one random ticker from each cluster
    portfolios = [tuple(random.choice(tickers) for tickers in clusters_tickers) for _ in range(num_portfolios)]
    
    return portfolios

def select_diversified_portfolio(df_tics_daily_window, df_vgm_score_top, df_sector, method= DIV_METHOD):
    if DIV_METHOD == 'KMeans':
        # Function to compute stock returns
        # def StockReturnsComputing(stock_price, rows, cols):            
        #     stock_return = np.zeros([rows-1, cols])
        #     for j in range(cols):  # j: Assets
        #         for i in range(rows-1):  # i: Daily Prices
        #             stock_return[i, j] = (stock_price[i+1, j] - stock_price[i, j]) / stock_price[i, j]
        #     return stock_return

        # Prepare stock price data
        tickers_list = df_vgm_score_top['tic'].to_list()
        df_stock_prices = df_tics_daily_window[df_tics_daily_window['tic'].isin(tickers_list)].pivot_table(index='date', columns='tic', values='close')
        # asset_labels = df_stock_prices.columns.tolist()
        ar_stock_prices = np.asarray(df_stock_prices)
        rows, cols = ar_stock_prices.shape

        # Compute daily returns
        ar_returns = StockReturnsComputing(ar_stock_prices) #, rows, cols)

        # Compute mean returns and covariance matrix
        mean_returns = np.mean(ar_returns, axis=0).reshape(len(ar_returns[0]), 1)
        cov_returns = np.cov(ar_returns, rowvar=False)

        # Prepare asset parameters for KMeans clustering
        asset_parameters = np.concatenate([mean_returns, cov_returns], axis=1)

        # KMeans clustering
        clusters = PORTFOLIO_SIZE
        assets_cluster = KMeans(algorithm='lloyd', max_iter=600, n_clusters=clusters)
        assets_cluster.fit(asset_parameters)
        labels = assets_cluster.labels_

        assets = np.array(tickers_list)

        # Create a dictionary to store assets in each cluster
        cluster_dict = {f'Cluster_{i+1}': list(assets[np.where(labels == i)]) for i in range(clusters)}

        # # Function to create portfolios
        # def create_portfolios(cluster_dict):
        #     clusters_tickers = [tickers for tickers in cluster_dict.values()]
        #     portfolios = list(itertools.product(*clusters_tickers))
        #     return portfolios

        # # Generate portfolios and select 20 random portfolios
        # portfolios = create_portfolios(cluster_dict)
        # Function to create 20 random portfolios without generating all possible combinations


        # Example usage: generate 20 random portfolios
        portfolios = create_random_portfolios(cluster_dict, num_portfolios=50)

        # Create random portfolios dictionary
        portfolio_dict = {f'Portfolio_{i+1}': list(portfolio) for i, portfolio in enumerate(portfolios)}

        # Initialize a dictionary to store portfolio scores
        portfolio_scores = {}

        # Compute diversification scores for each portfolio and store them in the dictionary
        for portfolio_num, tickers in portfolio_dict.items():
            score = diversification_score(df_tics_daily_window, tickers, method='equal-weight')
            portfolio_scores[portfolio_num] = score

        # Select the portfolio with the maximum diversification score
        max_portfolio = max(portfolio_scores, key=portfolio_scores.get)

        # Print the portfolio with the highest diversification score
        print(f'The portfolio with the highest diversification score is {max_portfolio}: {portfolio_dict[max_portfolio]}')
        print(f'Highest diversification score: {portfolio_scores[max_portfolio]}')

        tickers_list = portfolio_dict[max_portfolio]

        df_vgm_score_div = df_vgm_score_top[df_vgm_score_top['tic'].isin(tickers_list)]
        # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
        
        if SAVE_EXCEL:
            df_vgm_score_div.to_excel(os.path.join(EXCEL_DIR, f'div_df_vgm_score_{window_start_date.strftime("%Y-%m-%d")}.xlsx'), index=False)
            # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
            # logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')

    elif DIV_METHOD == 'manual':
        tickers_list = lower_bounded_allocation(df_vgm_score_top, df_sector, tech_pct=0.4, consum_pct=0.1, fin_pct=0.2, med_pct=0.1, indstr_pct=0.1, enrg_pct=0.1, othrs_pct=0)

    return tickers_list


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_stock_world(df_vgm_score, formation_date):
    # global NAME_STOCK_WORLD
    try:
        if COUNTRY == 'US':
            hist_constituents = None
            if PORTFOLIO_STOCK_WORLD == 'SP500':
                hist_constituents = SP500_FILE
                NAME_STOCK_WORLD = 'S&P 500'
            elif PORTFOLIO_STOCK_WORLD == 'SP900':
                hist_constituents = SP900_FILE
                NAME_STOCK_WORLD = 'S&P 900'
            elif PORTFOLIO_STOCK_WORLD == 'SP1500':
                hist_constituents = SP1500_FILE
                NAME_STOCK_WORLD = 'S&P 1500'
            else:
                return df_vgm_score

            df_lar_mid_tickers = pd.read_csv(PATH_DATA + hist_constituents)
            df_lar_mid_tickers['Date'] = pd.to_datetime(df_lar_mid_tickers['Date'])
            latest_date = df_lar_mid_tickers[df_lar_mid_tickers['Date'] <= pd.to_datetime(formation_date)]['Date'].max()
            df_lar_mid_tickers = df_lar_mid_tickers[df_lar_mid_tickers['Date'] == latest_date].reset_index(drop=True)
            
            df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_lar_mid_tickers['Code'])]
            df_vgm_score = df_vgm_score.sort_values(by='vgm_score', ascending=False).reset_index(drop=True)

            return df_vgm_score

        elif COUNTRY == 'IN':
            if LARGE_CAP_FILTER:
                T = LARGE_CAP_THRESHOLD
            elif MID_CAP_FILTER:
                T = MID_CAP_THRESHOLD
            else:
                T = SMALL_CAP_THRESHOLD

            df_vgm_score_mcap_sort = df_vgm_score.sort_values(by='marketcap', ascending=False).head(T)
            df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

            if not (LARGE_CAP_FILTER or MID_CAP_FILTER):
                df_vgm_score = df_vgm_score[df_vgm_score['marketcap'] >= MARKETCAP_THRESHOLD]
                logging.info(f"Number of tickers in midcap = {len(df_vgm_score)}")

            return df_vgm_score

        else:
            logging.warning("Unsupported country specified.")
            return df_vgm_score

    except Exception as e:
        logging.error(f"Error in filter_stock_world: {e}")
        return df_vgm_score

# def filter_stock_world(df_vgm_score,formation_date):
#     if COUNTRY == 'US':
#         if PORTFOLIO_STOCK_WORLD == 'SP500':
#             hist_constituents = 's&p500_component_history.csv'
        
#         elif PORTFOLIO_STOCK_WORLD == 'SP900':
#             hist_constituents = 's&p900_component_history.csv'
        
#         elif PORTFOLIO_STOCK_WORLD == 'SP1500':
#             hist_constituents = 's&p1500_component_history.csv'
#         else:
#             return df_vgm_score

#         df_lar_mid_tickers = pd.read_csv(PATH_DATA + hist_constituents,index=True)
#         df_lar_mid_tickers['Date'] = pd.to_datetime(df_lar_mid_tickers['Date'])
#         df_lar_mid_tickers = df_lar_mid_tickers[df_lar_mid_tickers['Date'] == df_lar_mid_tickers[df_lar_mid_tickers['Date'] <= pd.to_datetime(formation_date)].sort_values(by=['Date']).iloc[-1]['Date']].reset_index(drop = True)
        
#         df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_lar_mid_tickers['Code'])]
#         df_vgm_score = df_vgm_score.sort_values(by = 'vgm_score', ascending = False).reset_index(drop = True)

#         return df_vgm_score

#     if COUNTRY ==  'IN':
#         if LARGE_CAP_FILTER:
#             T = 100
#             df_vgm_score_mcap_sort = df_vgm_score.copy()
#             df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
#             df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

#         elif MID_CAP_FILTER:
#             T = 250
#             df_vgm_score_mcap_sort = df_vgm_score.copy()
#             df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
#             df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

#         elif MID_CAP_FILTER:
#             T = 1000
#             df_vgm_score_mcap_sort = df_vgm_score.copy()
#             df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
#             df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

#         else:
#             df_vgm_score = df_vgm_score[df_vgm_score['marketcap'] >= 20000]
#             print(f"number of tickers in midcap = {len(df_vgm_score)}")


def compute_portfolio_return(date_list,df_tics_daily,df_marketcap, df_sector,df_funda):
    # ============  Initialize firstday return to zero  ======================
    firstday_bnh_return = pd.DataFrame(0,columns=['VGM Portfolio'], index=[date_list[0]])
    df_return_list = [firstday_bnh_return - TC]

    industry_counts_list = []
    lines = []
    monthly_return = {}
    monthly_return_index = {}

    for idx in range(len(date_list)-1):
        window_start_date = date_list[idx]
        window_end_date = date_list[idx + 1]
        logging.info(f"Trading period = {window_start_date.strftime('%Y-%m-%d')} - {window_end_date.strftime('%Y-%m-%d')}")
        print(f"Trading period = {window_start_date.strftime('%Y-%m-%d')} - {window_end_date.strftime('%Y-%m-%d')}")

        formation_date = window_start_date
        start_date_momentum = formation_date - timedelta(days = 365 * MOM_VOLATILITY_PERIOD + 30)
        end_date_momentum= formation_date

        tickers_list, df_marketcap = stock_filter_marketcap(df_marketcap, formation_date, MARKETCAP_TH)

        # if PRINT_FLAG:
        #     print(f"Number of tickers after removing tickers with marketcap < {int(MARKETCAP_TH/1000000)}M = {len(tickers_list)}")

        # logging.info(f"Number of tickers after removing tickers with marketcap < {int(MARKETCAP_TH/1000000)}M = {len(tickers_list)}")

        # # Sort the DataFrame by 'marketcap' in descending order
        # df_marketcap = df_marketcap.head(100)
        # TICKERS_LIST = df_marketcap['tic'].tolist()

        LP = Load_n_Preprocess(tickers_list, start_date_momentum, end_date_momentum)
        df_tics_daily_window = LP.filter_data(df_tics_daily)
        df_tics_daily_window, tickers_list = LP.clean_daily_data(df_tics_daily_window, missing_values_allowed = 0.01)

        if PRINT_FLAG:
            print(f"Number of tickers after removing tickers with missing values = {len(tickers_list)}")

        logging.info(f"Number of tickers after removing tickers with missing values = {len(tickers_list)}")

        # value_coeff, growth_coeff = load_factors_coefficients()
        # df_new_score = compute_value_growth_score(FORMATION_DATE, value_coeff, growth_coeff)
        if COUNTRY == 'IN':
            df_vgm_score = compute_vgm_score(df_tics_daily, formation_date, tickers_list, df_marketcap, df_funda, df_sector, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
            if PRODUCTION_FLAG:
                production_vgm_score_IN(df_vgm_score)

        elif COUNTRY == 'US':
            df_vgm_score = compute_vgm_score(df_tics_daily_window, formation_date, tickers_list, df_marketcap, df_funda, df_sector, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
            if PRODUCTION_FLAG:
                production_vgm_score_US(df_vgm_score)

        df_vgm_score = filter_stock_world(df_vgm_score,formation_date)

        df_vgm_score_marketcap = df_vgm_score.merge(df_marketcap[['tic','marketCap']], on = 'tic', how = 'left')
        if SAVE_EXCEL:
            df_vgm_score_marketcap.to_excel(os.path.join(EXCEL_DIR, f'df_vgm_score_{window_start_date.strftime("%Y-%m-%d")}.xlsx'), index=False)

        df_vgm_score_top = df_vgm_score_marketcap.iloc[0:N_TOP_TICKERS]
        

        # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
        # logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')

        # =========== Industrial Count - Diversification ======================
        # tickers_list = df_vgm_score_top['tic']   #df_vgm_score['tic'].iloc[0:N_TOP_TICKERS]             # selected tickers based on momentum
        # tickers_list = list(tickers_list)

        # with open(periodic_tickers_list_bnh, "a") as file: 
        #     tickers_list.sort()
        #     line = ', '.join([f"({item})" for item in tickers_list])
        #     line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
        #     file.write(line + "\n")

        if CONSIDER_DIVERSIFICATION == 'yes':
            tickers_list = select_diversified_portfolio(df_tics_daily_window, df_vgm_score_top, df_sector, method= DIV_METHOD)
          
        else:
            tickers_list = df_vgm_score_top['tic'].iloc[0:PORTFOLIO_SIZE]          
            tickers_list = list(tickers_list)

        if NAME_STOCK_WORLD == "RUSSELL-3000" and not (VGM_METHOD == 'only-momentum'):
            # =========== Industrial Count - Diversification ======================
            industry_counts = df_vgm_score[df_vgm_score['tic'].isin(tickers_list)]['industry'].value_counts().reset_index()
            industry_counts.columns = ['industry', 'count']
            industry_counts['date'] = formation_date.strftime("%Y-%m-%d")
            industry_counts_list.append(industry_counts)

        # Sort the tickers list and prepare the line
        tickers_list.sort()
        line = ', '.join([f"({item})" for item in tickers_list])
        line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
        lines.append(line)

        # LP = Load_n_Preprocess(tickers_list, window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
        # df_tics_daily = LP.load_daily_data()
        # df_tics_daily = LP.clean_daily_data(df_tics_daily, missing_values_allowed = 0.01)

        LP = Load_n_Preprocess(tickers_list, window_start_date, window_end_date)
        df_tics_daily_window = LP.filter_data(df_tics_daily)


        df_return = buy_hold_portfolio_return(df_tics_daily_window)

        try:
            monthly_return[date_list[idx].strftime('%Y-%m-%d')] = [round(100*ep.cum_returns_final(df_return),2).values[0]]
        except:
            print('Today is the inception day')
            sys.exit(1)  # Exit the program with a non-zero status


        if idx == 0:
            df_return.iloc[-1] = df_return.iloc[-1] - TC
            df_return_list.append(df_return)
        else:
            df_return.iloc[0] = df_return.iloc[0] - TC
            df_return.iloc[-1] = df_return.iloc[-1] - TC
            df_return_list.append(df_return)

        # LP = Load_n_Preprocess([MAJOR_INDEX], window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
        # df_tics_daily_window = LP.filter_data(df_tics_daily)
        # df_tics_daily_window, tickers_list = LP.clean_daily_data(df_tics_daily_window, missing_values_allowed = 0.01)

        LP = Load_n_Preprocess([MAJOR_INDEX], window_start_date, window_end_date)
        df_index_daily_window = LP.filter_data(df_tics_daily)
        df_return_index = buy_hold_portfolio_return(df_index_daily_window)
        monthly_return_index[date_list[idx].strftime('%Y-%m-%d')] = [round(100*ep.cum_returns_final(df_return_index),2).values[0]]

    df_return_bnh = pd.concat(df_return_list)

    if (NAME_STOCK_WORLD == "RUSSELL-3000") and not (VGM_METHOD == 'only-momentum'):
        df_industry_counts = pd.concat(industry_counts_list, ignore_index=True)
        if SAVE_EXCEL:
            df_industry_counts.to_excel(os.path.join(EXP_DIR, 'df_industry_div.xlsx'), index=False)

    df_monthly_return = pd.DataFrame.from_dict(monthly_return, orient='index', columns=['return'])
    if SAVE_EXCEL:
        df_monthly_return.to_excel(os.path.join(EXP_DIR, 'df_monthly_return.xlsx'), index=True)

    df_monthly_return_index = pd.DataFrame.from_dict(monthly_return_index, orient='index', columns=['return'])
    if SAVE_EXCEL:
        df_monthly_return_index.to_excel(EXP_DIR + 'df_monthly_return_index.xlsx', index = True)

    # ============================= Prepare data for to plot Buy-Hold returns  ======================
    df_return_bnh = df_return_bnh.reset_index()
    df_return_bnh.columns = ['date','VGM Portfolio']
    df_return_bnh['date'] = pd.to_datetime(df_return_bnh['date'])
    df_return_bnh = df_return_bnh.set_index(['date'])

    # ===============================     Plotting  ====================================
    LP = Load_n_Preprocess(INDICES_LIST, date_list[0], date_list[-1])
    df_indices_daily_window = LP.filter_data(df_tics_daily)

    df_indices_close = pd.pivot_table(df_indices_daily_window, values ='close', index = 'date',columns='tic')
    df_indices_return = df_indices_close.pct_change().fillna(0.001)
    # df_return_bnh['Portfolio_Trend'] = df_return_bnh['VGM Portfolio'].rolling(window=TREND_WINDOW).mean()
    df_return_bnh = df_return_bnh.merge(df_indices_return,how='left',on = 'date')
    # df_return_bnh = df_return_bnh.drop(columns=['Portfolio_Trend'])

    if NAME_STOCK_WORLD == "RUSSELL-3000":
        df_return_bnh = df_return_bnh.rename(columns = {'VGM Portfolio':'Our Portfolio','^GSPC': 'S&P 500','IJS': 'S&P 600', '^NDX': 'NASDAQ 100','^SP600': 'S&P 600','^RUT':'RUSSELL 2000','^RUA':'RUSSELL 3000'})
        # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
        if SAVE_EXCEL:
            df_return_bnh.to_excel(EXP_DIR + '/' + 'df_return_daily_monthly_ticker_bnh.xlsx')
        # plot_returns(df_return_bnh, tickers_list = [], filename = EXP_DIR + '/' + 'Portfolio_BnH', period = 'daily', name_stock_world = NAME_STOCK_WORLD)

    if NAME_STOCK_WORLD == "NSE-Stocks":
        if PORTFOLIO_FREQ == 'Monthly':
            df_return_bnh = df_return_bnh.rename(columns = {'VGM Portfolio':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        elif PORTFOLIO_FREQ == 'Weekly':
            df_return_bnh = df_return_bnh.rename(columns = {'VGM Portfolio':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        elif PORTFOLIO_FREQ == 'Fortnight':
            df_return_bnh = df_return_bnh.rename(columns = {'VGM Portfolio':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        
        
    # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
    if SAVE_EXCEL:
        df_return_bnh.to_excel(os.path.join(EXP_DIR, 'df_return_daily_monthly_ticker_bnh.xlsx'))

    return df_return_bnh, lines

def main():
    validate_combination(RATING_OR_SCORE, VGM_METHOD)                                     # Validate the combination of rating_or_score and vgm_method
    df_tics_daily, df_marketcap, df_sector, df_funda = load_raw_data()                    # Load raw data from the local directory
    logging.info(f'OHLCV and Marketcap data are loaded from local directory.')

    # find the count of tickers in df_tics_daily and df_marketcap. And cound the common ticker in both the dataframes.
    if PRINT_FLAG:
        print(f"Number of tickers in df_tics_daily: {len(df_tics_daily.tic.unique())}")
        print(f"Number of tickers in df_marketcap: {len(df_marketcap.tic.unique())}")
        print(f"Number of tickers in df_sector: {len(df_sector.tic.unique())}")
        print(f"Number of tickers in df_funda: {len(df_funda.tic.unique())}")
        print(f"Number of common tickers in df_tics_daily and df_marketcap: {len(set(df_tics_daily.tic.unique()).intersection(set(df_marketcap.tic.unique())))}")
        print(f"Number of common tickers in df_funda and df_marketcap: {len(set(df_funda.tic.unique()).intersection(set(df_marketcap.tic.unique())))}")

    logging.info(f"Number of tickers in df_tics_daily: {len(df_tics_daily.tic.unique())}")
    logging.info(f"Number of tickers in df_marketcap: {len(df_marketcap.tic.unique())}")
    logging.info(f"Number of common tickers in df_tics_daily and df_marketcap: {len(set(df_tics_daily.tic.unique()).intersection(set(df_marketcap.tic.unique())))}")
    
    TICKERS_LIST = list(set(df_tics_daily.tic.unique())-set(INDICES_LIST))
    logging.info(f"Number of ticker in the world of stocks: {len(TICKERS_LIST)}")
    logging.info(f'Stock world: {NAME_STOCK_WORLD}')

    date_list = create_date_list(df_tics_daily, PORTFOLIO_FREQ)
    
    df_portfolio_return, lines = compute_portfolio_return(date_list,df_tics_daily,df_marketcap,df_sector,df_funda)

    if PLOT_RANGE_FLAG:
        plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2024-01-01', end_date='2024-12-31')
        plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2023-01-01', end_date='2023-12-31')
        plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2022-01-01', end_date='2022-12-31')
        plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2021-01-01', end_date='2021-12-31')
        plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2020-01-01', end_date='2020-12-31')
    
    # plot_returns_range(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2020-01-01', end_date='2024-12-31')
    # plot_returns(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Our_Portfolio_1'), period = 'daily', name_stock_world = NAME_STOCK_WORLD)
    plot_returns_drawdown(df_portfolio_return, tickers_list = [], filename = os.path.join(EXP_DIR, 'Portfolio_drawdown'), period = 'daily', name_stock_world = NAME_STOCK_WORLD, pos = 'lower right')

    hours, minutes, seconds = computation_time(start_time, "Total execution time: ")

    column_names = df_portfolio_return.columns.tolist()
    # ===============        Save experiment settings        ===================================
    lines = '\n'.join(lines)
    line = [
                f"Country                       : {COUNTRY}",
                f"Filename                      : 20240816_Quant_Portfolio_v3_parametric.py",
                f"OHLCV data file               : {PATH_DAILY_DATA}",
                f"Market data file              : {PATH_MARKETCAP_DATA}",
                f"Stock world                   : {NAME_STOCK_WORLD}",
                ]

    if COUNTRY == 'IN':
        line = line + [
                f"Marketcap threshold           : Rs. {int(MARKETCAP_TH/10000000)} Cr.",]

    elif COUNTRY == 'US':
        line = line + [
                f"Marketcap threshold           : ${int(MARKETCAP_TH/1000000)} M",]

    # Ensure all the required variables are defined before using them
    line = line + [
                f"Rating or Score               : {RATING_OR_SCORE}",
                f"VGM method                    : {VGM_METHOD}",
                f"Portfolio update frequency    : {PORTFOLIO_FREQ}",
                f"No. of tickers in portfolio   : {N_TOP_TICKERS}",
                f"Portfolio size                : {PORTFOLIO_SIZE}",
                f"Consider risk factor          : {CONSIDER_RISK_FACTOR}",
                f"Momentum Volatility period    : {MOM_VOLATILITY_PERIOD}",
                f"Risk Volatility period        : {RISK_VOLATILITY_PERIOD}",
                f"Momentum periods              : {MOMENTUM_PERIOD_LIST}",
                f"Risk periods                  : {RISK_PERIOD_LIST}",
                f"With transaction cost         : {WITH_TRANS_COST}",
                f"Which broker                  : {BROKER}",
                f"Transaction cost              : {TC}",
                f"Start date                    : {START_DATE}",
                f"End date                      : {END_DATE}",
                f"Total execution time          : {str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}",]

    line = line + [            
                f"CAGR Portfolio                : {round(100*ep.cagr(df_portfolio_return[column_names[0]]), 2)} %",]

    if COUNTRY == 'IN':
        line = line + [
                f"CAGR NIFTY 50                 : {round(100*ep.cagr(df_portfolio_return['NIFTY 50']), 2)} %",
                f"CAGR NIFTY 500                : {round(100*ep.cagr(df_portfolio_return['NIFTY 500']), 2)} %",
                f"CAGR SENSEX                   : {round(100*ep.cagr(df_portfolio_return['SENSEX']), 2)} %",]

    elif COUNTRY == 'US':
        line = line + [
                f"CAGR S&P 500                  : {round(100*ep.cagr(df_portfolio_return['S&P 500']),2)} %",
                f"CAGR NASDAQ 100               : {round(100*ep.cagr(df_portfolio_return['NASDAQ 100']),2)} %",
                f"CAGR RUSSELL 3000             : {round(100*ep.cagr(df_portfolio_return['RUSSELL 3000']),2)} %",]

    line = line + [
                f"SHARPE Portfolio              : {round(ep.sharpe_ratio(df_portfolio_return[column_names[0]]), 2)}",]

    if COUNTRY == 'IN':
        line = line + [
                f"SHARPE NIFTY 50               : {round(ep.sharpe_ratio(df_portfolio_return['NIFTY 50']), 2)}",
                f"SHARPE NIFTY 500              : {round(ep.sharpe_ratio(df_portfolio_return['NIFTY 500']), 2)}",
                f"SHARPE SENSEX                 : {round(ep.sharpe_ratio(df_portfolio_return['SENSEX']), 2)}",]

    elif COUNTRY == 'US':
        line = line + [
                f"SHARPE S&P 500                : {round(ep.sharpe_ratio(df_portfolio_return['S&P 500']),2)}",
                f"SHARPE NASDAQ 100             : {round(ep.sharpe_ratio(df_portfolio_return['NASDAQ 100']),2)}",
                f"SHARPE RUSSELL 3000           : {round(ep.sharpe_ratio(df_portfolio_return['RUSSELL 3000']),2)}",]

    line = line + [
                f"MAX-DRAWDOWN Portfolio        : {round(-100*ep.max_drawdown(df_portfolio_return[column_names[0]]), 2)} %",]

    if COUNTRY == 'IN':
        line = line + [
                f"MAX-DRAWDOWN NIFTY 50         : {round(-100*ep.max_drawdown(df_portfolio_return['NIFTY 50']), 2)} %",
                f"MAX-DRAWDOWN NIFTY 500        : {round(-100*ep.max_drawdown(df_portfolio_return['NIFTY 500']), 2)} %",
                f"MAX-DRAWDOWN SENSEX           : {round(-100*ep.max_drawdown(df_portfolio_return['SENSEX']), 2)} %"]

    elif COUNTRY == 'US':
        line = line + [
                f"MAX-DRAWDOWN S&P 500          : {round(-100*ep.max_drawdown(df_portfolio_return['S&P 500']),2)} %",
                f"MAX-DRAWDOWN NASDAQ 100       : {round(-100*ep.max_drawdown(df_portfolio_return['NASDAQ 100']),2)} %",
                f"MAX-DRAWDOWN RUSSELL 3000     : {round(-100*ep.max_drawdown(df_portfolio_return['RUSSELL 3000']),2)} %",]


    # Joining the list to form a multi-line string
    line = '\n'.join(line) + '\n\n' + str(lines)

    Summary_file = os.path.join(EXP_DIR, "Summary_file.txt")
    with open(Summary_file, "w") as file:
        file.write(line + "\n")

        # print(line)


            

        # #     # ================ Play sound when execution finished  ===================
        # #     # if platform.system() == 'Darwin':
        # #     #     os.system('say "your program has finished"')
        # #     # else:
        # #     #     import winsound
        # #     #     duration = 1000  # milliseconds
        # #     #     freq = 440  # Hz
        # #     #     winsound.Beep(freq, duration)

        # # name = "Kundan Kumar"
        # # email = "erkundanec@gmail.com"
        # # message = "Your VGM portfolio execution is finished."
        # # formspree_email = 'f/xjvnpbrd'     # this code will be provided once user will create own account in formspree

        # # send_email(name, email, message, formspree_email)

        # print(f"No. of tickers in the portfolio: {N_TOP_TICKERS}")

def compute_industrial_momentum_score(df_tics_daily, df_value_growth,formation_date):
    df_industry_mom = pd.DataFrame([])
    for ind in df_value_growth.industry.unique():
        tickers_list = df_value_growth[df_value_growth['industry'] == ind].tic.to_list()
        df_tics_daily_ind = df_tics_daily[df_tics_daily.tic.isin(tickers_list)]
        # print(f'{ind} = {len(tickers_list)} and {len(df_tics_daily_ind)}')

        MS = Momentum_Score_updated(formation_date)
        df_momentum = MS.compute_momentum_score(df_tics_daily_ind)

        df_momentum = rank_scores(df_momentum, 'momentum_score')

        momentum_columns = [f'momentum_{period}' for period in MOMENTUM_PERIOD_LIST]
        for col in momentum_columns:
            df_momentum = rank_scores(df_momentum, col)

        df_momentum = df_momentum.sort_values(by='momentum_score',ascending=False)

        # df_tics_daily_temp = df_tics_daily_ind[df_tics_daily_ind['date'] <= pd.to_datetime(END_DATE)].reset_index(drop=True)
        # df_tics_daily_temp = pd.DataFrame(df_tics_daily_temp.pivot(index='date', columns='tic', values='close').iloc[-1]).reset_index()
        # df_tics_daily_temp.columns = ['tic','close']    
        # df_momentum = pd.merge(df_momentum, df_tics_daily_temp, on='tic', how='left')
        # df_momentum = df_momentum.sort_values(by='momentum_score',ascending=False)



        # df_tics_mom = compute_momentum_score(df_tics_daily_ind, formation_date, tickers_list)
        df_industry_mom = pd.concat([df_industry_mom, df_momentum], ignore_index= True)

    df_industry_mom = df_industry_mom.rename(columns = {'momentum_3_rank': 'ind_momentum_3_rank',
                                                        'momentum_6_rank': 'ind_momentum_6_rank',
                                                        'momentum_12_rank': 'ind_momentum_12_rank'})
    return df_industry_mom[['tic','ind_momentum_3_rank','ind_momentum_6_rank','ind_momentum_12_rank']]

def process_industry_dataframe_v1(df):
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

def main_prod():
    validate_combination(RATING_OR_SCORE, VGM_METHOD)                                     # Validate the combination of rating_or_score and vgm_method
    df_tics_daily, df_marketcap, df_sector, df_funda = load_raw_data()                    # Load raw data from the local directory
    logging.info(f'OHLCV and Marketcap data are loaded from local directory.')
    logging.info(f"Number of tickers in df_tics_daily: {len(df_tics_daily.tic.unique())}")
    logging.info(f"Number of tickers in df_marketcap: {len(df_marketcap.tic.unique())}")
    logging.info(f"Number of common tickers in df_tics_daily and df_marketcap: {len(set(df_tics_daily.tic.unique()).intersection(set(df_marketcap.tic.unique())))}")
    
    TICKERS_LIST = list(set(df_tics_daily.tic.unique())-set(INDICES_LIST))
    logging.info(f"Number of ticker in the world of stocks: {len(TICKERS_LIST)}")
    logging.info(f'Stock world: {NAME_STOCK_WORLD}')

    # ================ For production run, compute stock metrics   =================
    df_metrics = compute_stock_metrics(df_tics_daily[df_tics_daily['tic'].isin(TICKERS_LIST)])

    start_date_momentum = FORMATION_DATE - timedelta(days = 365 * MOM_VOLATILITY_PERIOD + 30)
    end_date_momentum= FORMATION_DATE

    tickers_list, df_marketcap = stock_filter_marketcap(df_marketcap, FORMATION_DATE, MARKETCAP_TH)

    LP = Load_n_Preprocess(tickers_list, start_date_momentum, end_date_momentum)
    df_tics_daily_window = LP.filter_data(df_tics_daily)
    df_tics_daily_window, tickers_list = LP.clean_daily_data(df_tics_daily_window, missing_values_allowed = 0.01)


    # value_coeff, growth_coeff = load_factors_coefficients()
    # df_new_score = compute_value_growth_score(FORMATION_DATE, value_coeff, growth_coeff)
    if COUNTRY == 'IN':
        df_vgm_score = compute_vgm_score(df_tics_daily, formation_date, tickers_list, df_marketcap, df_funda, df_sector, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
        if PRODUCTION_FLAG:
            production_vgm_score_IN(df_vgm_score)

    elif COUNTRY == 'US':
        df_vgm_score = compute_vgm_score(df_tics_daily_window, FORMATION_DATE, tickers_list, df_marketcap, df_funda, df_sector, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
    
        df_momentum_ind = compute_industrial_momentum_score(df_tics_daily_window, df_vgm_score,FORMATION_DATE)
        df_vgm_score = pd.merge(df_vgm_score, df_momentum_ind, on='tic', how='inner')

        production_vgm_score_US(df_vgm_score,df_sector,df_marketcap)

    df_vgm_score = filter_stock_world(df_vgm_score,formation_date)

    df_vgm_score_marketcap = df_vgm_score.merge(df_marketcap[['tic','marketCap']], on = 'tic', how = 'left')
    if SAVE_EXCEL:
        df_vgm_score_marketcap.to_excel(os.path.join(EXCEL_DIR, f'df_vgm_score_{window_start_date.strftime("%Y-%m-%d")}.xlsx'), index=False)

    df_vgm_score_top = df_vgm_score_marketcap.iloc[0:N_TOP_TICKERS]

    hours, minutes, seconds = computation_time(start_time, "Total execution time: ")



# defin main function
if __name__ == "__main__":
    # main()
    main_prod()