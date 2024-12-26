'''===================== 20240726_Quant_Portfolio ===================== %
Description                  : Evaluate VGM Score using Quant Portfolio
Input parameter              : IN/US Stock historical close price and VGM Score
Output parameter             : Cumulative return of Portfolio and Indices
Subroutine  called           : NA
Called by                    : NA
Reference                    :
Author of the code           : Dr.Kundan Kumar
Date of creation             : 16/08/2024
------------------------------------------------------------------------------------------ %
Modified on                  : 27/09/2024
Modification details         : All strateges are implemened in here.
Modified By                  : Dr.Kundan Kumar
Previous version             : 20240726_Quant_Portfolio_v3_parametric
========================================================================================== %'''

# ===========   disable all warnings   ======================================
import warnings
warnings.filterwarnings("ignore")

# ===========     Import libraries   ==========================================
import os
import platform
import sys
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import scipy.stats as stats
import math
from empyrical import max_drawdown
import numpy as np
import empyrical as ep
import logging
from sklearn.cluster import KMeans
import random
import shutil

def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' has been deleted.")
    except FileNotFoundError:
        print(f"Directory '{dir_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred: {e}")


# ===========    define start time   =========================================
import time
start_time = time.time()

# ===========     Import User defined         =====================
sys.path.append('FinRecipes/')                                       # comment this line for server
from shared_vgm import check_and_make_directories, computation_time
from shared_vgm import first_day_month_list, weekday_dates_list    #,third_fridays_dates_list, first_monday_dates_list
from shared_vgm import Load_n_Preprocess
from shared_vgm import buy_hold_portfolio_return
from shared_vgm import compute_vgm_score  #, compute_vgm_score_US
from shared_vgm import send_email
from shared_vgm import fmp_download_hist_marketcap
from shared_vgm import plot_returns, plot_returns_drawdown, plot_returns_range,plot_returns_range_v2,plot_returns_drawdown_v2,plot_returns_v2
from shared_vgm import diversification_score

import matplotlib.pyplot as plt
import matplotlib

# ==========  Parse Command-Line Arguments  ==============
import argparse  # New import for argument parsing
parser = argparse.ArgumentParser(description='Evaluate VGM Score using Quant Portfolio')
parser.add_argument('--RATING_OR_SCORE', type=str, required=True, help="'score' or 'rating'")
parser.add_argument('--VGM_METHOD', type=str, required=False, help="'min-max_avg' / 'percentile_avg' / 'z-score_avg' / 'rank_avg', 'mom-value' / 'mom-growth' / 'growth-value' / 'Avg-VGM' / 'no-momentum'")
parser.add_argument('--PORTFOLIO_FREQ', type=str, required=False, help="'Weekly'/ 'Fortnight' / 'Monthly'")
parser.add_argument('--N_TOP_TICKERS', type=str, required=False, help="'20', '25', '30'")
parser.add_argument('--PORTFOLIO_SIZE', type=str, required=False, help="'20', '25', '30'")
parser.add_argument('--CONSIDER_RISK_FACTOR', type=str, required=False, help="'yes' / 'no'")
parser.add_argument('--CONSIDER_DIV', type=str, required=False, help="'yes' / 'no'")
parser.add_argument('--START_DATE', type=str, required=False, help="'2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'")
parser.add_argument('--END_DATE', type=str, required=False, help="'2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31'")

args = parser.parse_args()

RATING_OR_SCORE = args.RATING_OR_SCORE
VGM_METHOD = args.VGM_METHOD
PORTFOLIO_FREQ = args.PORTFOLIO_FREQ
N_TOP_TICKERS = args.N_TOP_TICKERS
PORTFOLIO_SIZE = args.PORTFOLIO_SIZE
CONSIDER_RISK_FACTOR = args.CONSIDER_RISK_FACTOR
CONSIDER_DIV = args.CONSIDER_DIV
START_DATE = args.START_DATE
END_DATE = args.END_DATE


# Define your valid options
RATING_OR_SCORE_OPTIONS = ['score', 'rating']
VGM_METHOD_OPTIONS = ['min-max_avg', 'percentile_avg', 'z-score_avg', 'only-momentum', 'rank_avg', 'value','growth','mom-value', 'mom-growth','value-mom', 'growth-mom', 'growth-value-mom','growth-value']

# Define the valid combinations
valid_combinations = {
    'score': ['min-max_avg', 'percentile_avg', 'z-score_avg', 'rank_avg', 'only-momentum'],
    'rating': ['value','growth','mom-value', 'mom-growth','value-mom', 'growth-mom', 'growth-value-mom','growth-value']
}

def validate_combination(rating_or_score, vgm_method):
    # Check if the combination is valid
    if rating_or_score in valid_combinations and vgm_method in valid_combinations[rating_or_score]:
        print(f"Running {vgm_method} for {rating_or_score}...")
        # Place your code for the respective combinations here
    else:
        print(f"Invalid combination: {rating_or_score} with {vgm_method}. Terminating...")
        sys.exit()  # Terminate the program if invalid combination

# Example usage
validate_combination(RATING_OR_SCORE, VGM_METHOD)  # Valid

HEAD_STR = dt.datetime.now().strftime("%Y%m%d-%H%M")

ROOT_DIR = "./examples/Quant_Rating/"
PATH_DATA = ROOT_DIR + "datasets/"

# ==================   Initialization  ==========================================
from config_po_vgm import (
    COUNTRY,
    USE_BNH,
    TC,
    DEBUG,
    SAVE_EXCEL,
    MAJOR_INDEX,
    MARKETCAP_TH,
    MOM_VOLATILITY_PERIOD,
    RISK_VOLATILITY_PERIOD,

    DOWNLOAD_OHLCV_DATA,
    DOWNLOAD_HIST_MARKETCAP,
    N_YEARS_HIST_MARKETCAP,
    N_YEARS_HIST_OHLCV,
    LIVE_DATA,
    TICKERS_LIST,
    PATH_DAILY_DATA,
    PATH_MARKETCAP_DATA,
    INDICES_LIST,
    NAME_STOCK_WORLD,
    MOMENTUM_PERIOD_LIST,
    RISK_PERIOD_LIST,
    
    WITH_TRANS_COST,
    BROKER,
    INITIAL_AMOUNT,
    DEBUG,

    SCALING_METHOD,
    LARGE_CAP_FILTER,
    MID_CAP_FILTER,
    SMALL_CAP_FILTER,
)

if DEBUG:
    EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + NAME_STOCK_WORLD + '_' + 'DEBUG' +'_' + COUNTRY +'/'
else:
    if RATING_OR_SCORE == 'rating':
        EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_'  + RATING_OR_SCORE + '_' + VGM_METHOD + '_' + CONSIDER_RISK_FACTOR + '_' + CONSIDER_DIV + '_' + START_DATE + '_' + N_TOP_TICKERS + '_' + PORTFOLIO_SIZE + '_' + PORTFOLIO_FREQ +'_' + COUNTRY +'/'

    if RATING_OR_SCORE == 'score':
        EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + RATING_OR_SCORE + '_' + VGM_METHOD + '_' + CONSIDER_RISK_FACTOR + '_' + CONSIDER_DIV + '_' + START_DATE + '_' + N_TOP_TICKERS + '_' + PORTFOLIO_SIZE + '_' + PORTFOLIO_FREQ +'_' + COUNTRY +'/'


if SAVE_EXCEL == True:
    RESULTS_DIR = EXPERIMENT_DIR + 'results/'
else:
    RESULTS_DIR = EXPERIMENT_DIR

N_TOP_TICKERS = int(N_TOP_TICKERS)
PORTFOLIO_SIZE = int(PORTFOLIO_SIZE)

START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE) 

check_and_make_directories([EXPERIMENT_DIR, RESULTS_DIR])

# Configure logging
log_file = EXPERIMENT_DIR + 'Run_History.log'

# Check if log file exists, otherwise create one
if not os.path.exists(log_file):
    open(log_file, 'w').close()

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format without milliseconds

if DOWNLOAD_OHLCV_DATA:
    TICKERS_INDICES_LIST = TICKERS_LIST + INDICES_LIST
    start_download = time.time()
    PERIOD = 365*N_YEARS_HIST_OHLCV
    END_DATE_OHLCV = dt.datetime.now()
    START_DATE_OHLCV = END_DATE_OHLCV - dt.timedelta(days= PERIOD)
    LP = Load_n_Preprocess(TICKERS_INDICES_LIST, START_DATE_OHLCV, END_DATE_OHLCV)
    df_tics_daily = LP.download_yfinance(is_live = LIVE_DATA)
    # print(f"Historical ohlcv data downloaded till date: {df_tics_daily.sort_values(by = ['date'])['date'].iloc[-1]}")
    logging.info(f"Historical ohlcv data downloaded till date: {df_tics_daily.sort_values(by = ['date'])['date'].iloc[-1]}")
    TICKERS_INDICES_LIST = df_tics_daily['tic'].unique().tolist()
    # print(f"len(TICKERS_INDICES_LIST)")
    logging.info(f"len(TICKERS_INDICES_LIST)")
    df_tics_daily.to_hdf(PATH_DAILY_DATA,"df",mode = 'w')
    hours, minutes, seconds = computation_time(start = start_download, message = f"Total time to download OHLCV data of {NAME_STOCK_WORLD}")
   
else:
    df_tics_daily = pd.read_hdf(PATH_DAILY_DATA, "df",mode = 'r')
    print(f'Data loaded from local directory')
    logging.info(f'Data loaded from local directory')

if DOWNLOAD_HIST_MARKETCAP:                  # download time 00:27:59 (IN)
    # TICKERS_LIST = TICKERS_LIST[:3]
    start_download = time.time()
    PERIOD = 365*N_YEARS_HIST_MARKETCAP
    END_DATE_MARKETCAP = dt.datetime.now()
    START_DATE_MARKETCAP = END_DATE_MARKETCAP - dt.timedelta(days= PERIOD)
    df_marketcap = fmp_download_hist_marketcap(TICKERS_LIST, START_DATE_MARKETCAP, END_DATE_MARKETCAP, filename = PATH_MARKETCAP_DATA[:-3])
    # print(f"Historical data downloaded till date: {df_marketcap.sort_values(by = ['date'])['date'].iloc[-1]}")
    logging.info(f"Historical data downloaded till date: {df_marketcap.sort_values(by = ['date'])['date'].iloc[-1]}")
    hours, minutes, seconds = computation_time(start = start_download, message = f"Total time to download historical Marketcap data of {NAME_STOCK_WORLD}")

else:
    df_marketcap = pd.read_hdf(PATH_MARKETCAP_DATA, "df", mode = 'r')

df_marketcap['date'] = pd.to_datetime(df_marketcap['date'])

# START_DATE = args.START_DATE
# END_DATE = args.END_DATE
# START_DATE = pd.to_datetime(START_DATE)
# END_DATE = pd.to_datetime(END_DATE) 

TICKERS_LIST_WORLD = list(set(df_tics_daily.tic.unique())-set(INDICES_LIST))
# print(f"Number of ticker in the world of stocks: {len(TICKERS_LIST_WORLD)}")
# print(f'Stock world: {NAME_STOCK_WORLD}')
logging.info(f"Number of ticker in the world of stocks: {len(TICKERS_LIST_WORLD)}")
logging.info(f'Stock world: {NAME_STOCK_WORLD}\n')


# ================ Create List of Dates   ======================================
 
if PORTFOLIO_FREQ == 'Monthly':
    date_list = first_day_month_list(START_DATE, END_DATE)                  # on these dates tickers are updated
    date_list = [date for date in date_list if date <= df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
    for idx in np.arange(len(date_list)):
        while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
            date_list[idx] = date_list[idx] + timedelta(days = 1)   

elif PORTFOLIO_FREQ == 'Weekly':
    date_list = weekday_dates_list(START_DATE, END_DATE, weekday = 4)     # 0: Monday, 4: Friday 
    date_list = [date for date in date_list if date < df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
    for idx in np.arange(len(date_list)):
        while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
            date_list[idx] = date_list[idx] - timedelta(days = 1)   

elif PORTFOLIO_FREQ == 'Fortnight':
    date_list = weekday_dates_list(START_DATE, END_DATE, weekday = 4)     # 0: Monday, 4: Friday 
    date_list = [date for date in date_list if date < df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
    date_list = date_list[::2]
    for idx in np.arange(len(date_list)):
        while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
            date_list[idx] = date_list[idx] - timedelta(days = 1)   
# date_list = third_fridays_dates_list(start_date, end_date)
# date_list = first_monday_dates_list(START_DATE, END_DATE)

# # =============  Ensure all dates are bussiness day ====================

          

# append last date if not present in date_list
if END_DATE.date() not in date_list:        
    date_list.append(END_DATE.date())           

if DEBUG:
    start_point = -2                       # uncomment to evaluate last month only
    date_list = date_list[start_point:]
    print(f"List of dates: {[datu.strftime('%Y-%m-%d') for datu in date_list]}")

if USE_BNH:
    # ================  Buy-hold strategy  ====================================
    start_bnh = time.time()
    # periodic_tickers_list_bnh = RESULTS_DIR + 'periodic_tickers_list_bnh.txt'
    



    # if os.path.exists(periodic_tickers_list_bnh):
    #     os.remove(periodic_tickers_list_bnh)

    # ============  Initialize firstday return to zero  ======================
    firstday_bnh_return = pd.DataFrame(0,columns=['Buy_Hold_returns'], index=[date_list[0]])
    df_return_list = [firstday_bnh_return - TC]

    industry_counts_list = []
    lines = []

    monthly_return = {}
    monthly_return_index = {}
    for idx in range(len(date_list)-1):
        window_start_date = date_list[idx]
        window_end_date = date_list[idx + 1]                              # - timedelta(days=1)
        logging.info(f"Trading period = {window_start_date} - {window_end_date}")
        print(f"Trading period = {window_start_date} - {window_end_date}")

        FORMATION_DATE = window_start_date
        START_DATE_MONTH = FORMATION_DATE - timedelta(days = 365*MOM_VOLATILITY_PERIOD)
        END_DATE_MONTH = FORMATION_DATE

        df_marketcap_pivot = df_marketcap.pivot_table(index = 'date', columns = 'symbol',values='marketCap')
        temp_mktcap = df_marketcap_pivot[df_marketcap_pivot.index <= pd.to_datetime(FORMATION_DATE)]
        temp_mktcap = temp_mktcap.iloc[-1]
        df_rank_marketcap = pd.DataFrame(temp_mktcap).reset_index()
        df_rank_marketcap.columns = ['tic','marketcap']
        df_rank_marketcap = df_rank_marketcap.sort_values(by=['marketcap'],ascending=False)
        df_rank_marketcap = df_rank_marketcap.reset_index(drop = True)
        logging.info(f"Number of tickers for which marketcap is available = {len(df_rank_marketcap)} - {df_rank_marketcap.isna().sum().values[1]} = {len(df_rank_marketcap)- df_rank_marketcap.isna().sum().values[1]} tickers")

        TICKERS_LIST_WORLD = df_rank_marketcap[df_rank_marketcap['marketcap'] >= MARKETCAP_TH]['tic'].tolist()   #.index.tolist()
        # print(f"Number of tickers after removing tickers with marketcap < {int(MARKETCAP_TH/1000000)}M = {len(TICKERS_LIST_WORLD)}")
        logging.info(f"Number of tickers after removing tickers with marketcap < {int(MARKETCAP_TH/1000000)}M = {len(TICKERS_LIST_WORLD)}")

        # # Sort the DataFrame by 'marketcap' in descending order
        # df_rank_marketcap = df_rank_marketcap.head(100)
        # TICKERS_LIST_WORLD = df_marketcap['tic'].tolist()

        LP = Load_n_Preprocess(TICKERS_LIST_WORLD, START_DATE_MONTH, END_DATE_MONTH, path_daily_data = PATH_DAILY_DATA)
        df_tics_daily = LP.load_daily_data()
        df_tics_daily = LP.clean_daily_data(df_tics_daily, missing_values_allowed = 0.01)
        # df_tics_weekly = LP.convert_daily(df_tics_daily, timeframe = 'W')     # W - weekly   M - Monthly

        TICKERS_LIST_WORLD = df_tics_daily['tic'].unique().tolist()
        # print(f"Number of tickers after removing tickers with missing values = {len(TICKERS_LIST_WORLD)}")
        logging.info(f"Number of tickers after removing tickers with missing values (at least 3 yrs) = {len(TICKERS_LIST_WORLD)}")

        # value_coeff, growth_coeff = load_factors_coefficients()
        # df_new_score = compute_value_growth_score(FORMATION_DATE, value_coeff, growth_coeff)
        # if COUNTRY == 'IN':
        df_vgm_score = compute_vgm_score(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
            # df_vgm_score.to_excel(EXPERIMENT_DIR + 'df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
        # elif COUNTRY == 'US':
            # df_vgm_score = compute_vgm_score_US(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
            # df_vgm_score.to_excel(EXPERIMENT_DIR + 'df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
        
        # df_vgm_score_marketcap = df_vgm_score.merge(df_rank_marketcap, on = 'tic', how = 'left')


        # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
        if SAVE_EXCEL:
            df_vgm_score.to_excel(RESULTS_DIR + 'df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
        # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
        logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score[df_vgm_score["close"] < 10])}')
        
        # elif VGM_SCORE_METHOD == 'min-max_avg':
        #     df_vgm_score = compute_vgm_score(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, rank_based_on = 'both')
       

        #######################  Historical Large + Midcap Stocks US ============================
        if COUNTRY == 'US':
            if LARGE_CAP_FILTER:
                hist_constituents = 's&p500_component_history.csv'
            
            elif MID_CAP_FILTER:
                hist_constituents = 's&p900_component_history.csv'
            
            elif SMALL_CAP_FILTER:
                hist_constituents = 's&p1500_component_history.csv'


            df_lar_mid_tickers = pd.read_csv(PATH_DATA + hist_constituents)
            df_lar_mid_tickers['Date'] = pd.to_datetime(df_lar_mid_tickers['Date'])
            df_lar_mid_tickers = df_lar_mid_tickers[df_lar_mid_tickers['Date'] == df_lar_mid_tickers[df_lar_mid_tickers['Date'] <= pd.to_datetime(FORMATION_DATE)].sort_values(by=['Date']).iloc[-1]['Date']].reset_index(drop = True)
            
            df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_lar_mid_tickers['Code'])]
            # df_vgm_score = df_vgm_score.sort_values(by = 'vgm_score', ascending = False).reset_index(drop = True)

        if COUNTRY ==  'IN':
            if LARGE_CAP_FILTER:
                T = 100
                df_vgm_score_mcap_sort = df_vgm_score.copy()
                df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
                df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

            elif MID_CAP_FILTER:
                T = 250
                df_vgm_score_mcap_sort = df_vgm_score.copy()
                df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
                df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

            elif MID_CAP_FILTER:
                T = 1000
                df_vgm_score_mcap_sort = df_vgm_score.copy()
                df_vgm_score_mcap_sort = df_vgm_score_mcap_sort.sort_values(by = ['marketcap'], ascending = False).head(T)
                df_vgm_score = df_vgm_score[df_vgm_score['tic'].isin(df_vgm_score_mcap_sort['tic'])]

            else:
                df_vgm_score = df_vgm_score[df_vgm_score['marketcap'] >= 20000]
                print(f"number of tickers in midcap = {len(df_vgm_score)}")


            # print(f"Test")


        df_vgm_score_top = df_vgm_score.iloc[0:N_TOP_TICKERS] 
        df_vgm_score_marketcap = df_vgm_score_top.merge(df_rank_marketcap, on = 'tic', how = 'left')
        # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
        # if SAVE_EXCEL:
        #     df_vgm_score_marketcap.to_excel(RESULTS_DIR + 'top_df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
        # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
        logging.info(f'Number of tickers in top {N_TOP_TICKERS} less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')
        
        # =========== Industrial Count - Diversification ======================
        # tickers_list = df_vgm_score_top['tic']   #df_vgm_score['tic'].iloc[0:N_TOP_TICKERS]             # selected tickers based on momentum
        # tickers_list = list(tickers_list)
        
        # with open(periodic_tickers_list_bnh, "a") as file:
        #     tickers_list.sort()
        #     line = ', '.join([f"({item})" for item in tickers_list])
        #     line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
        #     file.write(line + "\n")

        if CONSIDER_DIV == 'yes':
            # Function to compute stock returns
            def StockReturnsComputing(stock_price, rows, cols):            
                stock_return = np.zeros([rows-1, cols])
                for j in range(cols):  # j: Assets
                    for i in range(rows-1):  # i: Daily Prices
                        stock_return[i, j] = (stock_price[i+1, j] - stock_price[i, j]) / stock_price[i, j]
                return stock_return

            # Prepare stock price data
            tickers_list = df_vgm_score_top['tic']
            df_stock_prices = df_tics_daily[df_tics_daily['tic'].isin(tickers_list)].pivot_table(index='date', columns='tic', values='close')
            asset_labels = df_stock_prices.columns.tolist()
            ar_stock_prices = np.asarray(df_stock_prices)
            rows, cols = ar_stock_prices.shape

            # Compute daily returns
            ar_returns = StockReturnsComputing(ar_stock_prices, rows, cols)

            # Compute mean returns and covariance matrix
            mean_returns = np.mean(ar_returns, axis=0).reshape(len(ar_returns[0]), 1)
            cov_returns = np.cov(ar_returns, rowvar=False)

            # Prepare asset parameters for KMeans clustering
            asset_parameters = np.concatenate([mean_returns, cov_returns], axis=1)

            # KMeans clustering
            clusters = PORTFOLIO_SIZE
            assets_cluster = KMeans(algorithm='lloyd', max_iter=600, n_clusters=clusters, n_init=5, random_state=42)
            assets_cluster.fit(asset_parameters)
            labels = assets_cluster.labels_

            assets = np.array(asset_labels)

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
            def create_random_portfolios(cluster_dict, num_portfolios=20):
                clusters_tickers = [tickers for tickers in cluster_dict.values()]
                
                # Generate 20 random portfolios by selecting one random ticker from each cluster
                portfolios = [tuple(random.choice(tickers) for tickers in clusters_tickers) for _ in range(num_portfolios)]
                
                return portfolios

            # Example usage: generate 20 random portfolios
            portfolios = create_random_portfolios(cluster_dict, num_portfolios=30)

            # Create random portfolios dictionary
            portfolio_dict = {f'Portfolio_{i+1}': list(portfolio) for i, portfolio in enumerate(portfolios)}

            # Initialize a dictionary to store portfolio scores
            portfolio_scores = {}

            # Compute diversification scores for each portfolio and store them in the dictionary
            for portfolio_num, tickers in portfolio_dict.items():
                score = diversification_score(df_tics_daily, tickers, method='equal-weight')
                portfolio_scores[portfolio_num] = score

            # Select the portfolio with the maximum diversification score
            max_portfolio = max(portfolio_scores, key=portfolio_scores.get)

            # Print the portfolio with the highest diversification score
            logging.info(f'The portfolio with the highest diversification score is {max_portfolio}: {portfolio_dict[max_portfolio]}')
            logging.info(f'Highest diversification score: {portfolio_scores[max_portfolio]}')

            tickers_list = portfolio_dict[max_portfolio]

            df_vgm_score_div = df_vgm_score_top[df_vgm_score_top['tic'].isin(tickers_list)]
            # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
            if SAVE_EXCEL:
                df_vgm_score_div.to_excel(RESULTS_DIR + 'div_df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
                # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
                # logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')
        
            if NAME_STOCK_WORLD == "RUSSELL-3000" and not (VGM_METHOD == 'only-momentum'):
                industry_counts = df_vgm_score_div['industry'].value_counts().reset_index()
                industry_counts.columns = ['industry', 'count']
                industry_counts['date'] = FORMATION_DATE.strftime("%Y-%m-%d")
                industry_counts_list.append(industry_counts)

        else:
            if NAME_STOCK_WORLD == "RUSSELL-3000" and not (VGM_METHOD == 'only-momentum'):
                # =========== Industrial Count - Diversification ======================
                top_30 = df_vgm_score.head(PORTFOLIO_SIZE)
                industry_counts = top_30['industry'].value_counts().reset_index()
                industry_counts.columns = ['industry', 'count']
                industry_counts['date'] = FORMATION_DATE.strftime("%Y-%m-%d")
                industry_counts_list.append(industry_counts)

            tickers_list = df_vgm_score['tic'].iloc[0:PORTFOLIO_SIZE]             # selected tickers based on momentum
            tickers_list = list(tickers_list)

        # Sort the tickers list and prepare the line
        tickers_list.sort()
        line = ', '.join([f"({item})" for item in tickers_list])
        line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
        lines.append(line)

        LP = Load_n_Preprocess(tickers_list, window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
        df_tics_daily = LP.load_daily_data()
        df_tics_daily = LP.clean_daily_data(df_tics_daily, missing_values_allowed = 0.01)

        df_return = buy_hold_portfolio_return(df_tics_daily)

        if idx == 0:
            df_return.iloc[-1] = df_return.iloc[-1] - TC
            df_return_list.append(df_return)
        else:
            df_return.iloc[0] = df_return.iloc[0] - TC
            df_return.iloc[-1] = df_return.iloc[-1] - TC
            df_return_list.append(df_return)

        try:
            monthly_return[date_list[idx]] = [round(100*ep.cum_returns_final(df_return),2).values[0]]
        except:
            print('Today is the inception day')
            sys.exit(1)  # Exit the program with a non-zero status

        LP = Load_n_Preprocess([MAJOR_INDEX], window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
        df_index_daily = LP.load_daily_data()
        df_return_index = buy_hold_portfolio_return(df_index_daily)
        monthly_return_index[date_list[idx]] = [round(100*ep.cum_returns_final(df_return_index),2).values[0]]

    df_return_bnh = pd.concat(df_return_list)

    if (NAME_STOCK_WORLD == "RUSSELL-3000") and not (VGM_METHOD == 'only-momentum'):
        df_industry_counts = pd.concat(industry_counts_list, ignore_index=True)
        if SAVE_EXCEL:
            df_industry_counts.to_excel(RESULTS_DIR + 'df_industry_div.xlsx', index = False)

    df_monthly_return = pd.DataFrame.from_dict(monthly_return, orient='index', columns=['return'])
    if SAVE_EXCEL:
        df_monthly_return.to_excel(RESULTS_DIR + 'df_monthly_return.xlsx', index = True)

    df_monthly_return_index = pd.DataFrame.from_dict(monthly_return_index, orient='index', columns=['return'])
    if SAVE_EXCEL:
        df_monthly_return_index.to_excel(RESULTS_DIR + 'df_monthly_return_index.xlsx', index = True)

    # ============================= Prepare data for to plot Buy-Hold returns  ======================
    df_return_bnh = df_return_bnh.reset_index()
    df_return_bnh.columns = ['date','Buy_Hold_Returns']
    df_return_bnh['date'] = pd.to_datetime(df_return_bnh['date'])
    df_return_bnh = df_return_bnh.set_index(['date'])

    # ===============================     Plotting  ====================================
    LP = Load_n_Preprocess(INDICES_LIST, date_list[0], date_list[-1], path_daily_data = PATH_DAILY_DATA)
    df_indices_daily = LP.load_daily_data()
    # df_indices_daily = LP.clean_daily_data(df_indices_daily, missing_values_allowed = 0.01)

    df_indices_close = pd.pivot_table(df_indices_daily, values ='close', index = 'date',columns='tic')
    df_indices_return = df_indices_close.pct_change().fillna(0.001)
    # df_return_bnh['Portfolio_Trend'] = df_return_bnh['Buy_Hold_returns'].rolling(window=TREND_WINDOW).mean()
    df_return_bnh = df_return_bnh.merge(df_indices_return,how='left',on = 'date')
    # df_return_bnh = df_return_bnh.drop(columns=['Portfolio_Trend'])

    if NAME_STOCK_WORLD == "RUSSELL-3000":
        df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^GSPC': 'S&P 500','IJS': 'S&P 600', '^NDX': 'NASDAQ 100','^SP600': 'S&P 600','^RUT':'RUSSELL 2000','^RUA':'RUSSELL 3000'})
        # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
        if SAVE_EXCEL:
            df_return_bnh.to_excel(RESULTS_DIR + '/' + 'df_return_daily_monthly_ticker_bnh.xlsx')
        # plot_returns(df_return_bnh, tickers_list = [], filename = RESULTS_DIR + '/' + 'Portfolio_BnH', period = 'daily', name_stock_world = NAME_STOCK_WORLD)

    if NAME_STOCK_WORLD == "NSE-Stocks":
        if PORTFOLIO_FREQ == 'Monthly':
            df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        elif PORTFOLIO_FREQ == 'Weekly':
            df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        elif PORTFOLIO_FREQ == 'Fortnight':
            df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
            df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
        
       
        # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
        if SAVE_EXCEL:
            df_return_bnh.to_excel(RESULTS_DIR + '/' + 'df_return_daily_monthly_ticker_bnh.xlsx')
        
    plot_returns_range_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2024-01-01', end_date='2024-12-31')
    plot_returns_range_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2023-01-01', end_date='2023-12-31')
    plot_returns_range_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2022-01-01', end_date='2022-12-31')
    plot_returns_range_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2021-01-01', end_date='2021-12-31')
    plot_returns_range_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2020-01-01', end_date='2020-12-31')
    
    plot_returns_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD)
    plot_returns_drawdown_v2(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + 'Portfolio_drawdown', period = 'daily', name_stock_world = NAME_STOCK_WORLD, pos = 'lower right')

hours, minutes, seconds = computation_time(start_time, "Total execution time: ")

column_names = df_return_bnh.columns.tolist()
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
            f"Marketcap threshold           : Rs. {int(MARKETCAP_TH/10000000)} Cr.",
            f"Largecap Portfolio            : {LARGE_CAP_FILTER}",
            f"Midcap Portfolio              : {MID_CAP_FILTER}",
            f"Smallcap Portfolio            : {SMALL_CAP_FILTER}",
            ]

elif COUNTRY == 'US':
    line = line + [
            f"Marketcap threshold           : ${int(MARKETCAP_TH/1000000)} M",
            f"Largecap Portfolio             : {LARGE_CAP_FILTER} (S&P 500)",
            f"Midcap Portfolio             : {MID_CAP_FILTER} (S&P 900)",
            f"Smallcap Portfolio            : {SMALL_CAP_FILTER} (S&P 1500)",
            ]


# Ensure all the required variables are defined before using them
line = line + [
            f"Rating or Score               : {RATING_OR_SCORE}",
            f"VGM method                    : {VGM_METHOD}",
            f"Scaling method                : {SCALING_METHOD}",
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
            f"CAGR Portfolio                : {round(100*ep.cagr(df_return_bnh[column_names[0]]), 2)} %",]

if COUNTRY == 'IN':
    line = line + [
            f"CAGR NIFTY 50                 : {round(100*ep.cagr(df_return_bnh['NIFTY 50']), 2)} %",
            f"CAGR NIFTY 500                : {round(100*ep.cagr(df_return_bnh['NIFTY 500']), 2)} %",
            f"CAGR SENSEX                   : {round(100*ep.cagr(df_return_bnh['SENSEX']), 2)} %",]

elif COUNTRY == 'US':
    line = line + [
            f"CAGR S&P 500                  : {round(100*ep.cagr(df_return_bnh['S&P 500']),2)} %",
            f"CAGR NASDAQ 100               : {round(100*ep.cagr(df_return_bnh['NASDAQ 100']),2)} %",
            f"CAGR RUSSELL 3000             : {round(100*ep.cagr(df_return_bnh['RUSSELL 3000']),2)} %",]

line = line + [
            f"SHARPE Portfolio              : {round(ep.sharpe_ratio(df_return_bnh[column_names[0]]), 2)}",]

if COUNTRY == 'IN':
    line = line + [
            f"SHARPE NIFTY 50               : {round(ep.sharpe_ratio(df_return_bnh['NIFTY 50']), 2)}",
            f"SHARPE NIFTY 500              : {round(ep.sharpe_ratio(df_return_bnh['NIFTY 500']), 2)}",
            f"SHARPE SENSEX                 : {round(ep.sharpe_ratio(df_return_bnh['SENSEX']), 2)}",]

elif COUNTRY == 'US':
    line = line + [
            f"SHARPE S&P 500                : {round(ep.sharpe_ratio(df_return_bnh['S&P 500']),2)}",
            f"SHARPE NASDAQ 100             : {round(ep.sharpe_ratio(df_return_bnh['NASDAQ 100']),2)}",
            f"SHARPE RUSSELL 3000           : {round(ep.sharpe_ratio(df_return_bnh['RUSSELL 3000']),2)}",]

line = line + [
            f"SORTINO Portfolio             : {round(ep.sortino_ratio(df_return_bnh[column_names[0]]), 2)}",]

if COUNTRY == 'IN':
    line = line + [
            f"SORTINO NIFTY 50              : {round(ep.sortino_ratio(df_return_bnh['NIFTY 50']), 2)}",
            f"SORTINO NIFTY 500             : {round(ep.sortino_ratio(df_return_bnh['NIFTY 500']), 2)}",
            f"SORTINO SENSEX                : {round(ep.sortino_ratio(df_return_bnh['SENSEX']), 2)}",]

elif COUNTRY == 'US':
    line = line + [
            f"SORTINO S&P 500               : {round(ep.sortino_ratio(df_return_bnh['S&P 500']),2)}",
            f"SORTINO NASDAQ 100            : {round(ep.sortino_ratio(df_return_bnh['NASDAQ 100']),2)}",
            f"SORTINO RUSSELL 3000          : {round(ep.sortino_ratio(df_return_bnh['RUSSELL 3000']),2)}",]

line = line + [
            f"MAX-DRAWDOWN Portfolio        : {round(-100*ep.max_drawdown(df_return_bnh[column_names[0]]), 2)} %",]

if COUNTRY == 'IN':
    line = line + [
            f"MAX-DRAWDOWN NIFTY 50         : {round(-100*ep.max_drawdown(df_return_bnh['NIFTY 50']), 2)} %",
            f"MAX-DRAWDOWN NIFTY 500        : {round(-100*ep.max_drawdown(df_return_bnh['NIFTY 500']), 2)} %",
            f"MAX-DRAWDOWN SENSEX           : {round(-100*ep.max_drawdown(df_return_bnh['SENSEX']), 2)} %"]

elif COUNTRY == 'US':
    line = line + [
            f"MAX-DRAWDOWN S&P 500          : {round(-100*ep.max_drawdown(df_return_bnh['S&P 500']),2)} %",
            f"MAX-DRAWDOWN NASDAQ 100       : {round(-100*ep.max_drawdown(df_return_bnh['NASDAQ 100']),2)} %",
            f"MAX-DRAWDOWN RUSSELL 3000     : {round(-100*ep.max_drawdown(df_return_bnh['RUSSELL 3000']),2)} %",]

line = line + [
            f"Std Dev Portfolio             : {round(100*df_return_bnh[column_names[0]].std(), 2)} %",]

if COUNTRY == 'IN':
    line = line + [
            f"Std Dev NIFTY 50              : {round(100*df_return_bnh['NIFTY 50'].std(), 2)} %",
            f"Std Dev NIFTY 500             : {round(100*df_return_bnh['NIFTY 500'].std(), 2)} %",
            f"Std Dev SENSEX                : {round(100*df_return_bnh['SENSEX'].std(), 2)} %"]

elif COUNTRY == 'US':
    line = line + [
            f"Std Dev S&P 500               : {round(100*df_return_bnh['S&P 500'].std(),2)} %",
            f"Std Dev NASDAQ 100            : {round(100*df_return_bnh['NASDAQ 100'].std(),2)} %",
            f"Std Dev RUSSELL 3000          : {round(100*df_return_bnh['RUSSELL 3000'].std(),2)} %",]


# Joining the list to form a multi-line string
line = '\n'.join(line) + '\n\n' + str(lines)

Summary_file = EXPERIMENT_DIR + "Summary_file.txt"
with open(Summary_file, "w") as file:
    file.write(line + "\n")

    # print(line)


     

#     # ================ Play sound when execution finished  ===================
#     # if platform.system() == 'Darwin':
#     #     os.system('say "your program has finished"')
#     # else:
#     #     import winsound
#     #     duration = 1000  # milliseconds
#     #     freq = 440  # Hz
#     #     winsound.Beep(freq, duration)

# name = "Kundan Kumar"
# email = "erkundanec@gmail.com"
# message = "Your VGM portfolio execution is finished."
# formspree_email = 'f/xjvnpbrd'     # this code will be provided once user will create own account in formspree

# send_email(name, email, message, formspree_email)

print(f"No. of tickers in the portfolio: {PORTFOLIO_SIZE}")