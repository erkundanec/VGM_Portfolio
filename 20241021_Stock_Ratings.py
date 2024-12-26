'''===================== 20240726_Quant_Portfolio ===================== %          azure file "Stock_Factor_Rating"
Description                  : Evaluate VGM Score using Quant Portfolio
Input parameter              : Indian Stock historical close price and VGM Score
Output parameter             : Cumulative return of Portfolio and Indices
Subroutine  called           : NA
Called by                    : NA
Reference                    :
Author of the code           : Dr.Kundan Kumar
Date of creation             : 16/08/2024
------------------------------------------------------------------------------------------ %
Modified on                  : 27/09/2024
Modification details         : All strateges are in a single file.
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
sys.path.append('./FinRecipes/')                                       # comment this line for server
from shared_vgm import check_and_make_directories, computation_time
from shared_vgm import first_day_month_list, weekday_dates_list    #,third_fridays_dates_list, first_monday_dates_list
from shared_vgm import Load_n_Preprocess
from shared_vgm import buy_hold_portfolio_return
from shared_vgm import compute_vgm_score
from shared_vgm import send_email
from shared_vgm import fmp_download_hist_marketcap
from shared_vgm import plot_returns, plot_returns_drawdown, plot_returns_range
from shared_vgm import diversification_score

import matplotlib.pyplot as plt
import matplotlib

RATING_OR_SCORE = 'score'
VGM_METHOD = 'min-max_avg'
PORTFOLIO_FREQ = 'Monthly'
N_TOP_TICKERS = '100'
PORTFOLIO_SIZE = '30'
CONSIDER_RISK_FACTOR = 'yes'
CONSIDER_DIV = 'yes'
START_DATE = '2020-10-01'
END_DATE = '2024-12-31'
FORMATION_DATE = pd.Timestamp.today().date()

# Define your valid options
RATING_OR_SCORE_OPTIONS = ['score', 'rating']
VGM_METHOD_OPTIONS = ['min-max_avg', 'percentile_avg', 'z-score_avg', 'only-momentum', 'rank_avg', 'value', 'growth', 'mom-value', 'mom-growth', 'growth_value', 'Avg_VGM', 'no-momentum']

# Define the valid combinations
valid_combinations = {
    'score': ['min-max_avg', 'percentile_avg', 'z-score_avg', 'rank_avg', 'only-momentum'],
    'rating': ['value','growth','mom-value', 'mom-growth', 'growth_value', 'Avg_VGM','no-momentum']
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

ROOT_DIR = "./"
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
)

if DEBUG:
    EXPERIMENT_DIR = ROOT_DIR +"results/"
else:
    if RATING_OR_SCORE == 'rating':
        EXPERIMENT_DIR = ROOT_DIR +"results/" + HEAD_STR + '_'  + RATING_OR_SCORE + '_' + VGM_METHOD + '_' + CONSIDER_RISK_FACTOR + '_' + CONSIDER_DIV + '_' + START_DATE + '_' + N_TOP_TICKERS + '_' + PORTFOLIO_SIZE + '_' + PORTFOLIO_FREQ +'_' + COUNTRY +'/'

    if RATING_OR_SCORE == 'score':
        EXPERIMENT_DIR = ROOT_DIR +"results/" + HEAD_STR + '_' + RATING_OR_SCORE + '_' + VGM_METHOD + '_' + CONSIDER_RISK_FACTOR + '_' + CONSIDER_DIV + '_' + START_DATE + '_' + N_TOP_TICKERS + '_' + PORTFOLIO_SIZE + '_' + PORTFOLIO_FREQ +'_' + COUNTRY +'/'

if SAVE_EXCEL == True:
    RESULTS_DIR = EXPERIMENT_DIR         # + 'results/'
else:
    RESULTS_DIR = EXPERIMENT_DIR

N_TOP_TICKERS = int(N_TOP_TICKERS)
PORTFOLIO_SIZE = int(PORTFOLIO_SIZE)

START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE) 

check_and_make_directories([EXPERIMENT_DIR, RESULTS_DIR])

# Configure logging
log_file = EXPERIMENT_DIR + 'experiment.log'

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


def compute_stock_metrics(df):
    metrics = []

    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Group data by ticker
    grouped = df.groupby('tic')

    for ticker, group in grouped:
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

       # Append the calculated metrics for the current ticker
        metrics.append({
            'Symbol': ticker,
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
        })

    return pd.DataFrame(metrics)

df_metrics = compute_stock_metrics(df_tics_daily)

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
logging.info(f'Stock world: {NAME_STOCK_WORLD}')


# # ================ Create List of Dates   ======================================
 
# if PORTFOLIO_FREQ == 'Monthly':
#     date_list = first_day_month_list(START_DATE, END_DATE)                  # on these dates tickers are updated
#     date_list = [date for date in date_list if date <= df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
#     for idx in np.arange(len(date_list)):
#         while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
#             date_list[idx] = date_list[idx] + timedelta(days = 1)   

# elif PORTFOLIO_FREQ == 'Weekly':
#     date_list = weekday_dates_list(START_DATE, END_DATE, weekday = 4)     # 0: Monday, 4: Friday 
#     date_list = [date for date in date_list if date < df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
#     for idx in np.arange(len(date_list)):
#         while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
#             date_list[idx] = date_list[idx] - timedelta(days = 1)   

# elif PORTFOLIO_FREQ == 'Fortnight':
#     date_list = weekday_dates_list(START_DATE, END_DATE, weekday = 4)     # 0: Monday, 4: Friday 
#     date_list = [date for date in date_list if date < df_tics_daily.sort_values(by='date')['date'].iloc[-1].date()]
#     date_list = date_list[::2]
#     for idx in np.arange(len(date_list)):
#         while pd.to_datetime(date_list[idx]) not in df_tics_daily['date'].values:
#             date_list[idx] = date_list[idx] - timedelta(days = 1)   
# # date_list = third_fridays_dates_list(start_date, end_date)
# # date_list = first_monday_dates_list(START_DATE, END_DATE)

# # =============  Ensure all dates are bussiness day ====================

          

# # append last date if not present in date_list
# if END_DATE.date() not in date_list:        
#     date_list.append(END_DATE.date())           

# if DEBUG:
#     start_point = -2                       # uncomment to evaluate last month only
#     date_list = date_list[start_point:]
#     print(f"List of dates: {[datu.strftime('%Y-%m-%d') for datu in date_list]}")

# if USE_BNH:
    # ================  Buy-hold strategy  ====================================
start_bnh = time.time()
# periodic_tickers_list_bnh = RESULTS_DIR + 'periodic_tickers_list_bnh.txt'




# if os.path.exists(periodic_tickers_list_bnh):
#     os.remove(periodic_tickers_list_bnh)

# # ============  Initialize firstday return to zero  ======================
# firstday_bnh_return = pd.DataFrame(0,columns=['Buy_Hold_returns'], index=[date_list[0]])
# df_return_list = [firstday_bnh_return - TC]

# industry_counts_list = []
# lines = []

# monthly_return = {}
# monthly_return_index = {}
# for idx in range(len(date_list)-1):
    # window_start_date = date_list[idx]
    # window_end_date = date_list[idx + 1]                              # - timedelta(days=1)
    # logging.info(f"Trading period = {window_start_date} - {window_end_date}")
    # print(f"Trading period = {window_start_date} - {window_end_date}")

END_DATE_MONTH = FORMATION_DATE
START_DATE_MONTH = FORMATION_DATE - timedelta(days = 365*MOM_VOLATILITY_PERIOD)

df_marketcap_pivot = df_marketcap.pivot_table(index = 'date', columns = 'symbol',values='marketCap')
temp_mktcap = df_marketcap_pivot[df_marketcap_pivot.index <= pd.to_datetime(FORMATION_DATE)]
temp_mktcap = temp_mktcap.iloc[-1]
df_rank_marketcap = pd.DataFrame(temp_mktcap).reset_index()
df_rank_marketcap.columns = ['tic','marketcap']
df_rank_marketcap = df_rank_marketcap.sort_values(by=['marketcap'],ascending=False)
df_rank_marketcap = df_rank_marketcap.reset_index(drop = True)

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
logging.info(f"Number of tickers after removing tickers with missing values = {len(TICKERS_LIST_WORLD)}")

# value_coeff, growth_coeff = load_factors_coefficients()
# df_new_score = compute_value_growth_score(FORMATION_DATE, value_coeff, growth_coeff)
if COUNTRY == 'IN':
    df_vgm_score = compute_vgm_score_IN(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)
   
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
    df_vgm_score1.to_excel(RESULTS_DIR + 'df_vgm_score' + '_' + FORMATION_DATE.strftime("%Y-%m-%d") + '_IN.xlsx', index = False)


elif COUNTRY == 'US':
    df_vgm_score = compute_vgm_score(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, df_rank_marketcap, RATING_OR_SCORE, VGM_METHOD, CONSIDER_RISK_FACTOR)

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
    


    df_fullname = pd.read_excel(PATH_DATA + '/russell_3000_gurufocus_10-18-2024.xlsx')
    df_vgm_score1 = pd.merge(df_vgm_score, df_fullname,  left_on='tic', right_on='Symbol', how='left')
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


    # df_vgm_score1.to_csv(RESULTS_DIR + 'df_vgm_score' + '_' + FORMATION_DATE.strftime("%Y-%m-%d") + '_US.csv', index = False)
    df_vgm_score1.to_csv(RESULTS_DIR + 'vgm_score_US.csv', index = False)

    
    # df_vgm_score.to_excel(EXPERIMENT_DIR + 'df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)

# df_vgm_score_marketcap = df_vgm_score.merge(df_rank_marketcap, on = 'tic', how = 'left')


# # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
# if SAVE_EXCEL:
#     df_fullname = pd.read_csv(PATH_DATA + '/NSE_Stocks.csv')
#     df_fullname['SYMBOL'] = df_fullname['SYMBOL'] + ".NS"
#     df_vgm_score1 = pd.merge(df_vgm_score, df_fullname,  left_on='tic', right_on='SYMBOL', how='left')
#     df_vgm_score1.to_excel(RESULTS_DIR + 'df_vgm_score' + '_' + FORMATION_DATE.strftime("%Y-%m-%d") + '.xlsx', index = False)
# print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
# logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score1[df_vgm_score1["close"] < 10])}')

# # elif VGM_SCORE_METHOD == 'min-max_avg':
# #     df_vgm_score = compute_vgm_score(df_tics_daily, FORMATION_DATE, TICKERS_LIST_WORLD, rank_based_on = 'both')

# if NAME_STOCK_WORLD == "RUSSELL-3000" and not (VGM_METHOD == 'only-momentum'):
#     # =========== Industrial Count - Diversification ======================
#     top_30 = df_vgm_score.head(N_TOP_TICKERS)
#     industry_counts = top_30['industry'].value_counts().reset_index()
#     industry_counts.columns = ['industry', 'count']
#     industry_counts['date'] = FORMATION_DATE.strftime("%Y-%m-%d")
#     industry_counts_list.append(industry_counts)


# df_vgm_score_top = df_vgm_score.iloc[0:N_TOP_TICKERS] 
# df_vgm_score_marketcap = df_vgm_score_top.merge(df_rank_marketcap, on = 'tic', how = 'left')
# # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
# if SAVE_EXCEL:
#     df_vgm_score_marketcap.to_excel(RESULTS_DIR + 'top_df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
# # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
# logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')

# # =========== Industrial Count - Diversification ======================
# # tickers_list = df_vgm_score_top['tic']   #df_vgm_score['tic'].iloc[0:N_TOP_TICKERS]             # selected tickers based on momentum
# # tickers_list = list(tickers_list)

# # with open(periodic_tickers_list_bnh, "a") as file:
# #     tickers_list.sort()
# #     line = ', '.join([f"({item})" for item in tickers_list])
# #     line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
# #     file.write(line + "\n")

# if CONSIDER_DIV == 'yes':
#     # Function to compute stock returns
#     def StockReturnsComputing(stock_price, rows, cols):            
#         stock_return = np.zeros([rows-1, cols])
#         for j in range(cols):  # j: Assets
#             for i in range(rows-1):  # i: Daily Prices
#                 stock_return[i, j] = (stock_price[i+1, j] - stock_price[i, j]) / stock_price[i, j]
#         return stock_return

#     # Prepare stock price data
#     tickers_list = df_vgm_score_top['tic']
#     df_stock_prices = df_tics_daily[df_tics_daily['tic'].isin(tickers_list)].pivot_table(index='date', columns='tic', values='close')
#     asset_labels = df_stock_prices.columns.tolist()
#     ar_stock_prices = np.asarray(df_stock_prices)
#     rows, cols = ar_stock_prices.shape

#     # Compute daily returns
#     ar_returns = StockReturnsComputing(ar_stock_prices, rows, cols)

#     # Compute mean returns and covariance matrix
#     mean_returns = np.mean(ar_returns, axis=0).reshape(len(ar_returns[0]), 1)
#     cov_returns = np.cov(ar_returns, rowvar=False)

#     # Prepare asset parameters for KMeans clustering
#     asset_parameters = np.concatenate([mean_returns, cov_returns], axis=1)

#     # KMeans clustering
#     clusters = PORTFOLIO_SIZE
#     assets_cluster = KMeans(algorithm='lloyd', max_iter=600, n_clusters=clusters)
#     assets_cluster.fit(asset_parameters)
#     labels = assets_cluster.labels_

#     assets = np.array(asset_labels)

#     # Create a dictionary to store assets in each cluster
#     cluster_dict = {f'Cluster_{i+1}': list(assets[np.where(labels == i)]) for i in range(clusters)}

#     # # Function to create portfolios
#     # def create_portfolios(cluster_dict):
#     #     clusters_tickers = [tickers for tickers in cluster_dict.values()]
#     #     portfolios = list(itertools.product(*clusters_tickers))
#     #     return portfolios

#     # # Generate portfolios and select 20 random portfolios
#     # portfolios = create_portfolios(cluster_dict)
#     # Function to create 20 random portfolios without generating all possible combinations
#     def create_random_portfolios(cluster_dict, num_portfolios=20):
#         clusters_tickers = [tickers for tickers in cluster_dict.values()]
        
#         # Generate 20 random portfolios by selecting one random ticker from each cluster
#         portfolios = [tuple(random.choice(tickers) for tickers in clusters_tickers) for _ in range(num_portfolios)]
        
#         return portfolios

#     # Example usage: generate 20 random portfolios
#     portfolios = create_random_portfolios(cluster_dict, num_portfolios=50)

#     # Create random portfolios dictionary
#     portfolio_dict = {f'Portfolio_{i+1}': list(portfolio) for i, portfolio in enumerate(portfolios)}

#     # Initialize a dictionary to store portfolio scores
#     portfolio_scores = {}

#     # Compute diversification scores for each portfolio and store them in the dictionary
#     for portfolio_num, tickers in portfolio_dict.items():
#         score = diversification_score(df_tics_daily, tickers, method='equal-weight')
#         portfolio_scores[portfolio_num] = score

#     # Select the portfolio with the maximum diversification score
#     max_portfolio = max(portfolio_scores, key=portfolio_scores.get)

#     # Print the portfolio with the highest diversification score
#     print(f'The portfolio with the highest diversification score is {max_portfolio}: {portfolio_dict[max_portfolio]}')
#     print(f'Highest diversification score: {portfolio_scores[max_portfolio]}')

#     tickers_list = portfolio_dict[max_portfolio]

#     df_vgm_score_div = df_vgm_score_top[df_vgm_score_top['tic'].isin(tickers_list)]
#     # df_vgm_score_marketcap = df_vgm_score_marketcap.sort_values(by=['marketcap'],ascending=False)
#     if SAVE_EXCEL:
#         df_vgm_score_div.to_excel(RESULTS_DIR + 'div_df_vgm_score' + '_' + window_start_date.strftime("%Y-%m-%d") + '.xlsx', index = False)
#         # print(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}')
#         # logging.info(f'Number of tickers in Universe less than Rs. 10 = {len(df_vgm_score_marketcap[df_vgm_score_marketcap["close"] < 10])}\n')


# else:
#     tickers_list = df_vgm_score['tic'].iloc[0:PORTFOLIO_SIZE]             # selected tickers based on momentum
#     tickers_list = list(tickers_list)

# # Sort the tickers list and prepare the line
# tickers_list.sort()
# line = ', '.join([f"({item})" for item in tickers_list])
# line = window_start_date.strftime("%Y-%m-%d") + ' ' + line
# lines.append(line)

# LP = Load_n_Preprocess(tickers_list, window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
# df_tics_daily = LP.load_daily_data()
# df_tics_daily = LP.clean_daily_data(df_tics_daily, missing_values_allowed = 0.01)

# df_return = buy_hold_portfolio_return(df_tics_daily)
# try:
#     monthly_return[date_list[idx]] = [round(100*ep.cum_returns_final(df_return),2).values[0]]
# except:
#     print('Today is the inception day')
#     sys.exit(1)  # Exit the program with a non-zero status


# if idx == 0:
#     df_return.iloc[-1] = df_return.iloc[-1] - TC
#     df_return_list.append(df_return)
# else:
#     df_return.iloc[0] = df_return.iloc[0] - TC
#     df_return.iloc[-1] = df_return.iloc[-1] - TC
#     df_return_list.append(df_return)

# LP = Load_n_Preprocess([MAJOR_INDEX], window_start_date, window_end_date, path_daily_data = PATH_DAILY_DATA)
# df_index_daily = LP.load_daily_data()
# df_return_index = buy_hold_portfolio_return(df_index_daily)
# monthly_return_index[date_list[idx]] = [round(100*ep.cum_returns_final(df_return_index),2).values[0]]

# df_return_bnh = pd.concat(df_return_list)

# if (NAME_STOCK_WORLD == "RUSSELL-3000") and not (VGM_METHOD == 'only-momentum'):
#     df_industry_counts = pd.concat(industry_counts_list, ignore_index=True)
#     if SAVE_EXCEL:
#         df_industry_counts.to_excel(RESULTS_DIR + 'df_industry_div.xlsx', index = False)

# df_monthly_return = pd.DataFrame.from_dict(monthly_return, orient='index', columns=['return'])
# if SAVE_EXCEL:
#     df_monthly_return.to_excel(RESULTS_DIR + 'df_monthly_return.xlsx', index = True)

# df_monthly_return_index = pd.DataFrame.from_dict(monthly_return_index, orient='index', columns=['return'])
# if SAVE_EXCEL:
#     df_monthly_return_index.to_excel(RESULTS_DIR + 'df_monthly_return_index.xlsx', index = True)

# # ============================= Prepare data for to plot Buy-Hold returns  ======================
# df_return_bnh = df_return_bnh.reset_index()
# df_return_bnh.columns = ['date','Buy_Hold_Returns']
# df_return_bnh['date'] = pd.to_datetime(df_return_bnh['date'])
# df_return_bnh = df_return_bnh.set_index(['date'])

# # ===============================     Plotting  ====================================
# LP = Load_n_Preprocess(INDICES_LIST, date_list[0], date_list[-1], path_daily_data = PATH_DAILY_DATA)
# df_indices_daily = LP.load_daily_data()
# # df_indices_daily = LP.clean_daily_data(df_indices_daily, missing_values_allowed = 0.01)

# df_indices_close = pd.pivot_table(df_indices_daily, values ='close', index = 'date',columns='tic')
# df_indices_return = df_indices_close.pct_change().fillna(0.001)
# # df_return_bnh['Portfolio_Trend'] = df_return_bnh['Buy_Hold_returns'].rolling(window=TREND_WINDOW).mean()
# df_return_bnh = df_return_bnh.merge(df_indices_return,how='left',on = 'date')
# # df_return_bnh = df_return_bnh.drop(columns=['Portfolio_Trend'])

# if NAME_STOCK_WORLD == "RUSSELL-3000":
#     df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^GSPC': 'S&P 500','IJS': 'S&P 600', '^NDX': 'NASDAQ 100','^SP600': 'S&P 600','^RUT':'RUSSELL 2000','^RUA':'RUSSELL 3000'})
#     # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
#     if SAVE_EXCEL:
#         df_return_bnh.to_excel(RESULTS_DIR + '/' + 'df_return_daily_monthly_ticker_bnh.xlsx')
#     # plot_returns(df_return_bnh, tickers_list = [], filename = RESULTS_DIR + '/' + 'Portfolio_BnH', period = 'daily', name_stock_world = NAME_STOCK_WORLD)

# if NAME_STOCK_WORLD == "NSE-Stocks":
#     if PORTFOLIO_FREQ == 'Monthly':
#         df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
#         df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
#     elif PORTFOLIO_FREQ == 'Weekly':
#         df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
#         df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
#     elif PORTFOLIO_FREQ == 'Fortnight':
#         df_return_bnh = df_return_bnh.rename(columns = {'Buy_Hold_Returns':'Our Portfolio','^NSEI': 'NIFTY 50', '^CRSLDX': 'NIFTY 500', '^BSESN': 'SENSEX'})
#         df_return_bnh = df_return_bnh[['Our Portfolio','NIFTY 50','NIFTY 500','SENSEX']]
    
    
#     # df_return_bnh = df_return_drl.merge(df_return_bnh,how='left',on = 'date')
#     if SAVE_EXCEL:
#         df_return_bnh.to_excel(RESULTS_DIR + '/' + 'df_return_daily_monthly_ticker_bnh.xlsx')
    
# plot_returns_range(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2024-01-01', end_date='2024-12-31')
# plot_returns_range(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2023-01-01', end_date='2023-12-31')
# plot_returns_range(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2022-01-01', end_date='2022-12-31')
# plot_returns_range(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2021-01-01', end_date='2021-12-31')
# plot_returns_range(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD, start_date='2020-01-01', end_date='2020-12-31')

# plot_returns(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Our_Portfolio', period = 'daily', name_stock_world = NAME_STOCK_WORLD)
# plot_returns_drawdown(df_return_bnh, tickers_list = [], filename = EXPERIMENT_DIR + '/' + 'Portfolio_drawdown', period = 'daily', name_stock_world = NAME_STOCK_WORLD, pos = 'lower right')

# hours, minutes, seconds = computation_time(start_time, "Total execution time: ")

# column_names = df_return_bnh.columns.tolist()
# # ===============        Save experiment settings        ===================================
# lines = '\n'.join(lines)
# line = [
#             f"Country                       : {COUNTRY}",
#             f"Filename                      : 20240816_Quant_Portfolio_v3_parametric.py",
#             f"OHLCV data file               : {PATH_DAILY_DATA}",
#             f"Market data file              : {PATH_MARKETCAP_DATA}",
#             f"Stock world                   : {NAME_STOCK_WORLD}",
#             ]

# if COUNTRY == 'IN':
#     line = line + [
#             f"Marketcap threshold           : Rs. {int(MARKETCAP_TH/10000000)} Cr.",]

# elif COUNTRY == 'US':
#     line = line + [
#             f"Marketcap threshold           : ${int(MARKETCAP_TH/1000000)} M",]

# # Ensure all the required variables are defined before using them
# line = line + [
#             f"Rating or Score               : {RATING_OR_SCORE}",
#             f"VGM method                    : {VGM_METHOD}",
#             f"Portfolio update frequency    : {PORTFOLIO_FREQ}",
#             f"No. of tickers in portfolio   : {N_TOP_TICKERS}",
#             f"Portfolio size                : {PORTFOLIO_SIZE}",
#             f"Consider risk factor          : {CONSIDER_RISK_FACTOR}",
#             f"Momentum Volatility period    : {MOM_VOLATILITY_PERIOD}",
#             f"Risk Volatility period        : {RISK_VOLATILITY_PERIOD}",
#             f"Momentum periods              : {MOMENTUM_PERIOD_LIST}",
#             f"Risk periods                  : {RISK_PERIOD_LIST}",
#             f"With transaction cost         : {WITH_TRANS_COST}",
#             f"Which broker                  : {BROKER}",
#             f"Transaction cost              : {TC}",
#             f"Start date                    : {START_DATE}",
#             f"End date                      : {END_DATE}",
#             f"Total execution time          : {str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}",]

# line = line + [            
#             f"CAGR Portfolio                : {round(100*ep.cagr(df_return_bnh[column_names[0]]), 2)} %",]

# if COUNTRY == 'IN':
#     line = line + [
#             f"CAGR NIFTY 50                 : {round(100*ep.cagr(df_return_bnh['NIFTY 50']), 2)} %",
#             f"CAGR NIFTY 500                : {round(100*ep.cagr(df_return_bnh['NIFTY 500']), 2)} %",
#             f"CAGR SENSEX                   : {round(100*ep.cagr(df_return_bnh['SENSEX']), 2)} %",]

# elif COUNTRY == 'US':
#     line = line + [
#             f"CAGR S&P 500                  : {round(100*ep.cagr(df_return_bnh['S&P 500']),2)} %",
#             f"CAGR NASDAQ 100               : {round(100*ep.cagr(df_return_bnh['NASDAQ 100']),2)} %",
#             f"CAGR RUSSELL 3000             : {round(100*ep.cagr(df_return_bnh['RUSSELL 3000']),2)} %",]

# line = line + [
#             f"SHARPE Portfolio              : {round(ep.sharpe_ratio(df_return_bnh[column_names[0]]), 2)}",]

# if COUNTRY == 'IN':
#     line = line + [
#             f"SHARPE NIFTY 50               : {round(ep.sharpe_ratio(df_return_bnh['NIFTY 50']), 2)}",
#             f"SHARPE NIFTY 500              : {round(ep.sharpe_ratio(df_return_bnh['NIFTY 500']), 2)}",
#             f"SHARPE SENSEX                 : {round(ep.sharpe_ratio(df_return_bnh['SENSEX']), 2)}",]

# elif COUNTRY == 'US':
#     line = line + [
#             f"SHARPE S&P 500                : {round(ep.sharpe_ratio(df_return_bnh['S&P 500']),2)}",
#             f"SHARPE NASDAQ 100             : {round(ep.sharpe_ratio(df_return_bnh['NASDAQ 100']),2)}",
#             f"SHARPE RUSSELL 3000           : {round(ep.sharpe_ratio(df_return_bnh['RUSSELL 3000']),2)}",]

# line = line + [
#             f"MAX-DRAWDOWN Portfolio        : {round(-100*ep.max_drawdown(df_return_bnh[column_names[0]]), 2)} %",]

# if COUNTRY == 'IN':
#     line = line + [
#             f"MAX-DRAWDOWN NIFTY 50         : {round(-100*ep.max_drawdown(df_return_bnh['NIFTY 50']), 2)} %",
#             f"MAX-DRAWDOWN NIFTY 500        : {round(-100*ep.max_drawdown(df_return_bnh['NIFTY 500']), 2)} %",
#             f"MAX-DRAWDOWN SENSEX           : {round(-100*ep.max_drawdown(df_return_bnh['SENSEX']), 2)} %"]

# elif COUNTRY == 'US':
#     line = line + [
#             f"MAX-DRAWDOWN S&P 500          : {round(-100*ep.max_drawdown(df_return_bnh['S&P 500']),2)} %",
#             f"MAX-DRAWDOWN NASDAQ 100       : {round(-100*ep.max_drawdown(df_return_bnh['NASDAQ 100']),2)} %",
#             f"MAX-DRAWDOWN RUSSELL 3000     : {round(-100*ep.max_drawdown(df_return_bnh['RUSSELL 3000']),2)} %",]


# # Joining the list to form a multi-line string
# line = '\n'.join(line) + '\n\n' + str(lines)

# Summary_file = EXPERIMENT_DIR + "/Summary_file.txt"
# with open(Summary_file, "w") as file:
#     file.write(line + "\n")

#     # print(line)


     

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