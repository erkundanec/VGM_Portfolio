# directory
from __future__ import annotations
import os
from sys import platform
import pandas as pd
import datetime as dt

# =============== Define Country   ========================
COUNTRY = 'US'     # 'US'

if COUNTRY == 'IN':
    NAME_STOCK_WORLD = "NSE-Stocks" 

elif COUNTRY == 'US':
    NAME_STOCK_WORLD = "RUSSELL-3000"

DEBUG = False
SAVE_EXCEL = True
SAVE_INDUSTRIAL_MOMENTUM_SCORE = False

USE_BNH = True
USE_BNH_MARKETCAP = True
PORTFOLIO_TYPE = 'long_only'          # 'long_only'/ 'short-only' / 'long_short
PORTFOLIO_FREQ = 'Monthly'                    # Weekly, Fortnight, Monthly
RATING_OR_SCORE = 'score'               #   'score'   /   'rating'
VGM_SCORE_METHOD = 'min-max_avg'               #   'min-max_avg' / percentile_avg  /  z-score_avg  /  rank_avg  
VGM_RATING_METHOD = 'no-momentum'             # 'mom-value'  /  'mom-growth'  /  'growth-value' / 'Avg-VGM' / 'no-momentum'

# ========= Hyperparameters  ===========================================
N_TOP_TICKERS = 20

TC = 0.001
CONSIDER_RISK_FACTOR = 'yes'

INITIAL_AMOUNT = 100000
MOM_VOLATILITY_PERIOD = 3   # 3 year  
RISK_VOLATILITY_PERIOD = 3  # 3 year                                                    # in years

# ===================== Dates: Start and End =============================
START_DATE = pd.to_datetime('2020-01-01')
# END_DATE = pd.to_datetime('2024-08-31')      
END_DATE = pd.to_datetime('today').normalize()

# ========================  Transaction Cost + Momentum   ================
WITH_TRANS_COST = True
if WITH_TRANS_COST:
    BROKER = 'other'                         # 'other'  or 'ibkr'
else:
    BROKER = 'NA'

# ====================  Directories and Datasets ==========================
HEAD_STR = dt.datetime.now().strftime("%Y%m%d-%H%M")

ROOT_DIR = "./"
PATH_DATA = ROOT_DIR + "datasets/"

if RATING_OR_SCORE == 'rating':
    EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + NAME_STOCK_WORLD + '_' + VGM_RATING_METHOD + '/'

if RATING_OR_SCORE == 'score':
    EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + NAME_STOCK_WORLD + '_' + VGM_SCORE_METHOD + '/'

RESULTS_DIR = EXPERIMENT_DIR # + "results/"
# RESULTS_DIR = ROOT_DIR + "results/"

if NAME_STOCK_WORLD == "RUSSELL-3000":
    LARGE_CAP_FILTER = False      
    MID_CAP_FILTER = True
    SMALL_CAP_FILTER =  False

    SCALING_METHOD = 'z-score'                          # z-score
    MOMENTUM_PERIOD_LIST =  [3, 6, 12]                       # [3, 12]           [3, 6, 12]
    RISK_PERIOD_LIST = [1]
    DOWNLOAD_OHLCV_DATA = False                 # last downloaded on 22/10/2024
    DOWNLOAD_HIST_MARKETCAP = False             # last downloaded on 22/10/2024
    LIVE_DATA = True 
    N_YEARS_HIST_OHLCV = 8
    N_YEARS_HIST_MARKETCAP = 5
    MARKETCAP_TH = 300000000          # 10000000000                                    


    # INDUSTRY_DATA_FILE = PATH_DATA + 'industry_dfs_US_26-07-2024.pkl'
    # INDUSTRY_DATA_FILE = PATH_DATA + 'industry_dfs_US_30-09-2024.pkl'
    INDUSTRY_DATA_FILE = PATH_DATA + 'industry_dfs_US_22-10-2024.pkl'
    
    # FMP_API = '8481ed700f2bf3bb0575f4e9d88f8bbf'
    # DOWNLOAD_FUNDAMENTALS = False
    # COMPUTE_COEFF = False

    # PARENT_DIR = "examples/Quant_Rating/"
    # RESULTS_DIR = ROOT_DIR + 'results/'

    # INDUSTRY_DATA_DIR = RESULTS_DIR + 'industry_data_rs3000/'
    # INDUSTRY_SCORE_DIR = RESULTS_DIR + 'ind_std_score_rs3000/'
    INDUSTRY_REGRESSION_DIR = PATH_DATA + 'ind_regression_results_US/'
    # COMPOSIT_SCORE_DIR = RESULTS_DIR + 'composite_score_rs3000/'

    # os.makedirs(INDUSTRY_DATA_DIR, exist_ok=True)
    # os.makedirs(INDUSTRY_SCORE_DIR, exist_ok=True)
    # os.makedirs(COMPOSIT_SCORE_DIR, exist_ok=True)

    MAJOR_INDEX = '^GSPC'
    PATH_DAILY_DATA = PATH_DATA + "df_ohlcv_daily_US_22-10-2024.h5"
    PATH_MARKETCAP_DATA = PATH_DATA + "df_hist_marketcap_US_22-10-2024.h5"

    INDICES_LIST = ['^NDX','^GSPC','^RUA']
    TICKERS_LIST = pd.read_excel(PATH_DATA + "russell_3000_gurufocus_10-18-2024.xlsx")
    TICKERS_LIST = TICKERS_LIST['Symbol'].drop_duplicates().to_list()
    # TICKERS_LIST = TICKERS_LIST[:100]

    LOAD_FUNDAMENTALS_FROM_EXCEL = True
    NUM_QUARTERS = 24  # Change this value 

    VALUE_METRICES = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio',
                    'dividendYield','priceToFreeCashFlowsRatio','enterpriseValueOverEBITDA','freeCashFlowYield']

    GROWTH_METRICES = ['revenueGrowth','epsgrowth','operatingIncomeGrowth','freeCashFlowGrowth','assetGrowth',
                    'returnOnAssets','returnOnEquity','returnOnCapitalEmployed','operatingProfitMargin','netProfitMargin']

    CAT1_RATIOS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio','priceToFreeCashFlowsRatio',
                    'enterpriseValueOverEBITDA','cashConversionCycle','debtToEquity','debtToAssets','netDebtToEBITDA']

    CAT2_RATIOS = ['dividendYield','freeCashFlowYield','revenueGrowth','epsgrowth',
                    'operatingIncomeGrowth','freeCashFlowGrowth','assetGrowth','netProfitMargin','returnOnAssets','returnOnEquity',
                    'returnOnCapitalEmployed','operatingProfitMargin','assetTurnover','inventoryTurnover','receivablesTurnover',
                    'payablesTurnover','currentRatio','quickRatio','cashRatio','workingCapital','interestCoverage',]
        

elif NAME_STOCK_WORLD == "NSE-Stocks":
    LARGE_CAP_FILTER = False
    MID_CAP_FILTER = False
    SMALL_CAP_FILTER =  False


    SCALING_METHOD = 'min-max'                         # z-score

    MOMENTUM_PERIOD_LIST = [3, 6, 12]
    RISK_PERIOD_LIST = [1, 3]
    DOWNLOAD_OHLCV_DATA = False                  # last downloaded on 24/09/2024    
    DOWNLOAD_HIST_MARKETCAP = False             # last downloaded on 24/09/2024
    LIVE_DATA = False
    N_YEARS_HIST_OHLCV = 8
    N_YEARS_HIST_MARKETCAP = 5

    INDUSTRY_DATA_FILE = PATH_DATA + 'industry_dfs_IN_22-10-2024.pkl'
    INDUSTRY_REGRESSION_DIR = PATH_DATA + 'ind_regression_results_IN/'

    TH_FORECAST = 6
    MARKETCAP_TH =  5000000000                        # 50000000000
    MAJOR_INDEX = '^NSEI'
    PATH_DAILY_DATA = PATH_DATA + "df_ohlcv_daily_IN_22-10-2024.h5"
    PATH_MARKETCAP_DATA = PATH_DATA + "df_hist_marketcap_IN_22-10-2024.h5"

    INDICES_LIST = ['^NSEI', '^CRSLDX', '^BSESN']
    TICKERS_LIST = pd.read_csv(PATH_DATA + "NSE_Stocks.csv")
    TICKERS_LIST = TICKERS_LIST['SYMBOL'].drop_duplicates().to_list()
    TICKERS_LIST = [symbol.strip() + ".NS" for symbol in TICKERS_LIST]

    LOAD_FUNDAMENTALS_FROM_EXCEL = True
    num_quarters = 40  # Adjust this value as needed

    #==================================================***=============================================

    VALUE_METRICES = ['priceToBookRatio', 'priceToSalesRatio', 'priceEarningsRatio', 
                    'priceToFreeCashFlowsRatio', 'enterpriseValueOverEBITDA', 'freeCashFlowYield']
    GROWTH_METRICES = ['revenueGrowth', 'revenue 3-Period Avg Growth', 'epsgrowth', 
                    'eps 3-Period Avg Growth', 'operatingIncomeGrowth', 
                    'freeCashFlowGrowth', 'incomeBeforeTax 3-Period Avg Growth']

    #=================================================***==============================================
    CAT1_RATIOS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio','priceToFreeCashFlowsRatio',
                'enterpriseValueOverEBITDA']
    CAT2_RATIOS = ['freeCashFlowYield','revenueGrowth','revenue 3-Period Avg Growth',
                'epsgrowth','eps 3-Period Avg Growth','operatingIncomeGrowth','freeCashFlowGrowth',
                'incomeBeforeTax 3-Period Avg Growth']






