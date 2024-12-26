# directory
from __future__ import annotations
import os
from sys import platform
import pandas as pd
import datetime as dt

# =============== Define Stock World   ========================
NAME_STOCK_WORLD = "RUSSELL-3000"
USE_BNH = True
USE_BNH_MARKETCAP = True
PORTFOLIO_TYPE = 'long_only'          # 'long_only'/ 'short-only' / 'long_short
DOWNLOAD_DATA = False
SAVE_EXCEL = False
SAVE_CSV = False
DEBUG = False

RATING_OR_SCORE = 'rating'               #   'score'   /   'rating'
VGM_SCORE_METHOD = 'z-score_avg'               #   'min-max_avg' / percentile_avg  /  z-score_avg  /  rank_avg  
VGM_RATING_METHOD = 'no-momentum'             # 'value'  /  'growth'  /  'growth-value' / 'Avg-VGM' / 'no-momentum'

# ========= Hyperparameters  ===========================================
N_TOP_TICKERS = 30
MOMENTUM_PERIOD_LIST = [3, 6, 12]
TC = 0.001

INITIAL_AMOUNT = 100000
VOLATILITY_PERIOD = 3                                                          # in years

# ===================== Dates: Start and End =============================
START_DATE = pd.to_datetime('2020-01-01')
END_DATE = pd.to_datetime('2024-07-25')      
# END_DATE = pd.to_datetime('today').normalize()

# ========================  Transaction Cost + Momentum   ================
WITH_TRANS_COST = True
if WITH_TRANS_COST:
    BROKER = 'other'                         # 'other'  or 'ibkr'
else:
    BROKER = 'NA'

# ====================  Directories and Datasets ==========================
HEAD_STR = dt.datetime.now().strftime("%Y%m%d-%H%M")

ROOT_DIR = "./examples/Quant_Rating/"
PATH_DATA = ROOT_DIR + "datasets/"

if RATING_OR_SCORE == 'rating':
    EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + NAME_STOCK_WORLD + '_' + VGM_RATING_METHOD + '/'

if RATING_OR_SCORE == 'score':
    EXPERIMENT_DIR = ROOT_DIR +"experiments/" + HEAD_STR + '_' + NAME_STOCK_WORLD + '_' + VGM_SCORE_METHOD + '/'

RESULTS_DIR = EXPERIMENT_DIR # + "results/"
# RESULTS_DIR = ROOT_DIR + "results/"

PATH_DAILY_DATA = PATH_DATA + "df_ohlcv_daily_russell-3000.h5"
PATH_MARKETCAP_DATA = PATH_DATA + "df_hist_marketcap.h5"

INDICES_LIST = ['^NDX','^GSPC','^RUA']
TICKERS_LIST = pd.read_excel(PATH_DATA + "russell-3000-index-06-20-2024.xlsx")
TICKERS_LIST = TICKERS_LIST['Symbol'].drop_duplicates().to_list()


# FMP_API = '8481ed700f2bf3bb0575f4e9d88f8bbf'
# DOWNLOAD_FUNDAMENTALS = False
# COMPUTE_COEFF = False

# PARENT_DIR = "examples/Quant_Rating/"
# RESULTS_DIR = ROOT_DIR + 'results/'

# INDUSTRY_DATA_DIR = RESULTS_DIR + 'industry_data_rs3000/'
# INDUSTRY_SCORE_DIR = RESULTS_DIR + 'ind_std_score_rs3000/'
INDUSTRY_REGRESSION_DIR = PATH_DATA + 'ind_regression_results/'
# COMPOSIT_SCORE_DIR = RESULTS_DIR + 'composite_score_rs3000/'

# os.makedirs(INDUSTRY_DATA_DIR, exist_ok=True)
# os.makedirs(INDUSTRY_SCORE_DIR, exist_ok=True)
# os.makedirs(COMPOSIT_SCORE_DIR, exist_ok=True)

INDUSTRY_DATA_FILE = PATH_DATA + 'industry_dfs.pkl'

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
    