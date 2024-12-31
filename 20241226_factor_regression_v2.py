import os
import sys
import time

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from numpy.linalg import cond
import argparse

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

from patsy import dmatrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from datetime import timedelta

import seaborn as sns

# ============================   Configure logging   ============================
import logging
log_file = 'logfile.log'

# Check if log file exists, otherwise create one
if not os.path.exists(log_file):
    open(log_file, 'w').close()

# Configure logging
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format without milliseconds

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14                                      # Change 14 to your preferred size


# parser = argparse.ArgumentParser(description='Run factor regression analysis.')
# parser.add_argument('--WHICH_FACTOR', type=str, required=True, help='Specify which factor to use in the analysis.')
# args = parser.parse_args()
# WHICH_FACTOR = args.WHICH_FACTOR

WHICH_FACTOR = 'value'                                             # 'value', 'growth', 'profitability'

DEBUG = False
COUNTRY = 'US'
PATH_DATA = "../FinRecipes/examples/Data_Download/data/Russell3000"
PATH_DAILY_DATA = os.path.join(PATH_DATA, "df_tics_ohlcv_russell3000.h5")
PATH_MARKETCAP_DATA = os.path.join(PATH_DATA, "df_tics_marketcap_russell3000.h5")
PATH_SECTOR_DATA = os.path.join(PATH_DATA, "df_tics_sector_info_russell3000.h5")
PATH_FUNDA_DATA = os.path.join(PATH_DATA, "df_tics_funda_russell3000.h5")

VALUE_METRICES = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio', 'dividendYield','enterpriseValueOverEBITDA']
GROWTH_METRICES = ['revenueGrowth','epsgrowth','freeCashFlowGrowth','returnOnEquity']
TARGET = ['er']

NUM_QUARTERS = 24
OUTLIER_WIDTH = 3

start = time.time()

def computation_time(start, message = "Computational time: "):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{message} {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
    return hours, minutes, seconds

if WHICH_FACTOR == 'value':
    TARGET = ['er']
    METRICS = VALUE_METRICES

elif WHICH_FACTOR == 'growth':
    METRICS = GROWTH_METRICES
    TARGET = ['er']

# elif WHICH_FACTOR == 'profitability':
#     METRICS = ['netProfitMargin','returnOnEquity', 'returnOnAssets']
#     TARGET = ['er']

TIMESTAMP = time.strftime("%Y%m%d_%H%M")
RESULTS_DIR = os.path.join('Reg_Results', TIMESTAMP, WHICH_FACTOR)
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_MULTICOLLINEARITY = True
PERFORM_SKEW_TEST = False
PERFORM_REGRESSION = True
FILTER_OUTLIERS_METRICS = True
FILTER_OUTLIERS_TARGET = True
SAVE_PLOTS = False
SAVE_EXCEL = False
USE_PERMUTATION_IMPORTANCE = True
COLLINEAR_IND_LIST = []
IMPUTATION_METHOD = 'knn'
VIF_LIST = []

global COUNT_MULCOL, perm_importance_value_coeffs, rf_value_coeffs, rf_value_coeffs_df
rf_value_coeffs_df  = pd.DataFrame()
perm_importance_value_coeffs_df = pd.DataFrame()
COUNT_MULCOL = 0

FORMATION_DATE = pd.to_datetime("today").normalize()

REQ_COLUMNS = METRICS + TARGET

def filter_outliers(df: pd.DataFrame, cols: list, method: str = 'zscore') -> pd.DataFrame:
    """
    Filter or clip outliers from specified columns using various statistical methods.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        cols (list): List of column names to process for outliers.
        method (str): Method to handle outliers. Options: 'zscore', 'IQR', 'clip'.
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled according to specified method.
    
    Raises:
        ValueError: If invalid method is specified or input validation fails.
    """
    try:
        logging.info(f"Starting outlier filtering using {method} method")
        
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame")
        if not all(col in df.columns for col in cols):
            raise ValueError("Some specified columns not found in DataFrame")
        if method not in ['zscore', 'IQR', 'clip']:
            raise ValueError("Method must be one of: 'zscore', 'IQR', 'clip'")
            
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        initial_rows = len(df_processed)
        
        if method == 'zscore':
            zscores = np.abs(df_processed[cols].apply(lambda x: (x - x.mean()) / x.std()))
            df_processed = df_processed[(zscores < 3).all(axis=1)]
            
        elif method == 'IQR':
            Q1 = df_processed[cols].quantile(0.25)
            Q3 = df_processed[cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df_processed[cols] < (Q1 - OUTLIER_WIDTH * IQR)) | 
                    (df_processed[cols] > (Q3 + OUTLIER_WIDTH * IQR))).any(axis=1)
            df_processed = df_processed[mask]
            
        elif method == 'clip':
            Q1 = df_processed[cols].quantile(0.25)
            Q3 = df_processed[cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - OUTLIER_WIDTH * IQR
            upper_cap = Q3 + OUTLIER_WIDTH * IQR
            df_processed[cols] = df_processed[cols].clip(lower=lower_cap, upper=upper_cap, axis=1)
        
        rows_removed = initial_rows - len(df_processed)
        logging.info(f"Outlier filtering complete. Rows affected: {rows_removed}")
        logging.info(f"Remaining data points: {len(df_processed)}")
        
        return df_processed
        
    except Exception as e:
        logging.error(f"Error in outlier filtering: {str(e)}")
        raise

# Function to filter the most recent quarters
def filter_recent_quarters(df):
    try:
        logging.info("Starting to filter recent {NUM_QUARTERS} quarters.")
        
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        logging.info("Converted 'date' column to datetime format.")
        
        # Sort by 'tic' and 'date'
        df = df.sort_values(by=['tic', 'date'], ascending=[True, False])
        logging.info("Sorted DataFrame by 'tic' and 'date'.")
        
        # Filter out dates beyond the formation date
        df = df[df['date'] < pd.to_datetime(FORMATION_DATE)]
        logging.info(f"Filtered DataFrame to include dates before {FORMATION_DATE}.")
        
        # Rank quarters for each ticker
        df['quarter_rank'] = df.groupby('tic')['date'].rank(method='first', ascending=False)
        logging.info("Ranked quarters for each ticker.")
        
        # Filter to include only the most recent quarters
        df_filtered = df[df['quarter_rank'] <= NUM_QUARTERS].drop(columns=['quarter_rank'])
        df_filtered = df_filtered.reset_index(drop=True)
        logging.info(f"Filtered DataFrame to include the most recent {NUM_QUARTERS} quarters.")
        
        logging.info("Finished filtering recent quarters.")
        return df_filtered
    
    except Exception as e:
        logging.error(f"Error occurred while filtering recent quarters: {e}")
        raise

def replace_outliers_with_median(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = column.median()
    
    # Replace values outside the bounds with the median
    return column.apply(lambda x: median if x < lower_bound or x > upper_bound else x)

# Function to perform regression analysis and evaluate Random Forest
def perform_regression(df, industry):

    global rf_value_coeffs_df, perm_importance_value_coeffs_df, COUNT_MULCOL, COLLINEAR_IND_LIST, VIF_LIST
    industry_df = filter_recent_quarters(df)       # filter last 24 quarters data used for regression
    # industry_df = industry_df[REQ_COLUMNS]

    nrows_industry_df = len(industry_df)            # number of data points availabe for regression

    industry_df = industry_df.dropna(subset = TARGET)     # First Target variable should not be NaN
    industry_df = industry_df.dropna(subset=METRICS, how='all')    # Then drop rows where all the METRICS are NaN

    logging.info(f"{industry}: Number of NaN values before imputation = {industry_df.isna().sum().sum()}")
    if industry_df.isna().sum().sum() > 0:
        if IMPUTATION_METHOD == 'median':
            industry_df[METRICS] = industry_df[METRICS].apply(lambda x: x.fillna(x.median()), axis=0)
            logging.info(f"{industry}: Imputed missing values using median.")
        
        elif IMPUTATION_METHOD == 'knn':
            knn_imputer = KNNImputer(n_neighbors=5)
            industry_df[METRICS] = pd.DataFrame(knn_imputer.fit_transform(industry_df[METRICS]), columns=METRICS)
            logging.info(f"{industry}: Imputed missing values using KNN imputer.")
        
        logging.info(f"{industry}: Shape after imputing rows: {industry_df.shape}")
    else:
        logging.info(f"{industry}: No missing values found, skipping imputation.")

    # Initialize vif_data with basic information
    vif_data = pd.DataFrame({
        "Feature": METRICS,
        "Industry": industry,
        "VIF": [0] * len(METRICS),
        "vif_mulcol": ["No"] * len(METRICS),
        "Condition_No": 0,
        "No_obs_with_nan": nrows_industry_df,
        "No_obs": len(industry_df)
    })

    if TEST_MULTICOLLINEARITY:
        formula = "0 + " + " + ".join(METRICS)              # '0 +' excludes intercept in dmatrix
        X = dmatrix(formula, industry_df)       # Create the design matrix
        
        # Calculate VIF (Variance Inflation Factor) for each feature
        vif_data = pd.DataFrame({
            "Feature": METRICS,
            "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        })

        correlated_features = vif_data[vif_data["VIF"] > 5]
        condition_number = cond(industry_df[METRICS].corr())
        
        vif_data['Industry'] = industry
        vif_data['Condition_No'] = condition_number
        vif_data['No_obs_with_nan'] = nrows_industry_df
        vif_data['No_obs'] = len(industry_df)

            # print(f"Correlated features are: {correlated_features}")

            #vif_data = vif_data[['Industry', 'Feature', 'VIF', 'Condition_No', 'No_obs_with_nan', 'No_obs']]
            # VIF_LIST.append(vif_data)

        # add a column in vif_data named "vif_mulcol", value will be "No" if vif is less than 5 else "Yes"
        vif_data['vif_mulcol'] = np.where(vif_data['VIF'] > 5, 'Yes', 'No')

    if industry_df.shape[0] <= industry_df[METRICS].shape[1]:
        print(f"Not enough data to perform regression for industry: {industry}")
        return

    if FILTER_OUTLIERS_METRICS:
        industry_df = filter_outliers(industry_df, METRICS, method = 'IQR')
    
    if FILTER_OUTLIERS_TARGET:
        industry_df = filter_outliers(industry_df, TARGET, method = 'clip')

    # Begin Skewness test ====================================================================
    if PERFORM_SKEW_TEST:
        skewness = industry_df[METRICS].skew()    # skewness test: 

        '''Interpretation:
            Skewness ≈ 0: Data is roughly symmetric (normal distribution).
            Skewness > 0: Positive skew (long tail on the right).
            Skewness < 0: Negative skew (long tail on the left).
        Threshold:
            Values outside the range (-0.5, 0.5) are often considered skewed.
            Highly skewed: |Skewness| > 1.0.'''
        if SAVE_PLOTS:
            # 1. Save the histogram plot
            hist_filename = os.path.join(RESULTS_DIR, f"{industry}_histogram.png")
            hist = industry_df[METRICS].hist(figsize=(12, 10), bins=50)
            plt.title(f"Histogram for {industry}")  # Add the industry name as the title
            plt.tight_layout()
            plt.savefig(hist_filename)
            plt.close()

            # 2. Save the box plot
            boxplot_filename = os.path.join(RESULTS_DIR, f"{industry}_boxplot.png")
            box = industry_df[METRICS].boxplot(figsize=(12, 8))
            plt.title(f"Box plot for {industry}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(boxplot_filename)
            plt.close()

            # Create a heatmap for the correlation matrix
            corr_filename = os.path.join(RESULTS_DIR, f"{industry}_corr.png")
            plt.figure(figsize=(10, 8))
            correlation_matrix = industry_df[METRICS].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f"Correlation heatmap for {industry}")
            plt.tight_layout()
            plt.savefig(corr_filename)
            plt.close()

            # # Create pairplot for scatter plots of all features in the DataFrame
            # scatter_plot_filename = os.path.join(RESULTS_DIR, f"{industry}_scatter_plot.png")
            # pairplot = sns.pairplot(industry_df)
            # plt.suptitle(f"Scatter Plot for {industry}", y=1.02)  # Adjust y for spacing
            # # plt.tight_layout()
            # plt.savefig(scatter_plot_filename)
            # plt.close()

    # End Skewness test ====================================================================


    # scaler = RobustScaler()
    # X = industry_df.drop('er', axis=1)
    # X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # # Transform features to reduce skewness
    # pt = PowerTransformer(method='yeo-johnson')
    # try:
    #     X_transformed = pd.DataFrame(pt.fit_transform(X_scaled), columns=X.columns)
    # except Exception as e:
    #     print(f"PowerTransformer failed for {industry}: {e}")
    #     X_transformed = X_scaled

    # y = industry_df_imputed['er']

    if PERFORM_REGRESSION:
        
        rf = RandomForestRegressor(random_state=42)
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [10, 20, 30, None]
        # }

        param_grid = {
                    'n_estimators': [50,100, 150],
                    # 'n_estimators': [50,100, 150,200],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],  # Add minimum samples split
                    'min_samples_leaf':  [1, 2, 5]
                }

        # Value Metrics
        X_value_transformed = industry_df[METRICS]
        y = industry_df[TARGET].values.ravel()  # Flatten the target array

        rf_grid_value = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
        rf_grid_value.fit(X_value_transformed, y)

        rf_best_value = rf_grid_value.best_estimator_
        # print(f"Random Forest Best Params for {industry} (value metrics):", rf_grid_value.best_params_)
        rf_value_feature_importances = rf_best_value.feature_importances_
        # print(f"Random Forest Feature Importances for {industry} (value metrics):", rf_value_feature_importances)
        rf_value_coeffs = pd.Series(rf_value_feature_importances, index=X_value_transformed.columns)
        rf_value_coeffs.name = industry
        rf_value_coeffs_df = pd.concat([rf_value_coeffs_df, rf_value_coeffs], axis=1)

        ## =============================== Evaluation =========================================================
        # # 1. Cross-validation (MSE, MAE, R²)
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        
        mse_scores = cross_val_score(rf_best_value, X_value_transformed, y, cv=cv, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(rf_best_value, X_value_transformed, y, cv=cv, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(rf_best_value, X_value_transformed, y, cv=cv, scoring='r2')

        vif_data['mse_scores'] = -mse_scores.mean()
        vif_data['mae_scores'] = -mae_scores.mean()
        vif_data['r2_scores'] = r2_scores.mean()

        # add a column in vif_data named "reg_model_eval", the value will be "good" if "r2_scores" is greater than 0.5 else "bad".
        vif_data['reg_model_eval'] = np.where(vif_data['r2_scores'] > 0.5, 'good', 'bad')
        # print(f"Correlated features are: {correlated_features}")

        vif_data = vif_data[['Industry', 'Feature', 'VIF', 'vif_mulcol', 'Condition_No', 'No_obs_with_nan', 'No_obs','mse_scores','mae_scores','r2_scores','reg_model_eval']]
        VIF_LIST.append(vif_data)
        

        if USE_PERMUTATION_IMPORTANCE:
            # 2. Permutation Importance
            perm_importance = permutation_importance(rf_best_value, X_value_transformed, y, n_repeats=100, random_state=42)
            perm_importance_value = perm_importance.importances_mean
            perm_importance_value = perm_importance_value / perm_importance_value.sum()
            
            perm_importance_value_coeffs = pd.Series(perm_importance_value, index=X_value_transformed.columns)
            perm_importance_value_coeffs.name = industry
            perm_importance_value_coeffs_df = pd.concat([perm_importance_value_coeffs_df, perm_importance_value_coeffs], axis=1)
            # print("Permutation Importance (Value Metrics):")
            # for feature, imp in zip(X_value_transformed.columns, perm_importance_value.importances_mean):
            #     print(f"{feature}: {imp:.4f}")

        if SAVE_PLOTS:
            # Residual Analysis
            residual_plot_filename = os.path.join(RESULTS_DIR, f"{industry}_residual_plot.png")
            y_pred_value = rf_best_value.predict(X_value_transformed)
            residuals_value = y - y_pred_value
            plt.figure(figsize=(6, 6))
            plt.scatter(y_pred_value, residuals_value, alpha=0.5)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual plot for {industry}')
            plt.tight_layout()
            plt.savefig(residual_plot_filename)
            plt.close()

        # # Plot for Growth Metrics
        # y_pred_growth = rf_best_growth.predict(X_growth_transformed)
        # residuals_growth = y - y_pred_growth
        
        # plt.subplot(1, 2, 2)
        # plt.scatter(y_pred_growth, residuals_growth, alpha=0.5)
        # plt.axhline(0, color='red', linestyle='--')
        # plt.xlabel('Predicted Values')
        # plt.ylabel('Residuals')
        # plt.title(f'Residual Plot for {industry} (growth metrics)')

        # plt.tight_layout()
        # plt.show()

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

        # df_funda = compute_profitability_metrics(df_funda)
        
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
        
        n_years_ago = FORMATION_DATE - pd.DateOffset(years=8)
        date_mask = (df_funda['date'].dt.date.between(n_years_ago.date(), FORMATION_DATE.date()))
        df_funda = df_funda.loc[date_mask]
        logging.info("Filtered fundamental data for the last 8 years.")

        # filter df_tics_daily for the last 8 years
        df_tics_daily['date'] = pd.to_datetime(df_tics_daily['date'])
        df_tics_daily = df_tics_daily[df_tics_daily['date'].dt.date.between(n_years_ago.date(), FORMATION_DATE.date())]
        logging.info("Filtered daily data for the last 8 years.")

        df_marketcap['date'] = pd.to_datetime(df_marketcap['date'])
        logging.info("Converted 'date' column in market cap data to datetime format.")
        
        return df_tics_daily, df_marketcap, df_sector, df_funda
    
    except Exception as e:
        logging.error(f"Error occurred while loading raw data: {e}")
        raise

def main():
    # Load the industry DataFrames
    df_tics_daily, df_marketcap, df_sector, df_funda = load_raw_data()
    industry_dfs = df_funda.copy()
    
    required_columns = ['tic', 'date','Industry'] + REQ_COLUMNS             # required columns
    industry_dfs  = industry_dfs[required_columns]                          # select required columns
    industry_dfs  = industry_dfs.dropna()                                   # drop rows where no quarterly data is available

    industry_name_list = industry_dfs['Industry'].unique().tolist()
    industry_name_list = [name for name in industry_name_list if name and name.strip()]    # new list containing only the non-empty, non-whitespace-only strings
    industry_name_list.sort()

    if DEBUG:
        industry_name_list = industry_name_list[:4]

    for industry in industry_name_list:
        df = industry_dfs[industry_dfs['Industry'] == industry]
        df = df.sort_values(by=['tic','date'], ascending=[True, True]).reset_index(drop = True)
                
        if (len(df['tic'].unique()) >= 7):
            print(f"Industry: {industry}, Number of Tikcers: {len(df['tic'].unique())}, Data Shape: {df.shape}")
            perform_regression(df, industry)
        else:
            print(f"Skipping industry: {industry} due to insufficient data.")

    df_multicollinearity = pd.concat(VIF_LIST, ignore_index=True)
    vif_filename = os.path.join(RESULTS_DIR, f"{WHICH_FACTOR}_Industry_Multicollinearity.xlsx")
    df_multicollinearity.to_excel(vif_filename, index = False)

    # Count and get lists of industries with multicollinearity and bad regression models
    mulcol_industries = df_multicollinearity[df_multicollinearity['vif_mulcol'] == 'Yes']['Industry'].unique()
    bad_reg_industries = df_multicollinearity[df_multicollinearity['reg_model_eval'] == 'bad']['Industry'].unique()
    
    COUNT_MULCOL = len(mulcol_industries)
    COUNT_BAD_REG_MODEL = len(bad_reg_industries)

    # Write statistics and lists to text file
    stats_filename = os.path.join(RESULTS_DIR, f"{WHICH_FACTOR}_regression_stats.txt")
    with open(stats_filename, 'w') as f:
        f.write(f"Total number of industries: {len(industry_name_list)}\n")
        f.write(f"Number of non-collinear industries: {COUNT_MULCOL}\n") 
        f.write(f"Number of industries with bad regression model: {COUNT_BAD_REG_MODEL}\n\n")
        
        f.write("Industries with multicollinearity:\n")
        for ind in sorted(mulcol_industries):
            f.write(f"- {ind}\n")
            
        f.write("\nIndustries with bad regression models:\n")
        for ind in sorted(bad_reg_industries):
            f.write(f"- {ind}\n")

    # # Write statistics to a text file
    # stats_filename = os.path.join(RESULTS_DIR, f"{WHICH_FACTOR}_regression_stats.txt")
    # with open(stats_filename, 'w') as f:
    #     f.write(f"Total number of industries: {len(industry_name_list)}\n")
    #     f.write(f"Number of non-collinear industries: {COUNT_MULCOL}\n") 
    #     f.write(f"Number of industries with bad regression model: {COUNT_BAD_REG_MODEL}\n")
    
    # Also print to console
    print(f"Total no. of industry: {len(industry_name_list)}, \
                \n Number of non-collinear industry: {COUNT_MULCOL} \
                \n Number of industry with bad regression model: {COUNT_BAD_REG_MODEL}")

    coeff_filename = os.path.join(RESULTS_DIR, f"rf_{WHICH_FACTOR}_coeffs_df.xlsx")
    rf_value_coeffs_df.transpose().to_excel(coeff_filename)

    # make a copy of the rf_value_coeffs_df to eq_rf_value_coeffs_df and assign equal weight in each column for each industry
    eq_rf_value_coeffs_df = rf_value_coeffs_df.copy()
    num_feature = len(eq_rf_value_coeffs_df.index)

    # Assign 1/number of columns to each cell in the dataframe 
    eq_rf_value_coeffs_df = eq_rf_value_coeffs_df.map(lambda x: 1/num_feature)

    # save the transposed eq_rf_value_coeffs_df to excel
    coeff_filename_eq = os.path.join(RESULTS_DIR, f"eq_rf_{WHICH_FACTOR}_coeffs_df.xlsx")
    eq_rf_value_coeffs_df.transpose().to_excel(coeff_filename_eq)

    coeff_filename_perm = os.path.join(RESULTS_DIR, f"perm_importance_{WHICH_FACTOR}_coeffs_df.xlsx")
    perm_importance_value_coeffs_df.transpose().to_excel(coeff_filename_perm)
    

    computation_time(start)

if __name__ == "__main__":
    main()

