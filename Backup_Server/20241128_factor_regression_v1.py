import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from numpy.linalg import cond
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14  # Change 14 to your preferred size
import time

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

from patsy import dmatrix

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ==========  Parse Command-Line Arguments  ==============
import argparse  # New import for argument parsing
parser = argparse.ArgumentParser(description='Evaluate VGM Score using Quant Portfolio')
parser.add_argument('--WHICH_FACTOR', type=str, required=True, help="'score' or 'rating'")

args = parser.parse_args()
WHICH_FACTOR = args.WHICH_FACTOR

start = time.time()

def computation_time(start, message = "Computational time: "):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{message} {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
    return hours, minutes, seconds

if WHICH_FACTOR == 'value':
    # METRICS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio',
    #               'dividendYield','priceToFreeCashFlowsRatio','enterpriseValueOverEBITDA']  #,'freeCashFlowYield']

    METRICS = ['priceToBookRatio','priceToSalesRatio','priceEarningsRatio', 'dividendYield','enterpriseValueOverEBITDA']
    TARGET = ['er']

elif WHICH_FACTOR == 'growth':
    METRICS = ['revenueGrowth','epsgrowth','freeCashFlowGrowth','returnOnEquity']

    TARGET = ['er']

# elif WHICH_FACTOR == 'profitability':
#     METRICS = ['netProfitMargin','returnOnEquity', 'returnOnAssets']

#     TARGET = ['er']

RESULTS_DIR = 'examples/Quant_Rating/debug/' + WHICH_FACTOR
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_MULTICOLLINEARITY = True
USE_CONDITION_NUMBER  = False
USE_VIF = True

PERFORM_SKEW_TEST = True
PERFORM_REGRESSION = True
FILTER_OUTLIERS_METRICS = True
FILTER_OUTLIERS_TARGET = True
SAVE_PLOTS = True
USE_PERMUTATION_IMPORTANCE = True
COLLINEAR_IND_LIST = []
IMPUTATION_METHOD = 'knn'
VIF_LIST = []

global COUNT, perm_importance_value_coeffs, rf_value_coeffs
COUNT = 0

FORMATION_DATE = pd.to_datetime("today").date()
REQ_COLUMNS = METRICS + TARGET

def filter_outliers(df,cols, method = 'zscore'):
    df = df.copy()

    if method == 'zscore':
        # # Calculate Z-scores for the selected financial ratios
        # z_scores = df.apply(zscore)

        # # Filter out data points with absolute Z-score greater than a threshold (e.g., 3)
        # df_no_outliers = df[(z_scores < 3).all(axis=1)]

        zscores = np.abs(df[cols].apply(lambda x: (x - x.mean()) / x.std()))
        df_no_outliers = df[(zscores < 3).all(axis=1)]
        return df_no_outliers

    elif method == 'IQR':
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        # Filter out outliers
        df_no_outliers = df[~((df[cols] < (Q1 - 2 * IQR)) | (df[cols] > (Q3 + 2 * IQR))).any(axis=1)]
        return df_no_outliers

    elif method == 'clip':
        # Cap values to the 95th percentile
        # upper_cap = industry_df[cols].quantile(0.95)
        # lower_cap = industry_df[cols].quantile(0.05)

        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds for outliers
        lower_cap = Q1 - 2 * IQR
        upper_cap = Q3 + 2 * IQR

        df[cols] = df[cols].apply(lambda x: x.clip(lower=lower_cap.values[0], upper=upper_cap.values[0]))
        return df

    else:
        print('Pass correct method to handle outliers')

# Function to filter the most recent quarters
def filter_recent_quarters(df, num_quarters):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])
    df = df[df['date'] < pd.to_datetime(FORMATION_DATE)]
    df['quarter_rank'] = df.groupby('symbol')['date'].rank(method='first', ascending=False)
    df_filtered = df[df['quarter_rank'] <= num_quarters].drop(columns=['quarter_rank'])
    return df_filtered

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

    global rf_value_coeffs_df, perm_importance_value_coeffs_df, COUNT, COLLINEAR_IND_LIST, VIF_LIST
    industry_df = filter_recent_quarters(df, num_quarters)
    industry_df = industry_df[REQ_COLUMNS]

    nrows_industry_df = len(industry_df)

    industry_df = industry_df.dropna(subset = TARGET)
    industry_df = industry_df.dropna(subset = METRICS)

    # print(f"{industry}: Number of NaN values = {industry_df.isna().sum().sum()}")
    if industry_df.isna().sum().sum() > 0:
        if IMPUTATION_METHOD == 'median':
            industry_df = industry_df.fillna(industry_df.median())
        
        elif IMPUTATION_METHOD == 'knn':
            knn_imputer = KNNImputer(n_neighbors=5)
            industry_df = pd.DataFrame(knn_imputer.fit_transform(industry_df[METRICS]), columns=METRICS)
            print(f"Shape after imputing rows: {industry_df.shape}")

    if TEST_MULTICOLLINEARITY:
        if USE_CONDITION_NUMBER:
            # Calculate the condition number
            condition_number = cond(industry_df.corr())
            if condition_number < 30:
                print(f"{industry}: No serious muliticollinearity")
                
            else:
                print(f"{industry}: Serious muliticollinearity")
                COUNT = COUNT + 1
                COLLINEAR_IND_LIST.append(industry)
                # return

        if USE_VIF:
            formula = "0 + " + " + ".join(METRICS)              # '0 +' excludes intercept in dmatrix
            X = dmatrix(formula, industry_df)       # Create the design matrix
            
            # Calculate VIF (Variance Inflation Factor) for each feature
            vif_data = pd.DataFrame({
                "Feature": METRICS,
                "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            })

            correlated_features = vif_data[vif_data["VIF"] > 5]

            condition_number = cond(industry_df.corr())
            
            vif_data['Industry'] = industry
            vif_data['Condition_No'] = condition_number
            vif_data['No_samples_raw'] = nrows_industry_df
            vif_data['No_samples'] = len(industry_df)

            # print(f"Correlated features are: {correlated_features}")

            # vif_data = vif_data[['Industry', 'Feature', 'VIF', 'Condition_No', 'No_samples_raw', 'No_samples']]
            # VIF_LIST.append(vif_data)



    if industry_df.shape[0] <= industry_df.shape[1]:
        print(f"Not enough data to perform regression for industry: {industry}")
        return

    if FILTER_OUTLIERS_METRICS == True:
        industry_df = filter_outliers(industry_df, METRICS, method = 'IQR')
    
    if FILTER_OUTLIERS_TARGET == True:
        industry_df = filter_outliers(industry_df, TARGET, method = 'clip')

    # Begin Skewness test ====================================================================
    if PERFORM_SKEW_TEST:
        skewness = industry_df.skew()    # skewness test: 

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

            # Create pairplot for scatter plots of all features in the DataFrame
            scatter_plot_filename = os.path.join(RESULTS_DIR, f"{industry}_scatter_plot.png")
            pairplot = sns.pairplot(industry_df)
            plt.suptitle(f"Scatter Plot for {industry}", y=1.02)  # Adjust y for spacing
            # plt.tight_layout()
            plt.savefig(scatter_plot_filename)
            plt.close()

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

    if PERFORM_REGRESSION == True:
        
        rf = RandomForestRegressor(random_state=42)
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [10, 20, 30, None]
        # }

        param_grid = {
                    'n_estimators': [50,100, 150,200],
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

        # print(f"Correlated features are: {correlated_features}")

        vif_data = vif_data[['Industry', 'Feature', 'VIF', 'Condition_No', 'No_samples_raw', 'No_samples','mse_scores','mae_scores','r2_scores']]
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

import pickle
with open('examples/Quant_Rating/datasets/industry_dfs_US_22-10-2024.pkl', 'rb') as f:
        industry_dfs = pickle.load(f)

# industry_data_path = 'examples/Quant_Rating/datasets/industry_data_rs3000'        #change the path here for quarter and annual, currently quarter
# industry_files = os.listdir(industry_data_path)
# industry_files = industry_files[:2]
# # industry_files = ['Auto - Parts.xlsx']
# # print(industry_files[:2])

# industry_dfs = {}
# for file in industry_files:
#     if file.endswith('.xlsx'):
#         industry_name = file.split('.')[0]
#         file_path = os.path.join(industry_data_path, file)
#         industry_dfs[industry_name] = pd.read_excel(file_path)

num_quarters = 24     # 24/4 = 6 years

rf_value_coeffs_df = pd.DataFrame()

perm_importance_value_coeffs_df = pd.DataFrame()

dict_industry = {}
# Iterate over each industry DataFrame and perform regression analysis
for industry, df in industry_dfs.items():
    # print(f"Industry: {industry}, Number of Symbols: {len(df['symbol'].unique())}, Data Shape: {df.shape}")
    if len(df['symbol'].unique()) >= 7 and df.shape[0] > df.shape[1] - 5:
        print(f"{industry} is processing")
        perform_regression(df, industry)
    else:
        print(f"Skipping industry: {industry} due to insufficient data.")

print(f"Total no. of industry: {len(industry_dfs)}, Number of non-collinear industry: {COUNT}")
# print(COLLINEAR_IND_LIST)


df_multicollinearity = pd.concat(VIF_LIST, ignore_index=True)
vif_filename = os.path.join(RESULTS_DIR, f"{WHICH_FACTOR}_Industry_Multicollinearity.xlsx")
df_multicollinearity.to_excel(vif_filename, index = False)

coeff_filename = os.path.join(RESULTS_DIR, f"rf_{WHICH_FACTOR}_coeffs_df.xlsx")
rf_value_coeffs_df.to_excel(coeff_filename)

coeff_filename_perm = os.path.join(RESULTS_DIR, f"perm_importance_{WHICH_FACTOR}_coeffs_df.xlsx")
perm_importance_value_coeffs_df.to_excel(coeff_filename_perm)


computation_time(start)

