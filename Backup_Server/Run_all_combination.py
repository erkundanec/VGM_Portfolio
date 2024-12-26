import sys
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

RATING_OR_SCORE_OPTIONS = ['score']             # ['score', 'rating']
VGM_METHOD_OPTIONS = ['z-score_avg']      # ['min-max_avg','z-score_avg', 'mom-value', 'mom-growth']           
                                                # 'only-momentum',
                                                # 'min-max_avg', 'percentile_avg', 'z-score_avg', 'rank_avg', 
                                                # 'value','growth','mom-value', 'mom-growth', 'growth_value', 'Avg_VGM','no-momentum' 

N_TOP_TICKERS_OPTIONS =  ['100']                # ['15','20','30','40','50','60','70','80','90','100']  
PORTFOLIO_SIZE_OPTIONS = ['20','30','50']                 # ['20','25','30']
PORTFOLIO_FREQ_OPTIONS = ['Fortnight','Monthly']            # ['Weekly', 'Fortnight', 'Monthly']
CONSIDER_RISK_FACTOR_OPTIONS = ['yes','no']           # ['yes', 'no']
CONSIDER_DIV_OPTIONS = ['yes','no']                   # ['yes', 'no']

START_DATE_OPTIONS = ['2020-01-01']
END_DATE_OPTIONS = ['2024-10-22']

# Define a function to run a single command
def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command)

if __name__ == "__main__":

    commands = []
    for rating_or_score in RATING_OR_SCORE_OPTIONS:
        for vgm_method in VGM_METHOD_OPTIONS:
            for portfolio_freq in PORTFOLIO_FREQ_OPTIONS:
                for n_top_stocks in N_TOP_TICKERS_OPTIONS:
                    for portfolio_size in PORTFOLIO_SIZE_OPTIONS:
                        for consider_risk in CONSIDER_RISK_FACTOR_OPTIONS:
                            for consider_div in CONSIDER_DIV_OPTIONS:
                                for start_date,end_date in zip(START_DATE_OPTIONS,END_DATE_OPTIONS):

                                    command = [
                                        'python', 'examples/Quant_Rating/20240816_Quant_Portfolio_v3_parametric.py',
                                        '--RATING_OR_SCORE', rating_or_score,
                                        '--VGM_METHOD', vgm_method,
                                        '--PORTFOLIO_FREQ', portfolio_freq,
                                        '--N_TOP_TICKERS', n_top_stocks,
                                        '--PORTFOLIO_SIZE', portfolio_size,
                                        '--CONSIDER_RISK_FACTOR', consider_risk,
                                        '--CONSIDER_DIV', consider_div,
                                        '--START_DATE', start_date,
                                        '--END_DATE', end_date,
                                    ]
                                    commands.append(command)

    # Run all the commands in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(run_command, commands)

    
# Indian VGM Score - 17:54