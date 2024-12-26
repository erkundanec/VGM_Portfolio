import sys
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

FACTOR_OPTIONS = ['value', 'growth']                   # ['value', 'growth','profitability']

# Define a function to run a single command
def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command)

if __name__ == "__main__":
    commands = []
    for factor in FACTOR_OPTIONS:
        command = [
            'python', 'examples/Quant_Rating/20241128_factor_regression_v1.py',
            '--WHICH_FACTOR', factor,
        ]
        commands.append(command)

    # Run all the commands in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(run_command, commands)