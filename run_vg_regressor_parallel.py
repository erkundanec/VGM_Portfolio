import sys
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

FACTOR_OPTIONS = ['value', 'growth']   #, 'profitability']  # Include all desired factors

# Define a function to run a single command
def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command)

if __name__ == "__main__":
    commands = []
    for factor in FACTOR_OPTIONS:
        command = [
            'python', 'c:/Users/Datacore/Documents/MEGA/Codes/03_Github_Repository/VGM_Score/20241226_factor_regression_v2.py',
            '--WHICH_FACTOR', factor,
        ]
        commands.append(command)

    # Run all the commands in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(run_command, commands)