import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace these with actual portfolio and Nifty data)
sectors = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 
           'Financials', 'Health Care', 'Industrials', 'Utilities', 'Information Technology', 'Materials']

nifty_weights = [2.59, 6.74, 10.04, 14.24, 28.19, 3.6, 3.0, 1.83, 20.82, 8.97]  # Nifty sector weights
portfolio_weights = [2.5, 5.5, 9.0, 12.5, 35, 19.5, 12.3, 0, 3, 0]  # Portfolio sector weights

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Positioning data
index = np.arange(len(sectors))
bar_width = 0.4

# Plot Nifty and Portfolio weights
ax.barh(index - bar_width/2, nifty_weights, bar_width, label='Nifty', color='blue', alpha=0.6)
ax.barh(index + bar_width/2, portfolio_weights, bar_width, label='Portfolio', color='red', alpha=0.6)

# Labels and titles
ax.set_xlabel('Percentage (%)')
ax.set_title('Sector Allocation: Nifty vs Portfolio')
ax.set_yticks(index)
ax.set_yticklabels(sectors)

# Adding legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
