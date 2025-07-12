# analyze_curve.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved equity curve
file_path = "results/final_portfolio_equity.csv"
df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

# Plotting
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(15, 8))

# Plot the equity curve
ax.plot(df.index, df['equity'], label='Strategy Equity', color='blue', linewidth=2)

# Plot the initial capital line for reference
ax.axhline(y=df['equity'].iloc[0], color='grey', linestyle='--', label=f"Initial Capital (${df['equity'].iloc[0]:,.0f})")

# Find and plot the peak equity to visualize drawdown
peak = df['equity'].expanding().max()
ax.plot(peak.index, peak, label='Peak Equity', color='green', linestyle=':', linewidth=1.5)

# Fill the area between peak and equity to show drawdown
ax.fill_between(df.index, df['equity'], peak, color='red', alpha=0.2, label='Drawdown')

# Formatting
ax.set_title('Equity Curve Analysis for Failed Final Candidate', fontsize=16)
ax.set_ylabel('Portfolio Value ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()