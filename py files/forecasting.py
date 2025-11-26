import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('./csv/steam_games_clean.csv')

def parse_owners_midpoint(owners_str):
    try:
        if pd.isna(owners_str):
            return np.nan
        parts = owners_str.split('-')
        min_val = int(parts[0].replace(',', ''))
        max_val = int(parts[1].replace(',', ''))
        return (min_val + max_val) / 2
    except:
        return np.nan

df['owners_numeric'] = df['owners'].apply(parse_owners_midpoint)
df['release_date_dt'] = pd.to_datetime(df['release_date'], errors='coerce')
df_clean = df.dropna(subset=['release_date_dt', 'positive_ratings', 'negative_ratings', 'average_playtime', 'owners_numeric', 'price'])
df_clean['release_year'] = df_clean['release_date_dt'].dt.year
df_clean['release_month'] = df_clean['release_date_dt'].dt.month
df_clean['release_year_month'] = df_clean['release_date_dt'].dt.to_period('M')

output_file = open('forecasting_prediction.txt', 'w')

output_file.write('=' * 80 + '\n')
output_file.write("TIME SERIES PREDICTION - STEAM GAMES DATASET\n")
output_file.write('=' * 80 + '\n')
output_file.write(f"\nDataset size: {len(df_clean)} games\n")
output_file.write(f"Date range: {df_clean['release_date_dt'].min()} to {df_clean['release_date_dt'].max()}\n\n")

# ============================================================================
# MODEL 1: PREDICT NUMBER OF GAMES RELEASED PER YEAR
output_file.write('=' * 80 + '\n')
output_file.write("MODEL 1: PREDICTING NUMBER OF GAMES RELEASED PER YEAR\n")
output_file.write('=' * 80 + '\n')
yearly_releases = df_clean.groupby('release_year').size().reset_index(name='num_games')
yearly_releases = yearly_releases.sort_values('release_year')

output_file.write(f"\nHistorical data: {len(yearly_releases)} years\n")
output_file.write("\nYearly Game Releases:\n")
output_file.write('=' * 27 + '\n')
for _, row in yearly_releases.iterrows():
    output_file.write(f"  {int(row['release_year'])}: {int(row['num_games'])} games\n")

X_year = yearly_releases['release_year'].values.reshape(-1, 1)
y_year = yearly_releases['num_games'].values
split_point = len(X_year) - 2
X_train_year = X_year[:split_point]
y_train_year = y_year[:split_point]
X_test_year = X_year[split_point:]
y_test_year = y_year[split_point:]
model_year = LinearRegression()
model_year.fit(X_train_year, y_train_year)
y_pred_test_year = model_year.predict(X_test_year)
r2_year = r2_score(y_test_year, y_pred_test_year)
rmse_year = np.sqrt(mean_squared_error(y_test_year, y_pred_test_year))
mae_year = mean_absolute_error(y_test_year, y_pred_test_year)

output_file.write("\n" + '=' * 40 + '\n')
output_file.write("MODEL PERFORMANCE\n")
output_file.write('=' * 40 + '\n')
output_file.write(f"Training years: {int(X_train_year[0][0])} - {int(X_train_year[-1][0])}\n")
output_file.write(f"Testing years:  {int(X_test_year[0][0])} - {int(X_test_year[-1][0])}\n")
output_file.write(f"\nR² Score: {r2_year:.4f}\n")
output_file.write(f"RMSE:     {rmse_year:.2f} games\n")
output_file.write(f"MAE:      {mae_year:.2f} games\n")

current_year = df_clean['release_year'].max()
future_years = np.array([[current_year + 1], [current_year + 2], [current_year + 3], [current_year + 4], [current_year + 5], [current_year + 6], [current_year + 7], [current_year + 8], [current_year + 9], [current_year + 10]])
future_predictions = model_year.predict(future_years)

output_file.write("\n" + '=' * 40 + '\n')
output_file.write("FUTURE PREDICTIONS\n")
output_file.write('=' * 40 + '\n')
for i, year in enumerate(future_years):
    output_file.write(f"  {int(year[0])}: {int(future_predictions[i])} games (predicted)\n")
output_file.write(f"\nModel Equation: y = {model_year.coef_[0]:.2f}x + {model_year.intercept_:.2f}\n")
output_file.write(f"Interpretation: Each year sees approximately {model_year.coef_[0]:.2f} {'more' if model_year.coef_[0] > 0 else 'fewer'} games released\n")

# ============================================================================
# MODEL 2: PREDICT AVERAGE PRICE OVER TIME
output_file.write("\n" + '=' * 80 + '\n')
output_file.write("MODEL 2: PREDICTING AVERAGE GAME PRICE OVER TIME\n")
output_file.write('=' * 80 + '\n')

yearly_prices = df_clean.groupby('release_year').agg({'price': ['mean', 'median', 'std', 'count']}).reset_index()
yearly_prices.columns = ['release_year', 'mean_price', 'median_price', 'std_price', 'count']
yearly_prices = yearly_prices.sort_values('release_year')

output_file.write("\nYearly Average Prices:\n")
output_file.write('-' * 50 + '\n')
for _, row in yearly_prices.iterrows():
    output_file.write(f"  {int(row['release_year'])}: Mean=${row['mean_price']:.2f}, Median=${row['median_price']:.2f}, Std=${row['std_price']:.2f}\n")

X_price = yearly_prices['release_year'].values.reshape(-1, 1)
y_price = yearly_prices['mean_price'].values
split_point = len(X_price) - 2
X_train_price = X_price[:split_point]
y_train_price = y_price[:split_point]
X_test_price = X_price[split_point:]
y_test_price = y_price[split_point:]
model_price = LinearRegression()
model_price.fit(X_train_price, y_train_price)
y_pred_test_price = model_price.predict(X_test_price)
r2_price = r2_score(y_test_price, y_pred_test_price)
rmse_price = np.sqrt(mean_squared_error(y_test_price, y_pred_test_price))
mae_price = mean_absolute_error(y_test_price, y_pred_test_price)

output_file.write("\n" + '-' * 50 + '\n')
output_file.write("MODEL PERFORMANCE\n")
output_file.write('-' * 50 + '\n')
output_file.write(f"R² Score: {r2_price:.4f}\n")
output_file.write(f"RMSE:     ${rmse_price:.2f}\n")
output_file.write(f"MAE:      ${mae_price:.2f}\n")

future_price_predictions = model_price.predict(future_years)

output_file.write("\n" + '-' * 40 + '\n')
output_file.write("FUTURE PREDICTIONS\n")
output_file.write('-' * 40 + '\n')
for i, year in enumerate(future_years):
    output_file.write(f"  {int(year[0])}: ${future_price_predictions[i]:.2f} (predicted)\n")
output_file.write(f"\nModel Equation: y = {model_price.coef_[0]:.4f}x + {model_price.intercept_:.2f}\n")
trend = "increasing" if model_price.coef_[0] > 0 else "decreasing"
output_file.write(f"Interpretation: Average game price is {trend} by ${abs(model_price.coef_[0]):.2f} per year\n")

# ============================================================================
# CREATE VISUALIZATIONS
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Time Series Predictions - Steam Games', fontsize=16, fontweight='bold')

axes[0].scatter(X_train_year, y_train_year, color='blue', label='Training Data', s=50)
axes[0].scatter(X_test_year, y_test_year, color='lightgreen', label='Testing Data', s=50)
axes[0].plot(X_year, model_year.predict(X_year), 'r--', label='Fitted Line', linewidth=2)
axes[0].scatter(future_years, future_predictions, color='orange', marker='*', s=200, label='Future Predictions')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Games')
axes[0].set_title(f'Games Released per Year (R²={r2_year:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_train_price, y_train_price, color='blue', label='Training Data', s=50)
axes[1].scatter(X_test_price, y_test_price, color='lightgreen', label='Testing Data', s=50)
axes[1].plot(X_price, model_price.predict(X_price), 'r--', label='Fitted Line', linewidth=2)
axes[1].scatter(future_years, future_price_predictions, color='orange', marker='*', s=200, label='Future Predictions')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Price ($)')
axes[1].set_title(f'Average Price Over Time (R²={r2_price:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecasting_predictions.png', dpi=300, bbox_inches='tight')

output_file.write("\n" + '=' * 80 + '\n')
output_file.write("SUMMARY OF FORECASTING PREDICTIONS\n")
output_file.write('=' * 80 + '\n')
output_file.write("\nModel Performance Summary:\n")
output_file.write(f"  1. Games Released:     R² = {r2_year:.4f}\n")
output_file.write(f"  2. Average Price:      R² = {r2_price:.4f}\n")
output_file.write("\nKey Insights:\n")
output_file.write(f"  - Number of games is {('increasing' if model_year.coef_[0] > 0 else 'decreasing')} by ~{abs(model_year.coef_[0]):.0f} games/year\n")
output_file.write(f"  - Average price is {('increasing' if model_price.coef_[0] > 0 else 'decreasing')} by ~${abs(model_price.coef_[0]):.2f}/year\n")
output_file.close()

print("\nResults saved to: forcasting_prediction.txt")
print("Visualization saved to: forecastings_predictions.png\n")