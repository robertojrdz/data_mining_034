import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("LINEAR REGRESSION MODEL - STEAM GAMES DATASET")
print("=" * 80)

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

df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
df['positive_ratio'] = df['positive_ratings'] / df['total_ratings']

df_clean = df.dropna(subset=['positive_ratings', 'negative_ratings', 'average_playtime', 
                               'owners_numeric', 'price', 'total_ratings', 'positive_ratio'])

# ============================================================================
# MODEL: PREDICT PRICE

features_price = ['positive_ratings', 'negative_ratings', 'average_playtime', 'owners_numeric']
X_price = df_clean[features_price]
y_price = df_clean['price']
X_train_pr, X_test_pr, y_train_pr, y_test_pr = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
scaler_pr = StandardScaler()
X_train_pr_scaled = scaler_pr.fit_transform(X_train_pr)
X_test_pr_scaled = scaler_pr.transform(X_test_pr)
model_price = LinearRegression()
model_price.fit(X_train_pr_scaled, y_train_pr)
y_pred_train_pr = model_price.predict(X_train_pr_scaled)
y_pred_test_pr = model_price.predict(X_test_pr_scaled)
r2_train_pr = r2_score(y_train_pr, y_pred_train_pr)
r2_test_pr = r2_score(y_test_pr, y_pred_test_pr)
mse_pr = mean_squared_error(y_test_pr, y_pred_test_pr)
mae_pr = mean_absolute_error(y_test_pr, y_pred_test_pr)
rmse_pr = np.sqrt(mse_pr)

# ============================================================================
# VISUALIZATION 1: ACTUAL VS PREDICTED (ALL MODELS)
plt.suptitle('Actual vs Predicted Values', fontsize=16, fontweight='bold')
plt.scatter(y_test_pr, y_pred_test_pr, alpha=0.5, s=20, c='green')
plt.plot([y_test_pr.min(), y_test_pr.max()], [y_test_pr.min(), y_test_pr.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Price Prediction - R² = {r2_test_pr:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 3: FEATURE IMPORTANCE (COEFFICIENTS)
plt.suptitle('Feature Importance (Standardized Coefficients)', fontsize=16, fontweight='bold')
plt.barh(features_price, model_price.coef_, color='lightgreen')
plt.xlabel('Coefficient Value')
plt.title('Price Prediction Features')
plt.grid(True, alpha=0.3, axis='x')
plt.savefig('linear_model_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 4: R² COMPARISON
plt.bar(['Training R²', 'Testing R²'], [r2_train_pr, r2_test_pr], color=['steelblue', 'coral'])
plt.title('Model Performance Comparison - R² Scores', fontsize=14, fontweight='bold')
plt.ylabel('R² Score')
plt.xlabel('Price')
plt.grid(True, alpha=0.3, axis='y')
plt.text('Training R²', r2_train_pr, f'{r2_train_pr:.4f}', ha='center', va='bottom', fontsize=10)
plt.text('Testing R²', r2_test_pr, f'{r2_test_pr:.4f}', ha='center', va='bottom', fontsize=10)
plt.savefig('linear_model_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE RESULTS TO TEXT FILE
with open('linear_model_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("LINEAR REGRESSION MODEL RESULTS - STEAM GAMES DATASET\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total games analyzed {len(df_clean)} games\n")
    f.write(f"Train/Test split: 80% / 20%\n\n")
    f.write("=" * 80 + "\n")
    f.write("    PREDICTING PRICE\n")
    f.write("=" * 80 + "\n")
    f.write(f"Features: {', '.join(features_price)}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  R² Score (Training):   {r2_train_pr:.4f}\n")
    f.write(f"  R² Score (Testing):    {r2_test_pr:.4f}\n")
    f.write(f"  RMSE:                  ${rmse_pr:.2f}\n")
    f.write(f"  MAE:                   ${mae_pr:.2f}\n\n")
    f.write("Feature Coefficients:\n")
    for feature, coef in zip(features_price, model_price.coef_):
        f.write(f"  {feature:20s}: {coef:10.4f}\n")
    f.write(f"  {'Intercept':20s}: {model_price.intercept_:10.4f}\n\n")

print("✓ Results exported to: linear_model_results.txt")
print("\nGenerated files:")
print("  1. linear_model_predictions.png - Actual vs Predicted plots")
print("  2. linear_model_residuals.png - Residuals analysis")
print("  3. linear_model_coefficients.png - Feature importance")
print("  4. linear_model_r2_comparison.png - R² scores comparison")
print("  5. linear_model_results.txt - Complete results summary")