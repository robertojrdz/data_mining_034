import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("flare")

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
df['release_year'] = df['release_date_dt'].dt.year

df_clean = df.dropna(subset=['positive_ratings', 'negative_ratings', 'average_playtime', 'owners_numeric', 'price'])

# ============================================================================
# 1. RATINGS ANALYSIS
fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('Ratings Analysis', fontsize=16, fontweight='bold')

rango0_0 = (df_clean['positive_ratings'].min(), df_clean['positive_ratings'].max()/250)
axes[0, 0].hist(df_clean['positive_ratings'], bins=50, range=rango0_0, color='green', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Positive Ratings')
axes[0, 0].set_ylabel('Number of Games')
axes[0, 0].set_title('Distribution of Positive Ratings')
axes[0, 0].grid(True, alpha=0.3)

rango0_1 = (df_clean['negative_ratings'].min(), df_clean['negative_ratings'].max()/46)
axes[0, 1].hist(df_clean['negative_ratings'], bins=50, range=rango0_1, color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Negative Ratings')
axes[0, 1].set_ylabel('Number of Games')
axes[0, 1].set_title('Distribution of Negative Ratings')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(df_clean['positive_ratings'], df_clean['negative_ratings'], alpha=0.5, s=20, c='blue')
axes[1, 0].set_xlabel('Positive Ratings')
axes[1, 0].set_ylabel('Negative Ratings')
axes[1, 0].set_title('Positive vs Negative Ratings')
axes[1, 0].grid(True, alpha=0.3)

ratings_data = [df_clean['positive_ratings'], df_clean['negative_ratings']]
box = axes[1, 1].boxplot(ratings_data, tick_labels=['Positive', 'Negative'], patch_artist=True)
box['boxes'][0].set_facecolor('green')
box['boxes'][1].set_facecolor('red')
axes[1, 1].set_ylabel('Number of Ratings')
axes[1, 1].set_title('Ratings Distribution Comparison')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations_ratings.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 2. PLAYTIME ANALYSIS
fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Average Playtime Analysis', fontsize=16, fontweight='bold')

rango0 = (df_clean['average_playtime'].min(), df_clean['average_playtime'].max()/100)
axes[0].hist(df_clean['average_playtime'], bins=50, range=rango0,  color='purple', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Average Playtime (minutes)')
axes[0].set_ylabel('Number of Games')
axes[0].set_title('Distribution of Average Playtime')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean['average_playtime'], vert=True, patch_artist=True)
axes[1].set_ylabel('Average Playtime (minutes)')
axes[1].set_title('Playtime Distribution (Box Plot)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations_playtime.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 3. OWNERS ANALYSIS
fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle('Game Owners Analysis', fontsize=16, fontweight='bold')

owner_counts = df['owners'].value_counts().sort_index()
axes[0].bar(range(len(owner_counts)), owner_counts.values, color='teal', alpha=0.7)
axes[0].set_xlabel('Owner Range')
axes[0].set_ylabel('Number of Games')
axes[0].set_title('Distribution of Games by Owner Range')
axes[0].set_xticks(range(len(owner_counts)))
axes[0].set_xticklabels(owner_counts.index, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

top_owners = owner_counts.nlargest(8)
other_count = owner_counts.sum() - top_owners.sum()
if other_count > 0:
    top_owners['Others'] = other_count
axes[1].pie(top_owners.values, labels=top_owners.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Percentage Distribution of Games by Owner Range')

plt.tight_layout()
plt.savefig('visualizations_owners.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 4. PRICE ANALYSIS
fig4, axes = plt.subplots(1, 2, figsize=(15, 6))
fig4.suptitle('Price Analysis', fontsize=16, fontweight='bold')

axes[0].hist(df_clean['price'], bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Number of Games')
axes[0].set_title('Distribution of Game Prices')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_clean['price'], vert=True, patch_artist=True)
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Price Distribution (Box Plot)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations_price.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 5. RELEASE DATE ANALYSIS
fig5, axes = plt.subplots(1, 2, figsize=(15, 6))
fig5.suptitle('Release Date Analysis', fontsize=16, fontweight='bold')

year_counts = df['release_year'].value_counts().sort_index()
axes[0].plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Games Released')
axes[0].set_title('Games Released Over Time')
axes[0].grid(True, alpha=0.3)

axes[1].bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Games Released')
axes[1].set_title('Games Released by Year (Bar Chart)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations_release_dates.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 6. COMBINED SCATTER PLOTS
fig7, axes = plt.subplots(2, 2, figsize=(15, 12))
fig7.suptitle('Relationships Between Variables', fontsize=16, fontweight='bold')

axes[0, 0].scatter(df_clean['price'], df_clean['average_playtime'], alpha=0.5, s=20, c='blue')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Average Playtime (minutes)')
axes[0, 0].set_title('Price vs Average Playtime')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(df_clean['positive_ratings'], df_clean['owners_numeric'], alpha=0.5, s=20, c='green')
axes[0, 1].set_xlabel('Positive Ratings')
axes[0, 1].set_ylabel('Owners (estimated)')
axes[0, 1].set_title('Positive Ratings vs Owners')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(df_clean['price'], df_clean['positive_ratings'], alpha=0.5, s=20, c='orange')
axes[1, 0].set_xlabel('Price ($)')
axes[1, 0].set_ylabel('Positive Ratings')
axes[1, 0].set_title('Price vs Positive Ratings')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(df_clean['average_playtime'], df_clean['positive_ratings'], alpha=0.5, s=20, c='purple')
axes[1, 1].set_xlabel('Average Playtime (minutes)')
axes[1, 1].set_ylabel('Positive Ratings')
axes[1, 1].set_title('Playtime vs Positive Ratings')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations_relationships.png', dpi=300, bbox_inches='tight')

print("\nGenerated files:")
print("1. visualizations_ratings.png")
print("2. visualizations_playtime.png")
print("3. visualizations_owners.png")
print("4. visualizations_price.png")
print("5. visualizations_release_dates.png")
print("6. visualizations_relationships.png\n")