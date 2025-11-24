import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, f_oneway, ttest_ind, kruskal, mannwhitneyu

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

output_file = open('statistical_tests.txt', 'w')
output_file.write("=" * 80 + "\n")
output_file.write("    STATISTICAL HYPOTHESIS TESTING - STEAM GAMES DATASET\n")
output_file.write("=" * 80 + "\n")
output_file.write(f"\nTotal games analyzed: {len(df)}\n")

# ============================================================================
# CREATE GROUPS FOR TESTING
df['price_category'] = pd.cut(df['price'], bins=[-0.01, 0, 10, 30, 1000], labels=['Free', 'Low ($0-10)', 'Medium ($10-30)', 'High ($30+)'])
df['owner_category'] = df['owners'].apply(lambda x: x if pd.notna(x) else 'Unknown')
df['platform_category'] = df['platforms']
df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
df['positive_ratio'] = df['positive_ratings'] / df['total_ratings']
df['rating_score'] = pd.cut(df['positive_ratio'], bins=[0, 0.5, 0.75, 1.0], labels=['Negative (<50%)', 'Mixed (50-75%)', 'Positive (75%+)'])

# ============================================================================
# AVERAGE PLAYTIME BY PRICE CATEGORY
output_file.write("\n" + "=" * 80 + "\n")
output_file.write("    AVERAGE PLAYTIME BY PRICE CATEGORY\n")
output_file.write("=" * 80 + "\n")
output_file.write("\nHypothesis: Different price categories have different average playtimes\n")

groups_playtime_price = []
price_categories = ['Free', 'Low ($0-10)', 'Medium ($10-30)', 'High ($30+)']
for cat in price_categories:
    group_data = df[df['price_category'] == cat]['average_playtime'].dropna()
    if len(group_data) > 0:
        groups_playtime_price.append(group_data)
        output_file.write(f"\n{cat}: n={len(group_data)}, mean={group_data.mean():.2f}, std={group_data.std():.2f}")

output_file.write("\n\n--- Normality Tests ---\n")
all_normal = True
for i, group in enumerate(groups_playtime_price):
    if len(group) > 3 and len(group) < 5000:
        stat, p_value = shapiro(group)
        output_file.write(f"{price_categories[i]}: W={stat:.4f}, p-value={p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Not Normal)'}")
        if p_value <= 0.05:
            all_normal = False

output_file.write("\n\n--- Homogeneity of Variance Test ---\n")
levene_stat, levene_p = levene(*groups_playtime_price)
output_file.write(f"Levene's Test: W={levene_stat:.4f}, p-value={levene_p:.4f}\n")
output_file.write(f"Equal variances: {'Yes' if levene_p > 0.05 else 'No'}\n")

if all_normal and levene_p > 0.05:
    output_file.write("\n--- ANOVA Test ---\n")
    f_stat, p_value = f_oneway(*groups_playtime_price)
    output_file.write(f"F-statistic: {f_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Price categories have different playtimes\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")
else:
    output_file.write("\n--- Kruskal-Wallis Test  ---\n")
    h_stat, p_value = kruskal(*groups_playtime_price)
    output_file.write(f"H-statistic: {h_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Price categories have different playtimes\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")

# ============================================================================
# POSITIVE RATINGS BY RATING QUALITY
output_file.write("\n" + "=" * 80 + "\n")
output_file.write("    POSITIVE RATINGS BY RATING QUALITY\n")
output_file.write("=" * 80 + "\n")
output_file.write("\nHypothesis: Different rating quality groups have different positive ratings counts\n")

groups_ratings_quality = []
quality_categories = ['Negative (<50%)', 'Mixed (50-75%)', 'Positive (75%+)']
for cat in quality_categories:
    group_data = df[df['rating_score'] == cat]['positive_ratings'].dropna()
    if len(group_data) > 0:
        groups_ratings_quality.append(group_data)
        output_file.write(f"\n{cat}: n={len(group_data)}, mean={group_data.mean():.2f}, std={group_data.std():.2f}")

output_file.write("\n\n--- Normality Tests ---\n")
all_normal = True
for i, group in enumerate(groups_ratings_quality):
    if len(group) > 3 and len(group) < 5000:
        stat, p_value = shapiro(group)
        output_file.write(f"{quality_categories[i]}: W={stat:.4f}, p-value={p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Not Normal)'}\n")
        if p_value <= 0.05:
            all_normal = False

output_file.write("\n--- Homogeneity of Variance Test ---\n")
levene_stat, levene_p = levene(*groups_ratings_quality)
output_file.write(f"Levene's Test: W={levene_stat:.4f}, p-value={levene_p:.4f}\n")
output_file.write(f"Equal variances: {'Yes' if levene_p > 0.05 else 'No'}\n")

if all_normal and levene_p > 0.05:
    output_file.write("\n--- ANOVA Test ---\n")
    f_stat, p_value = f_oneway(*groups_ratings_quality)
    output_file.write(f"F-statistic: {f_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Rating quality affects positive ratings\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")
else:
    output_file.write("\n--- Kruskal-Wallis Test  ---\n")
    h_stat, p_value = kruskal(*groups_ratings_quality)
    output_file.write(f"H-statistic: {h_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Rating quality affects positive ratings\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")

# ============================================================================
# PRICE BY RATING QUALITY (T-TEST)
output_file.write("\n" + "=" * 80 + "\n")
output_file.write("    PRICE COMPARISON - POSITIVE vs NEGATIVE RATINGS\n")
output_file.write("=" * 80 + "\n")
output_file.write("\nHypothesis: Games with excellent ratings have different prices than poor-rated games\n")

group_positive = df[df['rating_score'] == 'Positive (75%+)']['price'].dropna()
group_negative = df[df['rating_score'] == 'Negative (<50%)']['price'].dropna()

output_file.write(f"\nPositive (75%+): n={len(group_positive)}, mean=${group_positive.mean():.2f}, std=${group_positive.std():.2f}\n")
output_file.write(f"Negative (<50%): n={len(group_negative)}, mean=${group_negative.mean():.2f}, std=${group_negative.std():.2f}\n")

output_file.write("\n--- Normality Tests ---\n")
if len(group_positive) > 3 and len(group_positive) < 5000:
    stat_exc, p_exc = shapiro(group_positive)
    output_file.write(f"Positive: W={stat_exc:.4f}, p-value={p_exc:.4f} {'(Normal)' if p_exc > 0.05 else '(Not Normal)'}\n")
if len(group_negative) > 3 and len(group_negative) < 5000:
    stat_poor, p_poor = shapiro(group_negative)
    output_file.write(f"Negative: W={stat_poor:.4f}, p-value={p_poor:.4f} {'(Normal)' if p_poor > 0.05 else '(Not Normal)'}\n")

output_file.write("\n--- Homogeneity of Variance Test ---\n")
levene_stat, levene_p = levene(group_positive, group_negative)
output_file.write(f"Levene's Test: W={levene_stat:.4f}, p-value={levene_p:.4f}\n")
output_file.write(f"Equal variances: {'Yes' if levene_p > 0.05 else 'No'}\n")

if p_exc > 0.05 and p_poor > 0.05 and levene_p > 0.05:
    output_file.write("\n--- Independent T-Test ---\n")
    t_stat, p_value = ttest_ind(group_positive, group_negative)
    output_file.write(f"T-statistic: {t_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Prices differ between rating qualities\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")
else:
    output_file.write("\n--- Mann-Whitney U Test  ---\n")
    u_stat, p_value = mannwhitneyu(group_positive, group_negative)
    output_file.write(f"U-statistic: {u_stat:.4f}\n")
    output_file.write(f"P-value: {p_value:.6f}\n")
    if p_value < 0.05:
        output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Prices differ between rating qualities\n")
    else:
        output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")

# ============================================================================
# OWNERS BY PRICE CATEGORY
output_file.write("\n" + "=" * 80 + "\n")
output_file.write("    OWNERS BY PRICE CATEGORY\n")
output_file.write("=" * 80 + "\n")
output_file.write("\nHypothesis: Different price categories have different owner counts\n")

groups_owners_price = []
for cat in price_categories:
    group_data = df[df['price_category'] == cat]['owners_numeric'].dropna()
    if len(group_data) > 0:
        groups_owners_price.append(group_data)
        output_file.write(f"\n{cat}: n={len(group_data)}, mean={group_data.mean():.2f}, std={group_data.std():.2f}")

output_file.write("\n\n--- Kruskal-Wallis Test ---\n")
h_stat, p_value = kruskal(*groups_owners_price)
output_file.write(f"H-statistic: {h_stat:.4f}\n")
output_file.write(f"P-value: {p_value:.6f}\n")
if p_value < 0.05:
    output_file.write(f"Result: SIGNIFICANT (p < 0.05) - Price categories have different owner counts\n")
else:
    output_file.write(f"Result: NOT SIGNIFICANT (p >= 0.05) - No significant difference\n")


output_file.write("\n" + "=" * 80 + "\n")
output_file.write("    NOTES\n")
output_file.write("=" * 80 + "\n")
output_file.write("\nAll tests performed with significance level α = 0.05")
output_file.write("\nNon-parametric tests were used when assumptions of normality or homogeneity of variance were violated.")

output_file.close()
print("\nResults exported to 'statistical_tests_results.txt'\n")