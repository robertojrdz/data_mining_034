import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('./csv/steam_games_clean.csv')
output_file = open('./descriptive_statistics.txt', 'w')

output_file.write("=" * 80 + '\n')
output_file.write("    STEAM GAMES - DESCRIPTIVE STATISTICS\n")
output_file.write("=" * 80 + '\n')
output_file.write(f"\nTotal games analyzed: {len(df)}\n")

numerical_fields = ['positive_ratings', 'negative_ratings', 'average_playtime', 'owners', 'price']

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

numerical_fields_processed = ['positive_ratings', 'negative_ratings', 'average_playtime', 'owners_numeric', 'price']

for field in numerical_fields_processed:
    display_name = field.replace('_numeric', '').replace('_', ' ').title()
    output_file.write(f"\n{'=' * 80}\n")
    output_file.write(f"    {display_name.upper()}\n")
    output_file.write("=" * 80 + '\n')
    
    data = df[field].dropna()
    
    if len(data) == 0:
        output_file.write("No data available for this field.\n")
        continue
    
    mean_val = data.mean()
    median_val = data.median()
    mode_result = stats.mode(data, keepdims=True)
    mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else "No unique mode"
    range_val = data.max() - data.min()
    variance_val = data.var()
    std_dev_val = data.std()
    
    output_file.write(f"Mean:               {mean_val:,.2f}\n")
    output_file.write(f"Median:             {median_val:,.2f}\n")
    output_file.write(f"Mode:               {mode_val if isinstance(mode_val, str) else f'{mode_val:,.2f}'}\n")
    output_file.write(f"Range:              {range_val:,.2f} (Min: {data.min():,.2f}, Max: {data.max():,.2f})\n")
    output_file.write(f"Variance:           {variance_val:,.2f}\n")
    output_file.write(f"Standard Deviation: {std_dev_val:,.2f}\n")

output_file.write(f"\n{'=' * 80}\n")
output_file.write("    RELEASE DATE\n")
output_file.write("=" * 80 + '\n')

df['release_date_dt'] = pd.to_datetime(df['release_date'], errors='coerce')
valid_dates = df['release_date_dt'].dropna()

if len(valid_dates) > 0:
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    date_range = (max_date - min_date).days
    
    output_file.write(f"Earliest Date:      {min_date.strftime('%Y-%m-%d')}\n")
    output_file.write(f"Latest Date:        {max_date.strftime('%Y-%m-%d')}\n")
    output_file.write(f"Range:              {date_range} days ({date_range / 365.25:.2f} years)")
else:
    output_file.write("No valid dates available for analysis.")

output_file.close()
print("\nAnalysis exported to 'descriptive_statistics.txt'\n")