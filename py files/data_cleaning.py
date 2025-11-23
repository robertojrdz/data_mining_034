import pandas as pd

df = pd.read_csv('./csv/steam_games.csv')

fields_to_keep = ['appid', 'name', 'release_date', 'developer', 'platforms', 'positive_ratings', 'negative_ratings', 'average_playtime', 'owners', 'price']
df_filtered = df[fields_to_keep].copy()

def parse_owners(owners_str):
    try:
        if pd.isna(owners_str):
            return 0
        min_owners = int(owners_str.split('-')[0].replace(',', ''))
        return min_owners
    except:
        return 0

df_filtered['min_owners'] = df_filtered['owners'].apply(parse_owners)
df_filtered = df_filtered[df_filtered['min_owners'] >= 50000]
df_filtered = df_filtered.drop('min_owners', axis=1)

df_filtered.to_csv('./csv/steam_games_clean.csv', index=False)