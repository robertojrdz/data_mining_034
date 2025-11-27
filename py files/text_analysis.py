import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

df = pd.read_csv('./csv/steam_games_clean.csv')

all_names = ' '.join(df['name'].dropna().astype(str))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned_text = clean_text(all_names)
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['edition', 'game', 'dlc', 'pack', 'bundle', 'complete', 'ii', 'version', 'deluxe', 'ultimate', 'digital', 'special', 'iii', 'gold', 'enhanced', 'remastered', 'definitive', 'goty', 'steam', 'pc', 'vr', 'set', 'collection', 'cut', '3d', 's', 'online', 'free'])

# ============================================================================
# GENERATE WORD CLOUDS
# Word Cloud 1: All Games
wordcloud_all = WordCloud(width=1600, height=800, background_color='white', stopwords=custom_stopwords, max_words=200, relative_scaling=0.5, colormap='viridis', min_font_size=10).generate(cleaned_text)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - All Steam Games', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout(pad=0)
plt.savefig('wordcloud_all_games.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Word Cloud 2: Best Rated Games
threshold = df['positive_ratings'].quantile(0.75)
best_rated_games = df[df['positive_ratings'] >= threshold]
best_rated_text = ' '.join(best_rated_games['name'].dropna().astype(str))
best_rated_text_cleaned = clean_text(best_rated_text)

wordcloud_popular = WordCloud(width=1600, height=800, background_color='white', stopwords=custom_stopwords, max_words=200, relative_scaling=0.5, colormap='plasma', min_font_size=10).generate(best_rated_text_cleaned)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_popular, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Best Rated Games', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout(pad=0)
plt.savefig('wordcloud_best_rated_games.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Word Cloud 3: Free Games
free_games = df[df['price'] == 0]
free_text = ' '.join(free_games['name'].dropna().astype(str))
free_text_cleaned = clean_text(free_text)

wordcloud_free = WordCloud(width=1600, height=800, background_color='white', stopwords=custom_stopwords, max_words=200, relative_scaling=0.5, colormap='cool', min_font_size=10).generate(free_text_cleaned)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_free, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Free Games', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout(pad=0)
plt.savefig('wordcloud_free_games.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Word Cloud 4: Paid Games
paid_games = df[df['price'] > 0]
paid_text = ' '.join(paid_games['name'].dropna().astype(str))
paid_text_cleaned = clean_text(paid_text)

wordcloud_paid = WordCloud(width=1600, height=800, background_color='white', stopwords=custom_stopwords, max_words=200, relative_scaling=0.5, colormap='autumn', min_font_size=10).generate(paid_text_cleaned)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_paid, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Paid Games', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout(pad=0)
plt.savefig('wordcloud_paid_games.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\nWord cloud exported to 'wordcloud_best_rated_games.png'")
print("Word cloud exported to 'wordcloud_best_rated_games.png'")
print("Word cloud exported to 'wordcloud_free_games.png'")
print("Word cloud exported to 'wordcloud_paid_games.png'\n")