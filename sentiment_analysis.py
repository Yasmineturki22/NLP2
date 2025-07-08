import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Télécharger les ressources nécessaires
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Charger les données
df = pd.read_csv('tweets-data.csv').sample(500, random_state=42)

# Nettoyage des tweets
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", '', str(tweet).lower())
    tokens = word_tokenize(tweet)
    tokens = [w for w in tokens if w not in stopwords.words('english') and len(w) > 2]
    return ' '.join(tokens)
print(df.columns)

df['cleaned_tweet'] = df['Tweets'].astype(str).apply(clean_tweet)

# Analyse VADER
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return pd.Series([label, score])

df[['vader_sentiment', 'vader_score']] = df['cleaned_tweet'].apply(vader_sentiment)

# Analyse Transformers (Hugging Face)
classifier = pipeline('sentiment-analysis')

def transformer_sentiment(text):
    result = classifier(text[:512])[0]
    label = result['label'].lower()
    score = result['score']
    return pd.Series([label, score])

df[['transformer_sentiment', 'transformer_score']] = df['cleaned_tweet'].apply(transformer_sentiment)

# Sauvegarde du fichier final
df.to_csv('tweets_sentiment_analysis.csv', index=False)

print("✅ Sentiment analysis terminé. Résultats enregistrés dans 'tweets_sentiment_analysis.csv'.")
