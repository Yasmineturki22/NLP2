import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

corpus_cleaned = [clean_text(doc) for doc in newsgroup_data]


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)

tfidf = tfidf_vectorizer.fit_transform(corpus_cleaned)
tf = tf_vectorizer.fit_transform(corpus_cleaned)


lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tf)

nmf = NMF(n_components=10, random_state=42)
nmf.fit(tfidf)


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        ax.set_title(f'Topic {topic_idx +1}')
        ax.invert_yaxis()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_top_words(lda, tf_vectorizer.get_feature_names_out(), 10, 'Top words per topic (LDA)')
plot_top_words(nmf, tfidf_vectorizer.get_feature_names_out(), 10, 'Top words per topic (NMF)')


def generate_wordcloud(model, feature_names, title):
    topic_words = {}
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-50 - 1:-1]:
            word = feature_names[i]
            topic_words[word] = topic_words.get(word, 0) + topic[i]

    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(topic_words)
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_wordcloud(lda, tf_vectorizer.get_feature_names_out(), 'WordCloud (LDA Topics)')
generate_wordcloud(nmf, tfidf_vectorizer.get_feature_names_out(), 'WordCloud (NMF Topics)')
