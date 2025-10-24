# scrape_news.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline

# Load sentiment analysis pipeline once
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"Warning: Could not load FinBERT model. Sentiment will be neutral. Error: {e}")
    sentiment_pipeline = None

def get_market_sentiment(max_headlines=10):
    """
    Scrapes Moneycontrol for market headlines, analyzes sentiment, and returns an average score.
    Score: 1.0 (Positive) to -1.0 (Negative).
    Returns 0.0 if scraping or analysis fails.
    """
    if not sentiment_pipeline:
        print("Sentiment analysis model not loaded. Returning neutral score.")
        return 0.0

    URL = "https://www.moneycontrol.com/news/business/markets/"
    try:
        page = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})
        if page.status_code != 200:
            print(f"Error: Failed to fetch {URL}, Status Code: {page.status_code}")
            return 0.0

        soup = BeautifulSoup(page.content, "html.parser")
        
        # Find headlines
        headline_tags = soup.find_all('h2', limit=max_headlines)
        headlines = [tag.get_text(strip=True) for tag in headline_tags]

        if not headlines:
            print("No headlines found.")
            return 0.0

        # Analyze sentiment
        analysis = sentiment_pipeline(headlines)
        
        # Convert sentiment to a numerical score
        # FinBERT labels: 'positive', 'negative', 'neutral'
        scores = []
        for result in analysis:
            if result['label'] == 'positive':
                scores.append(result['score'])
            elif result['label'] == 'negative':
                scores.append(-result['score'])
            else: # neutral
                scores.append(0.0)
        
        if not scores:
            return 0.0
            
        # Return the average sentiment
        return np.mean(scores)

    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return 0.0

if __name__ == '__main__':
    # Test the function
    print("Testing market sentiment analysis...")
    average_sentiment = get_market_sentiment()
    print(f"---------------------------------")
    print(f"Average Market Sentiment: {average_sentiment:.4f}")
    print("---------------------------------")
