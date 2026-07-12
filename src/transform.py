
import pandas as pd
import extract
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GEMINI_API"))


def clean_financial_data(filt_data):
    try: 
        filt_data["Date"] = pd.to_datetime(filt_data["Date"])
        filt_data_clean = filt_data.dropna()
        return filt_data_clean

    except Exception as e:
        print(f"Error cleaning data: {e}")
        return pd.DataFrame()
    
def analyze_sentiment(news_data):
    try:
        model = genai.GenerativeModel("gemini-3.5-flash")

        news_lst = news_data["articles"]
        news_titles = []
        for i in news_lst:
            news_titles.append(i["title"])

        scores = []
        for title in news_titles:
            prompt = f"""
                You are an expert financial analyst. Your task is to classify the market sentiment of the following stock market news headline.

                Strict rule: You must respond ONLY with a single integer. Do not add any extra text, explanations, or punctuation.
                1 = Positive (indicates a potential price increase or good news)
                0 = Neutral (informative, no clear market impact)
                -1 = Negative (indicates a potential price decrease or bad news)

                Examples:
                Headline: "Apple reports record third-quarter profits"
                Response: 1

                Headline: "Stock market closes flat ahead of inflation data"
                Response: 0

                Headline: "Shares tumble 5% following antitrust lawsuit"
                Response: -1

                Headline: "{title}"
                Response:
                """
            answer = model.generate_content(prompt)

            score = int(answer.text.strip())

            scores.append(score)

        return scores
    
    except Exception as e:
        print(f"Error on analysis of sentiment: {e}")
        return []
    

def merge_data(financial_filt_data, sentiment_scores):
    try:
        if not sentiment_scores:
            financial_filt_data["Sentiment"] = 0
        else:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            financial_filt_data["Sentiment"] = avg_sentiment
        return financial_filt_data
    
    except Exception as e:
        print(f"Error merging data: {e}")
        return financial_filt_data
    
if __name__ == "__main__":
    # Prueba de integración de la Fase 2
    print("--- Probando Pipeline de Transformación ---")
    
    # 1. Simulación de datos (Mocking)
    mock_financial = pd.DataFrame({'Date': ['2026-07-12'], 'Close': [150.0], 'Volume': [1000]})
    mock_news = {"articles": [{"title": "Apple hits record profits"}, {"title": "Market flat"}]}
    
    # 2. Ejecución
    clean_df = clean_financial_data(mock_financial)
    scores = analyze_sentiment(mock_news)
    final_df = merge_data(clean_df, scores)
    
    print("DataFrame Final Resultante:")
    print(final_df)



    
