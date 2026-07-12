
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

        bloque_titulares = "\n- ".join(news_titles)
        prompt = f"""
        You are an expert financial analyst. Your task is to evaluate the overall daily market sentiment based on the following list of news headlines.

        Strict rule: You must respond ONLY with a single float number between -1.0 and 1.0. Do not add any extra text, explanations, or punctuation.
        -1.0 = Extremely Negative (panic, potential price drops)
        0.0 = Neutral (informative, mixed signals, no clear market impact)
        1.0 = Extremely Positive (euphoria, potential price increases)

        Headlines for today:
        - {bloque_titulares}

        Response:
        """

        answer = model.generate_content(prompt)
        final_score = float(answer.text.strip())
        
        return final_score
    
    except Exception as e:
        print(f"Error on analysis of sentiment: {e}")
        return 0.0
    

def merge_data(financial_filt_data, sentiment_score):
    try:
        financial_filt_data["Sentiment"] = sentiment_score
        
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



    
