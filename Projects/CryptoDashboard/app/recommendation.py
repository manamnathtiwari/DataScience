# app/recommendation.py

import os
from config import GEMINI_API_KEY
import openai  # assuming we're using OpenAI-style Gemini interface
import pandas as pd

openai.api_key = GEMINI_API_KEY

def format_data_for_gemini(df: pd.DataFrame, coin: str) -> str:
    # Convert DataFrame into readable text
    formatted = df[['timestamp', 'price', 'SMA_5', 'EMA_5', 'Volatility', 'Return']].fillna("N/A").to_string(index=False)
    
    prompt = f"""
You are a crypto trading expert. Analyze the following technical indicators for {coin} and give a recommendation (Buy / Sell / Hold).
Be precise and explain briefly.

Crypto Analysis:
{formatted}

Recommendation:
"""
    return prompt

def get_recommendation(df: pd.DataFrame, coin: str = "bitcoin") -> str:
    prompt = format_data_for_gemini(df, coin)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or gemini model ID if different
            messages=[
                {"role": "system", "content": "You are a crypto investment advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        recommendation = response['choices'][0]['message']['content']
        return recommendation.strip()

    except Exception as e:
        return f"Error from Gemini: {str(e)}"
