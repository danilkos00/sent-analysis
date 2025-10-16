import pandas as pd
import re

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Removing nans and text cleaning"""
    processed_data = data.dropna().copy()

    processed_data['text'] = processed_data['text'].apply(_clear_text)

    return processed_data
    

def _clear_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|#[\w-]+|http\S+|\n', '', text)

    text = re.sub(r'[^\w\s]', ' ', text)
    return text
