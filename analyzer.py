import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import re
import os

nltk.download('stopwords')

def read_chat_log(csv_file_path: str) -> pd.DataFrame:
    """Read the chat log from a CSV file."""
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {csv_file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("No data found in the file.")
    
    # Strip any leading/trailing spaces from the column headers
    df.columns = df.columns.str.strip()
    df['Date'] = df['Date'].str.strip()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the chat log data."""
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
    df = df[~df['Message'].astype(str).str.contains('joined|left|removed|changed|image omitted|video omitted|video call|voice call|audio omitted|missed voice|missed video|google jr', case=False)]
    df.dropna(subset=['Date', 'Time'], inplace=True)
    df['Message'] = df['Message'].astype(str)
    df.dropna(subset=['Message'], inplace=True)
    return df

def analyze_participants(df: pd.DataFrame) -> tuple:
    """Analyze participants and their message counts."""
    participants = df['Sender'].unique()
    if len(participants) < 2:
        raise ValueError("Not enough participants found in the chat log.")
    
    participant1, participant2 = participants[:2]
    participant1_message_count = len(df[df['Sender'] == participant1])
    participant2_message_count = len(df[df['Sender'] == participant2])
    
    return participant1_message_count, participant2_message_count

def preprocess_message(message: str) -> str:
    """Preprocess a message by removing non-word characters and converting to lowercase."""
    return re.sub(r'[^\w\s]', '', message).lower()

def perform_analysis(df: pd.DataFrame) -> dict:
    """Perform various analyses on the chat log."""
    most_active_day = df['Date'].value_counts().idxmax()
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    most_active_time = df['Hour'].value_counts().idxmax()
    top_participants = df['Sender'].value_counts().nlargest(2)
    
    # Average number of messages sent per day
    avg_messages_per_day = df['Date'].value_counts().mean()
    
    # Top emojis and links
    emojis = df['Message'].str.extractall(r'([\U0001f600-\U0001f650])')[0].value_counts().head(10)
    links = df['Message'].str.extractall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')[0].value_counts().head(10)
    
    # Commonly used words
    stop_words = set(stopwords.words('english'))
    all_messages = ' '.join(df['Message'].apply(preprocess_message))
    words = [word for word in all_messages.split() if word not in stop_words]
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(20)
    
    return {
        "most_active_day": most_active_day.date(),
        "most_active_time": f"{most_active_time}:00",
        "most_common_words": most_common_words,
        "top_participants": top_participants.to_dict(),
        "first_message_date": df['Date'].min().date(),
        "last_message_date": df['Date'].max().date(),
        "avg_messages_per_day": avg_messages_per_day,
        "emojis": emojis.to_dict(),
        "links": links.to_dict(),
    }

def generate_visualizations(df: pd.DataFrame) -> str:
    """Generate visualizations from the chat log."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, max_words=200).generate(' '.join(df['Message']))
    
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
    wordcloud_image_path = f'static/visuals/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)
    
    return wordcloud_image_path

def analyze_chat_log(csv_file_path: str) -> dict:
    """Analyze chats into various statistics"""
    df = read_chat_log(csv_file_path)
    df = preprocess_data(df)
    
    participant1_message_count, participant2_message_count = analyze_participants(df)
    analysis_results = perform_analysis(df)
    wordcloud_image_path = generate_visualizations(df)

    results = {
        "total_messages": len(df),
        "unique_senders": df['Sender'].nunique(),
        "participant1_message_count": participant1_message_count,
        "participant2_message_count": participant2_message_count,
        **analysis_results,
        "wordcloud": wordcloud_image_path
    }

    return results
