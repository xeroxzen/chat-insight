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

def analyze_chat_log(csv_file_path: str) -> dict:
    """Analyze chats into various statistics"""
    
    # Error handling for file reading
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {csv_file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("No data found in the file.")
    
    # Strip any leading/trailing spaces from the column headers
    df.columns = df.columns.str.strip()

    # Stripping spaces from the Date column
    df['Date'] = df['Date'].str.strip()

    # Converting 'Date' and 'Time' columns to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
    
    # Remove words not part of the chat e.g. 'joined', 'left', 'removed', 'changed', 'image omitted', 'video omitted'
    df = df[~df['Message'].astype(str).str.contains('joined|left|removed|changed|image omitted|video omitted|video call|voice call|audio omitted|missed voice|missed video|google jr', case=False)]

    # Dropping rows with NaT in 'Date' or NaT in 'Time'
    df.dropna(subset=['Date', 'Time'], inplace=True)
    
    # Converting 'Message' column to string and removing NaN values
    df['Message'] = df['Message'].astype(str)
    df.dropna(subset=['Message'], inplace=True)

    # Ensure there are at least two participants
    participants = df['Sender'].unique()
    if len(participants) < 2:
        raise ValueError("Not enough participants found in the chat log.")

    participant1, participant2 = participants[:2]

    # Message counts
    participant1_message_count = len(df[df['Sender'] == participant1])
    participant2_message_count = len(df[df['Sender'] == participant2])

    # Day with the most activity
    most_active_day = df['Date'].value_counts().idxmax()

    # Most active time during the day
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    most_active_time = df['Hour'].value_counts().idxmax()

    # Commonly used words
    stop_words = set(stopwords.words('english'))
    def preprocess_message(message: str) -> str:
        return re.sub(r'[^\w\s]', '', message).lower()

    # Top emojis and links
    emojis = df['Message'].str.extractall(r'([\U0001f600-\U0001f650])')[0].value_counts().head(5)
    links = df['Message'].str.extractall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)')[0].value_counts().head(5)

    # Tokenizing and counting words
    all_messages = ' '.join(df['Message'].apply(preprocess_message))
    words = [word for word in all_messages.split() if word not in stop_words]
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(20)
    
    # Generate a Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, max_words=200).generate(all_messages)
    
    # Save the Word Cloud to a file(found in a folder named visuals)
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
    # random_number = random.randint(1, 1000)
    wordcloud_image_path = f'static/visuals/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)

    # Getting the top participants
    top_participants = df['Sender'].value_counts().nlargest(2)

    # Average number of messages sent per day
    avg_messages_per_day = df['Date'].value_counts().mean()

    results = {
        "total_messages": len(df),
        "unique_senders": df['Sender'].nunique(),
        "participant1_message_count": participant1_message_count,
        "participant2_message_count": participant2_message_count,
        "most_active_day": most_active_day.date(),
        "most_active_time": f"{most_active_time}:00",
        "most_common_words": most_common_words,
        "top_participants": top_participants.to_dict(),
        "first_message_date": df['Date'].min().date(),
        "last_message_date": df['Date'].max().date(),
        "emojis": emojis.to_dict(),
        "links": links.to_dict(),
        "avg_messages_per_day": avg_messages_per_day,
        
        # Visuals go here
        'wordcloud': wordcloud_image_path
    }

    return results
