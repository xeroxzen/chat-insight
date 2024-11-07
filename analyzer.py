import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import re
import os
from textblob import TextBlob

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
    df = df.dropna(subset=['Date', 'Time'])
    df.loc[:, 'Message'] = df['Message'].astype(str)
    df = df.dropna(subset=['Message'])
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

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment of messages in the chat log."""
    df['Sentiment'] = df['Message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    return df

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
    
    # Analyze sentiment
    df = analyze_sentiment(df)
    sentiment_counts = df['Sentiment_Label'].value_counts().to_dict()
    
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
        "sentiment_counts": sentiment_counts,
    }

def generate_visualizations(df: pd.DataFrame) -> str:
    """Generate visualizations from the chat log."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, max_words=200).generate(' '.join(df['Message']))
    
    if not os.path.exists('visuals'):
        os.makedirs('visuals')
    wordcloud_image_path = 'static/visuals/wordcloud.png'
    wordcloud.to_file(wordcloud_image_path)
    
    """
    Who sends the first message of the day more often.
    
    A new day starts at 6am. So first step is to combine Date and Time into a single datetime column, then filter the date part for grouping, and proceed to find the first message
    """
    # Filter messages starting from 6 AM and find the first message of each day
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df_filtered = df[df['Time'] >= pd.to_datetime('06:00:00').time()]
    first_messages = df_filtered.groupby('Date').first()
    first_sender_counts = first_messages['Sender'].value_counts()

    # Bar chart
    plt.figure(figsize=(10, 6))
    first_sender_counts.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
    plt.title('First Message of the Day by Sender (After 6 AM)')
    plt.xlabel('Sender')
    plt.ylabel('Number of First Messages')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # Saved figure
    bar_graph_image_path = 'static/visuals/first_message_sender.png'
    os.makedirs(os.path.dirname(bar_graph_image_path), exist_ok=True)
    plt.savefig(bar_graph_image_path)
    plt.close()
    
    # Who says I love you more
    love_counts = df[df['Message'].str.contains('i love you', case=False)]['Sender'].value_counts()
    
    # Create a bar graph
    plt.figure(figsize=(10, 6))
    love_counts.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
    plt.title('Number of Times "I Love You" Sent by Sender')
    plt.xlabel('Sender')
    plt.ylabel('Number of Times "I Love You" Sent')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save the figure
    love_graph_image_path = 'static/visuals/love_counts.png'
    os.makedirs(os.path.dirname(love_graph_image_path), exist_ok=True)
    plt.savefig(love_graph_image_path)
    plt.close()
    
    # Average messages per day by sender
    avg_messages_per_day_by_sender = df.groupby('Sender')['Date'].value_counts().groupby('Sender').mean()
    
    plt.figure(figsize=(10, 6))
    avg_messages_per_day_by_sender.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
    plt.title('Average Messages per Day by Sender')
    plt.xlabel('Sender')
    plt.ylabel('Average Number of Messages per Day')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    avg_messages_graph_image_path = 'static/visuals/average_messages_per_day.png'
    os.makedirs(os.path.dirname(avg_messages_graph_image_path), exist_ok=True)
    plt.savefig(avg_messages_graph_image_path)
    plt.close()
    
    # Sentiment counts
    sentiment_counts = df['Sentiment_Label'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Messages')
    plt.tight_layout()
    
    # Save the figure
    sentiment_counts_graph_image_path = 'static/visuals/sentiment_counts.png'
    os.makedirs(os.path.dirname(sentiment_counts_graph_image_path), exist_ok=True)
    plt.savefig(sentiment_counts_graph_image_path)
    plt.close()

    # Calculate the number of links shared by each sender
    links_shared = df[df['Message'].str.contains('http[s]?://')]['Sender'].value_counts()

    # Create a pie chart for links shared
    plt.figure(figsize=(8, 8))  # Adjust size for pie chart
    plt.pie(links_shared, labels=links_shared.index, autopct='%1.1f%%', startangle=90)
    plt.title('Links Shared by Each Sender')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()

    # Save the figure
    links_pie_chart_path = 'static/visuals/links_shared_by_sender.png'
    os.makedirs(os.path.dirname(links_pie_chart_path), exist_ok=True)
    plt.savefig(links_pie_chart_path)
    plt.close()

    # Count occurrences of each emoji in the messages
    emoji_counts = df['Message'].str.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]').explode().value_counts()

    # Get the top 10 emojis
    top_10_emojis = emoji_counts.head(10)

    # Create a bar chart for the top 10 emojis
    plt.figure(figsize=(10, 6))
    top_10_emojis.plot(kind='bar', color=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'black'])
    plt.title('Top 10 Emojis Used')
    plt.xlabel('Emojis')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    top_emojis_graph_path = 'static/visuals/top_emojis.png'
    os.makedirs(os.path.dirname(top_emojis_graph_path), exist_ok=True)
    plt.savefig(top_emojis_graph_path)
    plt.close()

    # Count total messages
    total_messages_count = len(df)

    # Create a bar chart for total messages count
    plt.figure(figsize=(8, 6))
    plt.bar(['Total Messages'], [total_messages_count], color='lightblue')
    plt.title('Total Messages Count')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    total_messages_graph_path = 'static/visuals/total_messages_count.png'
    os.makedirs(os.path.dirname(total_messages_graph_path), exist_ok=True)
    plt.savefig(total_messages_graph_path)
    plt.close()

    # Count the number of messages per sender
    messages_count_per_sender = df['Sender'].value_counts()

    # Create a bar chart for messages count per sender
    plt.figure(figsize=(10, 6))
    messages_count_per_sender.plot(kind='bar', color=['lightblue', 'yellow'])
    plt.title('Total Messages Count per Sender')
    plt.xlabel('Sender')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    messages_count_graph_path = 'static/visuals/messages_count_per_sender.png'
    os.makedirs(os.path.dirname(messages_count_graph_path), exist_ok=True)
    plt.savefig(messages_count_graph_path)
    plt.close()
    
    """Line Chart of Message Activity Over Time"""
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    
    # Group by date and count messages
    message_activity = df.groupby('Date')['Message'].count()
    
    plt.figure(figsize=(10, 6))
    message_activity.plot(kind='line', marker='o', color='skyblue')
    plt.title('Message Activity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    message_activity_graph_path = 'static/visuals/message_activity_over_time.png'
    os.makedirs(os.path.dirname(message_activity_graph_path), exist_ok=True)
    plt.savefig(message_activity_graph_path)
    plt.close()
    
    """Heatmap to Display Message Activity by Hour and Day of the Week"""
    
    # Step 1: Combine Date and Time into a single datetime column
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Step 2: Extract hour and day of the week
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.day_name()

    # Step 3: Create a pivot table
    heatmap_data = df.pivot_table(index='DayOfWeek', columns='Hour', values='Message', aggfunc='count', fill_value=0)

    # Reorder days of the week
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(ordered_days)

    # Step 4: Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
    plt.title('Message Activity by Hour and Day of the Week')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Day of the Week')

    # Save the figure
    message_activity_heatmap_path = 'static/visuals/message_activity_heatmap.png'
    os.makedirs(os.path.dirname(message_activity_heatmap_path), exist_ok=True)
    plt.savefig(message_activity_heatmap_path)
    plt.close()

    return wordcloud_image_path, bar_graph_image_path, love_graph_image_path, avg_messages_graph_image_path, sentiment_counts_graph_image_path, links_pie_chart_path, top_emojis_graph_path, messages_count_graph_path, message_activity_graph_path, message_activity_heatmap_path

def analyze_chat_log(csv_file_path: str) -> dict:
    """Analyze chats into various statistics"""
    df = read_chat_log(csv_file_path)
    df = preprocess_data(df)
    
    participant1_message_count, participant2_message_count = analyze_participants(df)
    analysis_results = perform_analysis(df)
    wordcloud_image_path, bar_graph_image_path, love_graph_image_path, avg_messages_graph_image_path, sentiment_counts_graph_image_path, links_pie_chart_path, top_emojis_graph_path, messages_count_graph_path, message_activity_graph_path, message_activity_heatmap_path = generate_visualizations(df)

    results = {
        "total_messages": len(df),
        "unique_senders": df['Sender'].nunique(),
        "participant1_message_count": participant1_message_count,
        "participant2_message_count": participant2_message_count,
        **analysis_results,
        "wordcloud": wordcloud_image_path,
        "first_message_sender": bar_graph_image_path,
        "love_counts": love_graph_image_path,
        "average_messages_per_day": avg_messages_graph_image_path,
        "sentiment_counts": sentiment_counts_graph_image_path,
        "links_pie_chart": links_pie_chart_path,
        "top_emojis": top_emojis_graph_path,
        "messages_count_per_sender": messages_count_graph_path,
        "message_activity_over_time": message_activity_graph_path,
        "message_activity_heatmap": message_activity_heatmap_path
    }

    return results
