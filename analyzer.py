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
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
import numpy as np

nltk.download('stopwords')

# Add configuration
@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figure_sizes: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'default': (10, 6),
        'pie': (8, 8),
        'heatmap': (12, 6)
    })
    colors: List[str] = field(default_factory=lambda: [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'
    ])
    output_dir: Path = Path('static/visuals')
    dpi: int = 300

# Add constants
EMOJI_PATTERN = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]'
URL_PATTERN = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
DAY_START_HOUR = 6
EXCLUDED_MESSAGES = [
    'joined', 'left', 'removed', 'changed', 'image omitted', 
    'video omitted', 'video call', 'voice call', 'audio omitted',
    'missed voice', 'missed video', 'google jr'
]

# Add logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_output_directory(config: VisualizationConfig) -> None:
    """Create output directory for visualizations if it doesn't exist."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

def read_chat_log(csv_file_path: str) -> pd.DataFrame:
    """
    Read and validate the chat log from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file containing chat data
        
    Returns:
        DataFrame containing the chat data
        
    Raises:
        ValueError: If file is not found or contains no data
    """
    try:
        file_path = Path(csv_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {csv_file_path}")
            
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("No data found in the file.")
        
        # Strip any leading/trailing spaces from the column headers and date
        df.columns = df.columns.str.strip()
        if 'Date' in df.columns:
            df['Date'] = df['Date'].str.strip()
        
        required_columns = {'Date', 'Time', 'Sender', 'Message'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading chat log: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the chat log data with improved handling of date/time.
    
    Args:
        df: Raw DataFrame containing chat data
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Convert Date and Time to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
        
        # Filter out system messages and media
        pattern = '|'.join(EXCLUDED_MESSAGES)
        df = df[~df['Message'].astype(str).str.contains(pattern, case=False)]
        
        # Clean up missing values
        df = df.dropna(subset=['Date', 'Time', 'Message'])
        df.loc[:, 'Message'] = df['Message'].astype(str)
        
        # Add DateTime column for easier analysis
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df['Time'].astype(str)
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

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

def generate_visualizations(df: pd.DataFrame, config: VisualizationConfig) -> Tuple[Path, ...]:
    """Generate visualizations from the chat log."""
    # Create wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=STOPWORDS, 
        max_words=200
    ).generate(' '.join(df['Message']))
    
    wordcloud_path = config.output_dir / 'wordcloud.png'
    wordcloud.to_file(str(wordcloud_path))  # Convert Path to string
    
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
    bar_graph_image_path = config.output_dir / 'first_message_sender.png'
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
    love_graph_image_path = config.output_dir / 'love_counts.png'
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
    avg_messages_graph_image_path = config.output_dir / 'average_messages_per_day.png'
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
    sentiment_counts_graph_path = config.output_dir / 'sentiment_counts.png'
    os.makedirs(os.path.dirname(sentiment_counts_graph_path), exist_ok=True)
    plt.savefig(sentiment_counts_graph_path)
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
    links_pie_chart_path = config.output_dir / 'links_shared_by_sender.png'
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
    top_emojis_graph_path = config.output_dir / 'top_emojis.png'
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
    total_messages_graph_path = config.output_dir / 'total_messages_count.png'
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
    messages_count_graph_path = config.output_dir / 'messages_count_per_sender.png'
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
    message_activity_graph_path = config.output_dir / 'message_activity_over_time.png'
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
    message_activity_heatmap_path = config.output_dir / 'message_activity_heatmap.png'
    os.makedirs(os.path.dirname(message_activity_heatmap_path), exist_ok=True)
    plt.savefig(message_activity_heatmap_path)
    plt.close()

    return (
        wordcloud_path,
        bar_graph_image_path,
        love_graph_image_path,
        avg_messages_graph_image_path,
        sentiment_counts_graph_path,
        links_pie_chart_path,
        top_emojis_graph_path,
        messages_count_graph_path,
        message_activity_graph_path,
        message_activity_heatmap_path
    )

def analyze_chat_log(csv_file_path: str) -> dict:
    """Analyze chats into various statistics"""
    # Create a config instance and set up directories
    config = VisualizationConfig()
    setup_output_directory(config)
    
    df = read_chat_log(csv_file_path)
    df = preprocess_data(df)
    
    participant1_message_count, participant2_message_count = analyze_participants(df)
    analysis_results = perform_analysis(df)
    
    # Pass config to generate_visualizations
    visualization_paths = generate_visualizations(df, config)
    
    results = {
        "total_messages": len(df),
        "unique_senders": df['Sender'].nunique(),
        "participant1_message_count": participant1_message_count,
        "participant2_message_count": participant2_message_count,
        **analysis_results,
        "wordcloud": visualization_paths[0],
        "first_message_sender": visualization_paths[1],
        "love_counts": visualization_paths[2],
        "average_messages_per_day": visualization_paths[3],
        "sentiment_counts": visualization_paths[4],
        "links_pie_chart": visualization_paths[5],
        "top_emojis": visualization_paths[6],
        "messages_count_per_sender": visualization_paths[7],
        "message_activity_over_time": visualization_paths[8],
        "message_activity_heatmap": visualization_paths[9]
    }

    return results

def save_visualization(fig: plt.Figure, filename: str, config: VisualizationConfig) -> Path:
    """Save visualization to file and return the path."""
    output_path = config.output_dir / filename
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path

def create_message_activity_heatmap(df: pd.DataFrame, config: VisualizationConfig) -> Path:
    """Create heatmap of message activity by hour and day of week."""
    try:
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        
        heatmap_data = df.pivot_table(
            index='DayOfWeek', 
            columns='Hour',
            values='Message',
            aggfunc='count',
            fill_value=0
        )
        
        ordered_days = [
            'Monday', 'Tuesday', 'Wednesday', 
            'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]
        heatmap_data = heatmap_data.reindex(ordered_days)
        
        fig, ax = plt.subplots(figsize=config.figure_sizes['heatmap'])
        sns.heatmap(
            heatmap_data,
            cmap='YlGnBu',
            annot=True,
            fmt='d',
            ax=ax
        )
        
        plt.title('Message Activity by Hour and Day of the Week')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Day of the Week')
        
        return save_visualization(fig, 'message_activity_heatmap.png', config)
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        raise
