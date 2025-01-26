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
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from datetime import datetime

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
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S').dt.time
        
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
    
    # Add new analyses
    # 1. Response time analysis
    df_sorted = df.sort_values('DateTime')
    df_sorted['TimeDiff'] = df_sorted['DateTime'].diff()
    avg_response_time = df_sorted.groupby('Sender')['TimeDiff'].mean()
    
    # 2. Most active hours for each person
    active_hours_by_sender = df.groupby(['Sender', df['DateTime'].dt.hour])['Message'].count().unstack()
    peak_hours = {sender: active_hours_by_sender.loc[sender].idxmax() for sender in active_hours_by_sender.index}
    
    # 3. Longest message streak analysis
    df_sorted['TimeDiff'] = df_sorted['DateTime'].diff().dt.total_seconds()
    streak_threshold = 300  # 5 minutes
    df_sorted['NewStreak'] = df_sorted['TimeDiff'] > streak_threshold
    df_sorted['StreakId'] = df_sorted['NewStreak'].cumsum()
    streak_counts = df_sorted.groupby('StreakId').size()
    longest_streak = streak_counts.max()
    
    # 4. Question analysis
    question_patterns = r'\b(who|what|when|where|why|how)\b|\?'
    df['IsQuestion'] = df['Message'].str.contains(question_patterns, case=False)
    questions_by_sender = df[df['IsQuestion']]['Sender'].value_counts()
    
    # 5. Message length analysis
    df['MessageLength'] = df['Message'].str.len()
    avg_message_length = df.groupby('Sender')['MessageLength'].mean()
    
    # Update return dictionary
    analysis_results = {
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
        "average_response_times": avg_response_time.to_dict(),
        "peak_activity_hours": peak_hours,
        "longest_message_streak": longest_streak,
        "questions_asked": questions_by_sender.to_dict(),
        "average_message_length": avg_message_length.to_dict()
    }
    
    return analysis_results

def generate_visualizations(df: pd.DataFrame, config: VisualizationConfig, chat_type='friends') -> Dict[str, Path]:
    """Generate visualizations from the chat log."""
    visualizations = {}
    
    # Create and save wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=STOPWORDS, 
        max_words=200
    ).generate(' '.join(df['Message']))
    
    visualizations['wordcloud'] = save_visualization(wordcloud, 'wordcloud.png', config)
    
    # First message sender analysis
    df_filtered = df[df['Time'] >= pd.to_datetime('06:00:00').time()]
    first_messages = df_filtered.groupby('Date').first()
    first_sender_counts = first_messages['Sender'].value_counts()
    
    plt.figure(figsize=(10, 6))
    first_sender_counts.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
    plt.title('First Message of the Day by Sender (After 6 AM)')
    plt.xlabel('Sender')
    plt.ylabel('Number of First Messages')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    visualizations['first_message_sender'] = save_visualization(plt.gcf(), 'first_message_sender.png', config)
    plt.close()
    
    # Love counts visualization with chat type handling
    if chat_type.lower() == 'romantic':
        try:
            love_patterns = ['❤️', '💕', '💗', '💓', '💖']
            love_counts = pd.Series({
                pattern: df['Message'].str.count(pattern).sum() 
                for pattern in love_patterns
            })
            
            if not love_counts.empty and love_counts.sum() > 0:
                plt.figure(figsize=config.figure_sizes['default'])
                love_counts.plot(kind='bar', color=['red', 'pink', 'magenta', 'crimson', 'deeppink'])
                plt.title('Love Emoji Usage')
                plt.xlabel('Emoji Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                love_counts_path = config.output_dir / 'love_counts.png'
                plt.savefig(love_counts_path, bbox_inches='tight')
                plt.close()
            else:
                plt.figure(figsize=config.figure_sizes['default'])
                plt.text(0.5, 0.5, 'No love emojis found in the chat', 
                        horizontalalignment='center',
                        verticalalignment='center')
                plt.axis('off')
                love_counts_path = config.output_dir / 'love_counts.png'
                plt.savefig(love_counts_path, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error generating love counts visualization: {e}")
            plt.figure(figsize=config.figure_sizes['default'])
            plt.text(0.5, 0.5, 'Unable to generate love emoji visualization', 
                    horizontalalignment='center',
                    verticalalignment='center')
            plt.axis('off')
            love_counts_path = config.output_dir / 'love_counts.png'
            plt.savefig(love_counts_path, bbox_inches='tight')
            plt.close()
    else:
        # For non-romantic chats, generate a friendship emoji analysis instead
        try:
            friendship_patterns = ['😊', '😄', '👍', '🤝', '🙌', '✌️', '🎉', '🤗']
            friendship_counts = pd.Series({
                pattern: df['Message'].str.count(pattern).sum() 
                for pattern in friendship_patterns
            })
            
            if not friendship_counts.empty and friendship_counts.sum() > 0:
                plt.figure(figsize=config.figure_sizes['default'])
                friendship_counts.plot(kind='bar', color=['gold', 'orange', 'dodgerblue', 'mediumseagreen', 
                                                        'cornflowerblue', 'mediumpurple', 'coral', 'lightseagreen'])
                plt.title('Friendly Emoji Usage')
                plt.xlabel('Emoji Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                love_counts_path = config.output_dir / 'friendship_emoji_counts.png'
                plt.savefig(love_counts_path, bbox_inches='tight')
                plt.close()
            else:
                plt.figure(figsize=config.figure_sizes['default'])
                plt.text(0.5, 0.5, 'No friendly emojis found in the chat', 
                        horizontalalignment='center',
                        verticalalignment='center')
                plt.axis('off')
                love_counts_path = config.output_dir / 'friendship_emoji_counts.png'
                plt.savefig(love_counts_path, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error generating friendship emoji visualization: {e}")
            plt.figure(figsize=config.figure_sizes['default'])
            plt.text(0.5, 0.5, 'Unable to generate friendship emoji visualization', 
                    horizontalalignment='center',
                    verticalalignment='center')
            plt.axis('off')
            love_counts_path = config.output_dir / 'friendship_emoji_counts.png'
            plt.savefig(love_counts_path, bbox_inches='tight')
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

    # New visualizations
    
    # 1. Message Length Distribution
    plt.figure(figsize=config.figure_sizes['default'])
    sns.boxplot(x='Sender', y='MessageLength', data=df)
    plt.title('Message Length Distribution by Sender')
    plt.ylabel('Message Length (characters)')
    message_length_path = config.output_dir / 'message_length_distribution.png'
    plt.savefig(message_length_path)
    plt.close()
    
    # 2. Daily Conversation Pattern
    daily_pattern = df.groupby([df['DateTime'].dt.hour, 'Sender'])['Message'].count().unstack()
    plt.figure(figsize=config.figure_sizes['default'])
    daily_pattern.plot(kind='line', marker='o')
    plt.title('Daily Conversation Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Messages')
    plt.legend(title='Sender')
    daily_pattern_path = config.output_dir / 'daily_pattern.png'
    plt.savefig(daily_pattern_path)
    plt.close()
    
    # 3. Response Time Distribution
    plt.figure(figsize=config.figure_sizes['default'])
    df_sorted = df.sort_values('DateTime')
    response_times = df_sorted['DateTime'].diff().dt.total_seconds() / 60  # Convert to minutes
    sns.histplot(data=response_times[response_times < 60], bins=30)  # Show responses within 1 hour
    plt.title('Response Time Distribution (within 1 hour)')
    plt.xlabel('Response Time (minutes)')
    response_time_path = config.output_dir / 'response_time_distribution.png'
    plt.savefig(response_time_path)
    plt.close()

    return visualizations

@dataclass
class ChatAnalysis:
    """Container for chat analysis results"""
    is_group: bool
    participant_count: int
    total_messages: int
    analysis_period: Tuple[str, str]  # (start_date, end_date)
    participant_stats: Dict[str, Dict[str, Any]]
    message_patterns: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    visualization_paths: Dict[str, Path]
    top_participants: Dict[str, int]
    most_active_day: datetime
    emojis: Dict[str, int]
    links: Dict[str, int]
    first_message_date: datetime
    last_message_date: datetime
    avg_messages_per_day: float
    
    # Group-specific metrics
    group_dynamics: Optional[Dict[str, Any]] = None
    interaction_network: Optional[Dict[str, Any]] = None
    
    # Individual-specific metrics
    conversation_balance: Optional[Dict[str, Any]] = None
    response_patterns: Optional[Dict[str, Any]] = None

def determine_chat_type(df: pd.DataFrame) -> bool:
    """
    Determine if the chat is a group chat or individual chat.
    
    Returns:
        bool: True if group chat, False if individual chat
    """
    unique_senders = df['Sender'].nunique()
    return unique_senders > 2

def analyze_group_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze group-specific metrics."""
    # Get top 15 participants by message count
    message_counts = df['Sender'].value_counts()
    top_15_participants = message_counts.head(15)
    
    # Message distribution for top 15
    most_active_member = top_15_participants.index[0]
    least_active_member = top_15_participants.index[-1]
    
    # Interaction patterns for top 15
    df_filtered = df[df['Sender'].isin(top_15_participants.index)]
    df_filtered['Hour'] = df_filtered['DateTime'].dt.hour
    peak_hours = df_filtered.groupby(['Sender', 'Hour'])['Message'].count().groupby('Sender').idxmax()
    
    # Create interaction network data for top 15
    df_sorted = df_filtered.sort_values('DateTime')
    df_sorted['Next_Sender'] = df_sorted['Sender'].shift(-1)
    interactions = df_sorted.groupby(['Sender', 'Next_Sender']).size().reset_index(name='weight')
    
    return {
        "participant_count": len(message_counts),  # Keep total count
        "top_15_count": len(top_15_participants),
        "most_active_member": most_active_member,
        "least_active_member": least_active_member,
        "peak_hours": peak_hours.to_dict(),
        "interaction_data": interactions.to_dict('records')
    }

def analyze_individual_chat(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze metrics specific to individual chats."""
    participants = df['Sender'].unique()
    if len(participants) != 2:
        raise ValueError("Individual chat analysis requires exactly 2 participants")
    
    # Message balance
    message_counts = df['Sender'].value_counts()
    total_messages = len(df)
    message_ratio = f"{message_counts[0]}/{message_counts[1]}"
    
    # Response patterns
    df_sorted = df.sort_values('DateTime')
    df_sorted['TimeDiff'] = df_sorted['DateTime'].diff()
    avg_response_times = df_sorted.groupby('Sender')['TimeDiff'].mean()
    
    return {
        "participants": participants.tolist(),
        "message_ratio": message_ratio,
        "message_balance_percentage": (message_counts[0] / total_messages) * 100,
        "avg_response_times": avg_response_times.to_dict()
    }

def generate_group_visualizations(df: pd.DataFrame, config: VisualizationConfig) -> Dict[str, Path]:
    """Generate visualizations specific to group chats."""
    # Get top 15 participants
    top_15_senders = df['Sender'].value_counts().head(15)
    
    # Participation distribution for top 15
    plt.figure(figsize=config.figure_sizes['default'])
    sns.barplot(x=top_15_senders.index, y=top_15_senders.values)
    plt.title('Top 15 Group Participation Distribution')
    plt.xticks(rotation=45, ha='right')
    participation_path = save_visualization(plt.gcf(), 'group_participation.png', config)
    
    # Interaction network for top 15
    plt.figure(figsize=config.figure_sizes['default'])
    df_filtered = df[df['Sender'].isin(top_15_senders.index)]
    G = nx.from_pandas_edgelist(
        df_filtered.sort_values('DateTime').assign(Next_Sender=lambda x: x['Sender'].shift(-1))
        .groupby(['Sender', 'Next_Sender']).size().reset_index(name='weight'),
        'Sender', 'Next_Sender', 'weight'
    )
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8)
    network_path = save_visualization(plt.gcf(), 'group_interaction_network.png', config)
    
    return {
        "participation_distribution": participation_path,
        "interaction_network": network_path
    }

def generate_individual_visualizations(df: pd.DataFrame, config: VisualizationConfig) -> Dict[str, Path]:
    """Generate visualizations specific to individual chats."""
    # Conversation flow
    plt.figure(figsize=config.figure_sizes['default'])
    timeline = df.set_index('DateTime')['Sender'].astype('category').cat.codes
    plt.plot(timeline.index, timeline.values, 'o-')
    plt.title('Conversation Flow')
    plt.ylabel('Participant')
    plt.yticks([0, 1], df['Sender'].unique())
    flow_path = save_visualization(plt.gcf(), 'conversation_flow.png', config)
    
    return {
        "conversation_flow": flow_path
    }

def analyze_chat_log(csv_file_path: str) -> ChatAnalysis:
    """Main function to analyze chat logs."""
    config = VisualizationConfig()
    setup_output_directory(config)
    
    # Read and preprocess data
    df = read_chat_log(csv_file_path)
    df = preprocess_data(df)
    
    # Determine chat type
    is_group = determine_chat_type(df)
    
    # Calculate most active day
    messages_per_day = df.groupby('Date').size()
    most_active_day = messages_per_day.idxmax()
    
    # Calculate average messages per day
    avg_messages_per_day = messages_per_day.mean()
    
    # Get first and last message dates
    first_message_date = df['Date'].min()
    last_message_date = df['Date'].max()
    
    # Common analysis
    common_analysis = perform_analysis(df)
    
    # Calculate top participants
    top_participants = df['Sender'].value_counts().to_dict()
    
    # Extract emojis
    emoji_counts = df['Message'].str.findall(EMOJI_PATTERN).explode().value_counts().to_dict()
    
    # Extract links
    links = df['Message'].str.extractall(URL_PATTERN)[0].value_counts().to_dict()
    
    # Type-specific analysis and visualizations
    if is_group:
        group_dynamics = analyze_group_dynamics(df)
        type_specific_viz = generate_group_visualizations(df, config)
        individual_metrics = None
    else:
        individual_metrics = analyze_individual_chat(df)
        type_specific_viz = generate_individual_visualizations(df, config)
        group_dynamics = None
    
    # Generate common visualizations
    common_viz = generate_visualizations(df, config)
    
    # Combine all visualizations
    all_visualizations = {**common_viz, **(type_specific_viz or {})}
    
    # Ensure dates are in datetime format before strftime
    start_date = df['Date'].min().strftime('%Y-%m-%d') if isinstance(df['Date'].min(), pd.Timestamp) else df['Date'].min()
    end_date = df['Date'].max().strftime('%Y-%m-%d') if isinstance(df['Date'].max(), pd.Timestamp) else df['Date'].max()
    
    return ChatAnalysis(
        is_group=is_group,
        participant_count=df['Sender'].nunique(),
        total_messages=len(df),
        analysis_period=(start_date, end_date),
        participant_stats=common_analysis['top_participants'],
        message_patterns=common_analysis,
        sentiment_analysis=common_analysis['sentiment_counts'],
        visualization_paths=all_visualizations,
        top_participants=top_participants,
        most_active_day=most_active_day,
        emojis=emoji_counts,
        links=links,
        first_message_date=first_message_date,
        last_message_date=last_message_date,
        avg_messages_per_day=avg_messages_per_day,
        group_dynamics=group_dynamics,
        conversation_balance=individual_metrics,
        interaction_network=group_dynamics['interaction_data'] if group_dynamics else None,
        response_patterns=individual_metrics['avg_response_times'] if individual_metrics else None
    )

def save_visualization(fig: Any, filename: str, config: VisualizationConfig) -> Path:
    """Save visualization to file and return the path."""
    output_path = config.output_dir / filename
    
    # Handle WordCloud objects differently
    if isinstance(fig, WordCloud):
        fig.to_file(str(output_path))
    else:
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
