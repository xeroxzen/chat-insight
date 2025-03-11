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

# Adding configuration
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

    def set_output_dir(self, output_dir: Path):
        """Set the output directory for visualizations"""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

# Adding constants
EMOJI_PATTERN = r'[\U0001F000-\U0001FFFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U00002B50\U00002705\U0001F1E6-\U0001F1FF\U00002639-\U0001F6FF\U0001F900-\U0001F9FF\U00002190-\U00002BFF]+'
URL_PATTERN = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
DAY_START_HOUR = 6
EXCLUDED_MESSAGES = [
    'joined', 'left', 'removed', 'changed', 'image omitted', 
    'video omitted', 'video call', 'voice call', 'audio omitted',
    'missed voice', 'missed video'
]

# Adding logger configuration
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
            df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d/%m/%Y')
        
        required_columns = {'Date', 'Time', 'Sender', 'Message'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading chat log: {str(e)}")
        raise

def normalize_sender_name(sender: str) -> str:
    """
    Normalize sender names by preserving emojis but ensuring consistent representation.
    This helps with proper grouping of messages by sender even when names contain emojis.
    
    Args:
        sender: The sender name that may contain emojis
        
    Returns:
        Normalized sender name with emojis preserved
    """
    # Extracting emojis from the sender name
    emojis = re.findall(EMOJI_PATTERN, sender)
    
    # Getting the text part of the name (without emojis)
    text_part = re.sub(EMOJI_PATTERN, '', sender).strip()
    
    # If there are emojis, appending them to the text part in a consistent order
    if emojis:
        return f"{text_part} {''.join(emojis)}"
    
    return text_part

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the chat log data with improved handling of date/time and emojis.
    """
    try:
        # Creating a copy to avoid modifying the original
        df = df.copy()
        
        # Converting Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Converting Time to datetime.time objects
        df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S').dt.time
        
        # Normalize sender names to handle emojis consistently
        df['Sender'] = df['Sender'].apply(normalize_sender_name)
        
        # Filtering out system messages and media
        pattern = '|'.join(EXCLUDED_MESSAGES)
        df = df[~df['Message'].astype(str).str.contains(pattern, case=False)]
        
        # Cleaning up missing values
        df = df.dropna(subset=['Date', 'Time', 'Message'])
        df.loc[:, 'Message'] = df['Message'].astype(str)
        
        # Adding DateTime column for easier analysis
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
    """
    Preprocess a message by removing non-word characters (except emojis) and converting to lowercase.
    Preserves emojis for better analysis.
    """
    # Extracting emojis from the message
    emojis = re.findall(EMOJI_PATTERN, message)
    
    # Removing non-word characters and converting to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', message).lower()
    
    # Adding the emojis back to the cleaned text
    return cleaned_text + ' ' + ' '.join(emojis)

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment of messages in the chat log with improved emoji handling.
    """
    try:
        # Creating a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Initializing sentiment columns with default values
        df['Sentiment'] = 0.0
        df['Sentiment_Label'] = 'Neutral'
        
        # Defining common positive and negative emojis for sentiment adjustment
        positive_emojis = ['😊', '😁', '😄', '😃', '😀', '🙂', '😍', '🥰', '❤️', '👍', '🙏', '🤗']
        negative_emojis = ['😢', '😭', '😞', '😔', '😟', '😕', '😠', '😡', '👎', '💔']
        
        # Applying sentiment analysis row by row with error handling
        for idx, row in df.iterrows():
            try:
                message = str(row['Message'])
                
                # Extracting emojis from the message
                message_emojis = re.findall(EMOJI_PATTERN, message)
                
                # Calculating base sentiment using TextBlob
                sentiment = TextBlob(message).sentiment.polarity
                
                # Adjusting sentiment based on emojis
                emoji_sentiment = 0
                for emoji in message_emojis:
                    if any(pos_emoji in emoji for pos_emoji in positive_emojis):
                        emoji_sentiment += 0.2
                    elif any(neg_emoji in emoji for neg_emoji in negative_emojis):
                        emoji_sentiment -= 0.2
                
                # Combining text and emoji sentiment (with a cap)
                adjusted_sentiment = max(min(sentiment + emoji_sentiment, 1.0), -1.0)
                
                df.at[idx, 'Sentiment'] = adjusted_sentiment
                df.at[idx, 'Sentiment_Label'] = (
                    'Positive' if adjusted_sentiment > 0 
                    else 'Negative' if adjusted_sentiment < 0 
                    else 'Neutral'
                )
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for message: {e}")
                continue
        
        return df
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Ensuring sentiment columns exist even if analysis fails
        if 'Sentiment' not in df.columns:
            df['Sentiment'] = 0.0
        if 'Sentiment_Label' not in df.columns:
            df['Sentiment_Label'] = 'Neutral'
        return df

def analyze_emoji_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze emoji usage patterns by sender.
    
    Args:
        df: DataFrame containing chat data
        
    Returns:
        Dictionary with emoji usage statistics
    """
    # Creating a copy to avoid modifying the original
    df = df.copy()
    
    # Extracting emojis from each message
    df['Emojis'] = df['Message'].apply(lambda x: re.findall(EMOJI_PATTERN, str(x)))
    
    # Counting emojis per message
    df['EmojiCount'] = df['Emojis'].apply(len)
    
    # Getting top 5 participants by message count
    top_participants = df['Sender'].value_counts().nlargest(5).index.tolist()
    
    # Calculating emoji usage statistics by sender
    emoji_stats = {}
    
    # Overall emoji counts
    all_emojis = [emoji for emojis in df['Emojis'] for emoji in emojis]
    top_emojis = Counter(all_emojis).most_common(10)
    
    # Emoji usage by sender (limited to top 5 participants)
    for sender in top_participants:
        sender_df = df[df['Sender'] == sender]
        
        # Counting total emojis used by this sender
        sender_emojis = [emoji for emojis in sender_df['Emojis'] for emoji in emojis]
        
        # Calculating emoji statistics
        emoji_stats[sender] = {
            'total_emoji_count': len(sender_emojis),
            'avg_emojis_per_message': len(sender_emojis) / len(sender_df) if len(sender_df) > 0 else 0,
            'messages_with_emojis': len(sender_df[sender_df['EmojiCount'] > 0]),
            'emoji_percentage': len(sender_df[sender_df['EmojiCount'] > 0]) / len(sender_df) * 100 if len(sender_df) > 0 else 0,
            'favorite_emojis': Counter(sender_emojis).most_common(5) if sender_emojis else []
        }
    
    return {
        'top_emojis': top_emojis,
        'emoji_stats_by_sender': emoji_stats,
        'total_emoji_count': len(all_emojis),
        'messages_with_emojis': len(df[df['EmojiCount'] > 0]),
        'emoji_percentage': len(df[df['EmojiCount'] > 0]) / len(df) * 100 if len(df) > 0 else 0
    }

def perform_analysis(df: pd.DataFrame) -> dict:
    """Perform various analyses on the chat log."""
    try:
        logger.info("Starting perform_analysis")
        
        # Ensuring Date is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Initializing basic metrics first
        date_counts = df.groupby(df['Date'].dt.date)['Message'].count()
        most_active_day = date_counts.idxmax().strftime('%Y-%m-%d')
        
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
        most_active_time = df['Hour'].value_counts().idxmax()
        top_participants = df['Sender'].value_counts().nlargest(2)
        avg_messages_per_day = date_counts.mean()
        
        # Basic text analysis
        # Extracting emojis using the same method as in analyze_emoji_usage for consistency
        df['Emojis'] = df['Message'].apply(lambda x: re.findall(EMOJI_PATTERN, str(x)))
        all_emojis = [emoji for emojis in df['Emojis'] for emoji in emojis]
        emojis = dict(Counter(all_emojis).most_common(15))
        
        links = df['Message'].str.extractall(URL_PATTERN)[0].value_counts().head(10).to_dict()
        
        # Word analysis
        stop_words = set(stopwords.words('english'))
        all_messages = ' '.join(df['Message'].apply(preprocess_message))
        words = [word for word in all_messages.split() if word not in stop_words]
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(20)
        
        logger.info("Basic analysis completed, starting sentiment analysis")
        
        # Simplified sentiment analysis
        try:
            df['Sentiment'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df['Sentiment_Label'] = df['Sentiment'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
            )
            sentiment_counts = df['Sentiment_Label'].value_counts().to_dict()
        except Exception as se:
            logger.error(f"Sentiment analysis failed: {str(se)}")
            sentiment_counts = {'Neutral': len(df)}
        
        logger.info("Sentiment analysis completed, starting additional analyses")
        
        # Emoji usage analysis
        emoji_analysis = analyze_emoji_usage(df)
        
        # Response time analysis
        df_sorted = df.sort_values('DateTime')
        df_sorted['TimeDiff'] = df_sorted['DateTime'].diff()
        avg_response_time = df_sorted.groupby('Sender')['TimeDiff'].mean()
        
        # Activity hours analysis
        active_hours_by_sender = df.groupby(['Sender', df['DateTime'].dt.hour])['Message'].count().unstack()
        peak_hours = {sender: active_hours_by_sender.loc[sender].idxmax() for sender in active_hours_by_sender.index}
        
        # Message streak analysis
        df_sorted['TimeDiff'] = df_sorted['DateTime'].diff().dt.total_seconds()
        streak_threshold = 300  # 5 minutes
        df_sorted['NewStreak'] = df_sorted['TimeDiff'] > streak_threshold
        df_sorted['StreakId'] = df_sorted['NewStreak'].cumsum()
        streak_counts = df_sorted.groupby('StreakId').size()
        longest_streak = streak_counts.max()
        
        # Question analysis
        df['IsQuestion'] = df['Message'].str.contains(r'\b(who|what|when|where|why|how)\b|\?', case=False)
        questions_by_sender = df[df['IsQuestion']]['Sender'].value_counts()
        
        # Message length analysis
        df['MessageLength'] = df['Message'].str.len()
        avg_message_length = df.groupby('Sender')['MessageLength'].mean()
        
        logger.info("All analyses completed, preparing results")
        
        # Prepare results
        analysis_results = {
            "most_active_day": most_active_day,
            "most_active_time": most_active_time,
            "most_common_words": most_common_words,
            "top_participants": top_participants.to_dict(),
            "first_message_date": df['Date'].min().strftime('%Y-%m-%d'),
            "last_message_date": df['Date'].max().strftime('%Y-%m-%d'),
            "avg_messages_per_day": avg_messages_per_day,
            "emojis": emojis,
            "links": links,
            "sentiment_counts": sentiment_counts,
            "average_response_times": avg_response_time.to_dict(),
            "peak_activity_hours": peak_hours,
            "longest_message_streak": longest_streak,
            "questions_asked": questions_by_sender.to_dict(),
            "average_message_length": avg_message_length.to_dict(),
            "emoji_usage": emoji_analysis
        }
        
        logger.info("Analysis completed successfully")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in perform_analysis: {str(e)}")
        logger.error(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        raise

def generate_visualizations(df: pd.DataFrame, config: VisualizationConfig, chat_type='friends') -> Dict[str, Path]:
    """Generate visualizations from the chat log."""
    try:
        visualizations = {}
        logger.info(f"Generating visualizations in directory: {config.output_dir}")
        
        # Getting top 20 participants by message count for group chat visualizations
        top_20_senders = df['Sender'].value_counts().nlargest(20).index.tolist()
        
        # Adding MessageLength column if it doesn't exist
        if 'MessageLength' not in df.columns:
            df['MessageLength'] = df['Message'].str.len()
        
        # Creating and saving wordcloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=STOPWORDS, 
            max_words=200
        ).generate(' '.join(df['Message']))
        
        visualizations['wordcloud'] = save_visualization(wordcloud, 'wordcloud.png', config)
        logger.info(f"Saved wordcloud to: {visualizations['wordcloud']}")
        
        # First message sender analysis
        df_filtered = df[df['Time'] >= pd.to_datetime('06:00:00').time()]
        first_messages = df_filtered.groupby('Date').first()
        first_sender_counts = first_messages['Sender'].value_counts()
        
        # Limiting to top 20 participants for group chats
        if chat_type == 'friends':
            first_sender_counts = first_sender_counts[first_sender_counts.index.isin(top_20_senders)]
        
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
        
        # Message Length Distribution
        plt.figure(figsize=config.figure_sizes['default'])
        if chat_type == 'friends':
            # Limiting to top 20 participants for group chats
            df_top20 = df[df['Sender'].isin(top_20_senders)]
            sns.boxplot(x='Sender', y='MessageLength', data=df_top20)
        else:
            sns.boxplot(x='Sender', y='MessageLength', data=df)
        plt.title('Message Length Distribution by Sender')
        plt.ylabel('Message Length (characters)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['message_length_distribution'] = save_visualization(plt.gcf(), 'message_length_distribution.png', config)
        plt.close()
        
        # Average messages per day by sender
        avg_messages_per_day_by_sender = df.groupby('Sender')['Date'].value_counts().groupby('Sender').mean()
        
        # Limiting to top 20 participants for group chats
        if chat_type == 'friends':
            avg_messages_per_day_by_sender = avg_messages_per_day_by_sender[avg_messages_per_day_by_sender.index.isin(top_20_senders)]
        
        plt.figure(figsize=(10, 6))
        avg_messages_per_day_by_sender.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
        plt.title('Average Messages per Day by Sender')
        plt.xlabel('Sender')
        plt.ylabel('Average Number of Messages per Day')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['average_messages_per_day'] = save_visualization(plt.gcf(), 'average_messages_per_day.png', config)
        plt.close()
        
        # Sentiment counts
        sentiment_counts = df['Sentiment_Label'].value_counts()
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar', color=['red', 'green', 'orange', 'skyblue', 'purple'])
        plt.title('Sentiment Counts')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Messages')
        plt.tight_layout()
        visualizations['sentiment_counts'] = save_visualization(plt.gcf(), 'sentiment_counts.png', config)
        plt.close()
        
        # Links shared by sender
        links_shared = df[df['Message'].str.contains('http[s]?://')]['Sender'].value_counts()
        if chat_type == 'friends':
            # Limiting to top 20 participants for group chats
            links_shared = links_shared[links_shared.index.isin(top_20_senders)]
        
        plt.figure(figsize=(8, 8))
        plt.pie(links_shared, labels=links_shared.index, autopct='%1.1f%%', startangle=90)
        plt.title('Links Shared by Each Sender')
        plt.axis('equal')
        plt.tight_layout()
        visualizations['links_shared_by_sender'] = save_visualization(plt.gcf(), 'links_shared_by_sender.png', config)
        plt.close()
        
        # Top emojis
        emoji_counts = df['Message'].str.findall(EMOJI_PATTERN).explode()
        if not emoji_counts.empty:
            top_10_emojis = emoji_counts.value_counts().head(10)
            plt.figure(figsize=(10, 6))
            top_10_emojis.plot(kind='bar', color=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'black'])
            plt.title('Top 10 Emojis Used')
            plt.xlabel('Emojis')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            visualizations['top_emojis'] = save_visualization(plt.gcf(), 'top_emojis.png', config)
            plt.close()
        
        # Messages per sender
        messages_count_per_sender = df['Sender'].value_counts()
        if chat_type == 'friends':
            # Limiting to top 20 participants for group chats
            messages_count_per_sender = messages_count_per_sender.head(20)
        
        plt.figure(figsize=(10, 6))
        messages_count_per_sender.plot(kind='bar', color=['lightblue', 'yellow'])
        plt.title('Total Messages Count per Sender')
        plt.xlabel('Sender')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['messages_count_per_sender'] = save_visualization(plt.gcf(), 'messages_count_per_sender.png', config)
        plt.close()
        
        # Message activity over time
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        message_activity = df.groupby('Date')['Message'].count()
        plt.figure(figsize=(10, 6))
        message_activity.plot(kind='line', marker='o', color='skyblue')
        plt.title('Message Activity Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Messages')
        plt.grid(True)
        plt.tight_layout()
        visualizations['message_activity_over_time'] = save_visualization(plt.gcf(), 'message_activity_over_time.png', config)
        plt.close()
        
        # Message activity heatmap
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        heatmap_data = df.pivot_table(index='DayOfWeek', columns='Hour', values='Message', aggfunc='count', fill_value=0)
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(ordered_days)
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        plt.title('Message Activity by Hour and Day of the Week')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Day of the Week')
        visualizations['message_activity_heatmap'] = save_visualization(plt.gcf(), 'message_activity_heatmap.png', config)
        plt.close()
        
        # Response time distribution
        df_sorted = df.sort_values('DateTime')
        response_times = df_sorted['DateTime'].diff().dt.total_seconds() / 60
        plt.figure(figsize=config.figure_sizes['default'])
        sns.histplot(data=response_times[response_times < 60], bins=30)
        plt.title('Response Time Distribution (within 1 hour)')
        plt.xlabel('Response Time (minutes)')
        visualizations['response_time_distribution'] = save_visualization(plt.gcf(), 'response_time_distribution.png', config)
        plt.close()
        
        # Daily pattern
        if chat_type == 'friends':
            # Limiting to top 20 participants for group chats
            df_pattern = df[df['Sender'].isin(top_20_senders)]
            daily_pattern = df_pattern.groupby([df_pattern['DateTime'].dt.hour, 'Sender'])['Message'].count().unstack()
        else:
            daily_pattern = df.groupby([df['DateTime'].dt.hour, 'Sender'])['Message'].count().unstack()
            
        plt.figure(figsize=config.figure_sizes['default'])
        daily_pattern.plot(kind='line', marker='o')
        plt.title('Daily Conversation Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Messages')
        plt.legend(title='Sender')
        visualizations['daily_pattern'] = save_visualization(plt.gcf(), 'daily_pattern.png', config)
        plt.close()
        
        # Emoji usage by sender
        if 'Emojis' not in df.columns:
            df['Emojis'] = df['Message'].apply(lambda x: re.findall(EMOJI_PATTERN, str(x)))
            
        if 'EmojiCount' not in df.columns:
            df['EmojiCount'] = df['Emojis'].apply(len)
        
        # Getting top 5 participants by message count for emoji visualization
        top_5_senders = df['Sender'].value_counts().nlargest(5).index.tolist()
        df_top5 = df[df['Sender'].isin(top_5_senders)]
        
        emoji_usage_by_sender = df_top5.groupby('Sender')['EmojiCount'].agg(['sum', 'mean'])
        emoji_usage_by_sender.columns = ['Total Emojis', 'Average Emojis per Message']
        
        plt.figure(figsize=(12, 6))
        emoji_usage_by_sender['Total Emojis'].plot(kind='bar', color='skyblue', position=0, width=0.4, alpha=0.7)
        
        # Creating a second axis for the average
        ax2 = plt.twinx()
        emoji_usage_by_sender['Average Emojis per Message'].plot(kind='bar', color='salmon', position=1, width=0.4, alpha=0.7, ax=ax2)
        
        plt.title('Emoji Usage by Sender')
        plt.xlabel('Sender')
        plt.ylabel('Total Emoji Count')
        ax2.set_ylabel('Average Emojis per Message')
        plt.legend(['Total Emojis'], loc='upper left')
        ax2.legend(['Average Emojis per Message'], loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        visualizations['emoji_usage_by_sender'] = save_visualization(plt.gcf(), 'emoji_usage_by_sender.png', config)
        plt.close()
        
        # Group-specific visualizations
        if chat_type == 'friends':
            # Group participation
            top_15_senders = df['Sender'].value_counts().head(15)
            plt.figure(figsize=config.figure_sizes['default'])
            sns.barplot(x=top_15_senders.index, y=top_15_senders.values)
            plt.title('Top 15 Group Participation Distribution')
            plt.xticks(rotation=45, ha='right')
            visualizations['group_participation'] = save_visualization(plt.gcf(), 'group_participation.png', config)
            plt.close()
            
            # Group interaction network
            df_filtered = df[df['Sender'].isin(top_15_senders.index)]
            G = nx.from_pandas_edgelist(
                df_filtered.sort_values('DateTime').assign(Next_Sender=lambda x: x['Sender'].shift(-1))
                .groupby(['Sender', 'Next_Sender']).size().reset_index(name='weight'),
                'Sender', 'Next_Sender', 'weight'
            )
            pos = nx.spring_layout(G)
            plt.figure(figsize=config.figure_sizes['default'])
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=1000, font_size=8)
            visualizations['group_interaction_network'] = save_visualization(plt.gcf(), 'group_interaction_network.png', config)
            plt.close()
        else:
            # Conversation flow for individual chats
            plt.figure(figsize=config.figure_sizes['default'])
            timeline = df.set_index('DateTime')['Sender'].astype('category').cat.codes
            plt.plot(timeline.index, timeline.values, 'o-')
            plt.title('Conversation Flow')
            plt.ylabel('Participant')
            plt.yticks([0, 1], df['Sender'].unique())
            visualizations['conversation_flow'] = save_visualization(plt.gcf(), 'conversation_flow.png', config)
            plt.close()
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return {}  # Return empty dict if visualizations fail

@dataclass
class ChatAnalysis:
    """Container for chat analysis results"""
    is_group: bool = True
    participant_count: int = 0
    total_messages: int = 0
    analysis_period: Tuple[str, str] = ('', '')  # (start_date, end_date)
    participant_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    message_patterns: Dict[str, Any] = field(default_factory=dict)
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    visualization_paths: Dict[str, Path] = field(default_factory=dict)
    top_participants: Dict[str, int] = field(default_factory=dict)
    most_active_day: str = None     
    most_active_time: int = None
    most_common_words: List[Tuple[str, int]] = field(default_factory=list)
    stop_words = set(stopwords.words('english')).union({'im', 'like', 'ill', 'the', 'will', 'edited', 'dont'})
    emojis: Dict[str, int] = field(default_factory=dict)
    emoji_usage: Dict[str, Any] = field(default_factory=dict)  # Detailed emoji analysis
    emoji_stats_by_sender: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Emoji stats per sender
    links: Dict[str, int] = field(default_factory=dict)
    first_message_date: str = None  
    last_message_date: str = None  
    avg_messages_per_day: float = 0.0
    
    # Group-specific metrics
    group_dynamics: Optional[Dict[str, Any]] = field(default_factory=dict)
    interaction_network: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Individual-specific metrics
    conversation_balance: Optional[Dict[str, Any]] = field(default_factory=dict)
    response_patterns: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert analysis results to dictionary"""
        return {
            "is_group": self.is_group,
            "participant_count": self.participant_count,
            "total_messages": self.total_messages,
            "analysis_period": self.analysis_period,
            "participant_stats": self.participant_stats,
            "message_patterns": self.message_patterns,
            "sentiment_analysis": self.sentiment_analysis,
            "visualization_paths": {k: str(v) for k, v in self.visualization_paths.items()},
            "top_participants": self.top_participants,
            "most_active_day": self.most_active_day,  # Already a string
            "most_active_time": self.most_active_time,
            "most_common_words": self.most_common_words,
            "emojis": self.emojis,
            "emoji_usage": self.emoji_usage,
            "emoji_stats_by_sender": self.emoji_stats_by_sender,
            "links": self.links,
            "first_message_date": self.first_message_date,  # Already a string
            "last_message_date": self.last_message_date,  # Already a string
            "avg_messages_per_day": self.avg_messages_per_day,
            "group_dynamics": self.group_dynamics,
            "interaction_network": self.interaction_network,
            "conversation_balance": self.conversation_balance,
            "response_patterns": self.response_patterns
        }

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
    # Getting top 15 participants by message count
    message_counts = df['Sender'].value_counts()
    top_15_participants = message_counts.head(15)
    
    # Message distribution for top 15
    most_active_member = top_15_participants.index[0]
    least_active_member = top_15_participants.index[-1]
    
    # Interaction patterns for top 15
    df_filtered = df[df['Sender'].isin(top_15_participants.index)]
    df_filtered['Hour'] = df_filtered['DateTime'].dt.hour
    peak_hours = df_filtered.groupby(['Sender', 'Hour'])['Message'].count().groupby('Sender').idxmax()
    
    # Creating interaction network data for top 15
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
    # Getting top 15 participants
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

def analyze_chat_log(csv_file_path: str, output_dir: Optional[Path] = None) -> dict:
    """
    Analyze the chat log and return results
    
    Args:
        csv_file_path: Path to the CSV file containing chat data
        output_dir: Optional path to user-specific visualization directory
    """
    try:
        # Reading and validating the CSV file
        df = read_chat_log(csv_file_path)
        
        # Preprocessing data
        df = preprocess_data(df)
        
        # Adding sentiment analysis columns before any other analysis
        try:
            df['Sentiment'] = df['Message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df['Sentiment_Label'] = df['Sentiment'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
            )
        except Exception as se:
            logger.warning(f"Sentiment analysis failed: {str(se)}")
            df['Sentiment'] = 0.0
            df['Sentiment_Label'] = 'Neutral'
        
        # Determining if it's a group chat
        is_group = determine_chat_type(df)
        
        # Creating visualization config
        config = VisualizationConfig()
        if output_dir:
            config.set_output_dir(output_dir)
        setup_output_directory(config)
        
        # Generating visualizations
        chat_type = 'friends' if is_group else 'romantic'
        visualization_paths = generate_visualizations(df, config, chat_type)
        
        # Creating analysis object
        analysis = ChatAnalysis()
        
        # Basic metrics
        analysis.is_group = is_group
        analysis.participant_count = df['Sender'].nunique()
        analysis.total_messages = len(df)
        
        # Handling date conversions properly
        first_date = pd.to_datetime(df['Date'].min())
        last_date = pd.to_datetime(df['Date'].max())
        analysis.first_message_date = first_date.strftime('%Y-%m-%d')
        analysis.last_message_date = last_date.strftime('%Y-%m-%d')
        
        analysis.visualization_paths = visualization_paths
        
        # Performing common analysis
        common_results = perform_analysis(df)
        
        # Updating analysis object with common results
        analysis.most_active_day = common_results['most_active_day']
        analysis.most_active_time = common_results['most_active_time']
        analysis.most_common_words = common_results['most_common_words']
        analysis.avg_messages_per_day = common_results['avg_messages_per_day']
        analysis.top_participants = common_results['top_participants']
        analysis.emojis = common_results.get('emojis', {})
        analysis.links = common_results.get('links', {})
        analysis.sentiment_analysis = common_results.get('sentiment_counts', {'Neutral': len(df)})
        analysis.message_patterns = common_results
        
        # Adding emoji usage analysis
        if 'emoji_usage' in common_results:
            analysis.emoji_usage = common_results['emoji_usage']
            if 'emoji_stats_by_sender' in common_results['emoji_usage']:
                analysis.emoji_stats_by_sender = common_results['emoji_usage']['emoji_stats_by_sender']
        
        # Group-specific analysis
        if is_group:
            group_results = analyze_group_dynamics(df)
            analysis.group_dynamics = group_results
            analysis.interaction_network = group_results.get('interaction_data', {})
            group_viz = generate_group_visualizations(df, config)
            visualization_paths.update(group_viz)
        else:
            individual_results = analyze_individual_chat(df)
            analysis.conversation_balance = individual_results
            analysis.response_patterns = individual_results.get('avg_response_times', {})
            individual_viz = generate_individual_visualizations(df, config)
            visualization_paths.update(individual_viz)
        
        # Creating message activity heatmap
        heatmap_path = create_message_activity_heatmap(df, config)
        visualization_paths['heatmap'] = heatmap_path
        
        # Extracting and counting emojis from all messages
        if not analysis.emojis:  # Only set if not already set from common_results
            all_emojis = []
            for message in df['Message']:
                emojis_found = re.findall(EMOJI_PATTERN, str(message))
                all_emojis.extend(emojis_found)
            
            emoji_counter = Counter(all_emojis)
            analysis.emojis = {emoji: count for emoji, count in emoji_counter.most_common(15)}
        
        # Returning results as dictionary
        return analysis.to_dict()
        
    except Exception as e:
        logger.error(f"Error analyzing chat log: {str(e)}")
        raise ValueError(f"Error analyzing chat log: {str(e)}") from e

def save_visualization(fig: Any, filename: str, config: VisualizationConfig) -> Path:
    """Save visualization to file and return the path."""
    output_path = config.output_dir / filename
    logger.info(f"Saving visualization: {filename} to {output_path}")
    
    # Handling WordCloud objects differently
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
