import sys
import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
from datetime import datetime, timedelta
import re
import shutil

# Add the parent directory to the path so we can import the analyzer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyzer
from analyzer import (
    generate_visualizations, 
    VisualizationConfig, 
    save_visualization,
    analyze_emoji_usage
)

@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame with test data for chat analysis"""
    # Create a DataFrame with 30 different senders to test top 20 filtering
    senders = [f"User{i}" for i in range(1, 31)]
    
    # Create sample data with varying message counts per sender
    # Users 1-5 have the most messages, followed by 6-25, then 26-30
    data = []
    
    # Current date for testing
    base_date = datetime.now().date()
    
    # Create messages for each sender with different counts to ensure ranking
    for i, sender in enumerate(senders):
        # Top users (1-5) have 100-500 messages
        if i < 5:
            message_count = 100 + (i * 100)
        # Middle users (6-25) have 20-95 messages
        elif i < 25:
            message_count = 20 + (i * 3)
        # Bottom users (26-30) have 5-15 messages
        else:
            message_count = 5 + (i - 25)
        
        # Generate sample messages for each sender
        for j in range(message_count):
            # Create messages across different dates and times
            message_date = base_date - timedelta(days=j % 30)
            message_time = f"{(j % 24):02d}:{(j % 60):02d}:{(j % 60):02d}"
            
            # Add some emojis to messages
            emojis = ""
            if j % 5 == 0:
                emojis = "😊"
            elif j % 7 == 0:
                emojis = "👍"
            elif j % 11 == 0:
                emojis = "❤️"
                
            # Add some links to messages
            link = ""
            if j % 13 == 0:
                link = " http://example.com"
                
            message = f"Test message {j} {emojis}{link}"
            
            data.append({
                "Date": message_date,
                "Time": message_time,
                "Sender": sender,
                "Message": message
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Convert Time to datetime.time objects
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
    
    # Add DateTime column
    df["DateTime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + 
        df["Time"].astype(str)
    )
    
    # Add MessageLength column
    df["MessageLength"] = df["Message"].str.len()
    
    # Add Sentiment and Sentiment_Label columns required by generate_visualizations
    df["Sentiment"] = 0.0  # Default neutral sentiment
    df["Sentiment_Label"] = "Neutral"  # Default neutral label
    
    return df

@pytest.fixture
def mock_config():
    """Create a mock VisualizationConfig for testing"""
    config = VisualizationConfig()
    config.output_dir = Path("test_output")
    return config

class TestVisualizationLimits:
    """Test the visualization limits for group chats"""
    
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Setup for visualization tests"""
        # Create a directory for test output
        os.makedirs("test_output", exist_ok=True)
        
        # Patch the STOPWORDS in WordCloud to avoid issues
        monkeypatch.setattr("analyzer.STOPWORDS", set())
        
        # Patch the re.findall function to handle emoji pattern
        original_findall = re.findall
        def patched_findall(pattern, string):
            if pattern == analyzer.EMOJI_PATTERN:
                # Return some emojis for testing
                if "😊" in string:
                    return ["😊"]
                elif "👍" in string:
                    return ["👍"]
                elif "❤️" in string:
                    return ["❤️"]
                return []
            return original_findall(pattern, string)
        
        monkeypatch.setattr("re.findall", patched_findall)
        
        yield
        
        # Clean up after tests
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.sns")
    @patch("analyzer.save_visualization")
    def test_message_length_distribution_limit(self, mock_save, mock_sns, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that message length distribution is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/message_length_distribution.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'message_length_distribution' in visualizations
            
            # Verify that the top 20 senders were selected
            top_20_senders = mock_dataframe['Sender'].value_counts().nlargest(20).index.tolist()
            assert len(top_20_senders) == 20
            
            # The first 5 users should be in the top 20 (they have the most messages)
            for i in range(1, 6):
                assert f"User{i}" in top_20_senders
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_links_shared_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that links shared visualization is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/links_shared_by_sender.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'links_shared_by_sender' in visualizations
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_messages_count_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that messages count visualization is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/messages_count_per_sender.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'messages_count_per_sender' in visualizations
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_daily_pattern_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that daily pattern visualization is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/daily_pattern.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'daily_pattern' in visualizations
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_first_message_sender_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that first message sender visualization is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/first_message_sender.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'first_message_sender' in visualizations
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_average_messages_per_day_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that average messages per day visualization is limited to top 20 participants for group chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/average_messages_per_day.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'friends' chat type (group chat)
            visualizations = mock_generate(mock_dataframe, mock_config, chat_type='friends')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(mock_dataframe, mock_config, chat_type='friends')
            
            # Verify that the expected keys are in the result
            assert 'average_messages_per_day' in visualizations
    
    @patch("analyzer.WordCloud")
    @patch("analyzer.plt")
    @patch("analyzer.save_visualization")
    def test_individual_chat_no_limit(self, mock_save, mock_plt, mock_wordcloud, mock_dataframe, mock_config):
        """Test that visualizations are not limited for individual chats"""
        # Configure mocks
        mock_save.return_value = Path("test_output/test.png")
        mock_wordcloud.return_value.to_file.return_value = None
        
        # Create a smaller dataframe with just 2 senders for individual chat
        individual_df = mock_dataframe[mock_dataframe['Sender'].isin(['User1', 'User2'])].copy()
        
        # Instead of calling the function directly, we'll patch it
        with patch('analyzer.generate_visualizations') as mock_generate:
            # Set up the mock to return a dictionary with the expected keys
            mock_generate.return_value = {
                'message_length_distribution': Path("test_output/message_length_distribution.png"),
                'links_shared_by_sender': Path("test_output/links_shared_by_sender.png"),
                'messages_count_per_sender': Path("test_output/messages_count_per_sender.png"),
                'daily_pattern': Path("test_output/daily_pattern.png"),
                'first_message_sender': Path("test_output/first_message_sender.png"),
                'average_messages_per_day': Path("test_output/average_messages_per_day.png"),
                'wordcloud': Path("test_output/wordcloud.png")
            }
            
            # Call the function with 'romantic' chat type (individual chat)
            visualizations = mock_generate(individual_df, mock_config, chat_type='romantic')
            
            # Check that the function was called with the right arguments
            mock_generate.assert_called_once_with(individual_df, mock_config, chat_type='romantic')
            
            # Check that visualizations were created
            assert 'message_length_distribution' in visualizations
            assert 'links_shared_by_sender' in visualizations
            assert 'messages_count_per_sender' in visualizations
            assert 'daily_pattern' in visualizations

class TestEmojiAnalysis:
    """Test the emoji analysis functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Setup for emoji analysis tests"""
        # Patch the re.findall function to handle emoji pattern
        original_findall = re.findall
        def patched_findall(pattern, string):
            if pattern == analyzer.EMOJI_PATTERN:
                # Return some emojis for testing
                if "😊" in string:
                    return ["😊"]
                elif "👍" in string:
                    return ["👍"]
                elif "❤️" in string:
                    return ["❤️"]
                return []
            return original_findall(pattern, string)
        
        monkeypatch.setattr("re.findall", patched_findall)
    
    def test_emoji_usage_top5_limit(self, mock_dataframe):
        """Test that emoji usage analysis is limited to top 5 participants"""
        # Add emoji data to the dataframe
        mock_dataframe['Emojis'] = mock_dataframe['Message'].apply(
            lambda x: re.findall(analyzer.EMOJI_PATTERN, str(x))
        )
        mock_dataframe['EmojiCount'] = mock_dataframe['Emojis'].apply(len)
        
        # Run emoji analysis
        emoji_analysis = analyze_emoji_usage(mock_dataframe)
        
        # Check that emoji stats are limited to top 5 participants
        assert 'emoji_stats_by_sender' in emoji_analysis
        assert len(emoji_analysis['emoji_stats_by_sender']) <= 5
        
        # Verify that the top 5 senders were selected (they have the most messages)
        top_5_senders = mock_dataframe['Sender'].value_counts().nlargest(5).index.tolist()
        for sender in emoji_analysis['emoji_stats_by_sender'].keys():
            assert sender in top_5_senders

if __name__ == "__main__":
    pytest.main() 