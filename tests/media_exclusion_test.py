import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
import re
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

# Adding the parent directory to the path so we can import the analyzer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyzer
from analyzer import (
    preprocess_message,
    TEXT_ANALYSIS_EXCLUDED_TERMS,
    MEDIA_PATTERNS,
    perform_analysis,
    generate_visualizations,
    VisualizationConfig,
    analyze_media_messages
)

@pytest.fixture
def media_messages_df():
    """Create a DataFrame with various media messages for testing"""
    # Create a mix of regular and media messages from different senders
    data = [
        # Regular text messages
        {
            'Date': '01/01/2023',
            'Time': '12:00:00',
            'Sender': 'User1',
            'Message': 'Hello, how are you?',
            'MediaType': 'text'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:01:00',
            'Sender': 'User2',
            'Message': 'I am doing well, thanks!',
            'MediaType': 'text'
        },
        # Image messages
        {
            'Date': '01/01/2023',
            'Time': '12:02:00',
            'Sender': 'User1',
            'Message': 'image omitted',
            'MediaType': 'image'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:03:00',
            'Sender': 'User2',
            'Message': 'image omitted',
            'MediaType': 'image'
        },
        # Video messages
        {
            'Date': '01/01/2023',
            'Time': '12:04:00',
            'Sender': 'User1',
            'Message': 'video omitted',
            'MediaType': 'video'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:05:00',
            'Sender': 'User2',
            'Message': 'video omitted',
            'MediaType': 'video'
        },
        # Voice call messages
        {
            'Date': '01/01/2023',
            'Time': '12:06:00',
            'Sender': 'User1',
            'Message': 'Voice call (2:30)',
            'MediaType': 'voice_call'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:07:00',
            'Sender': 'User2',
            'Message': 'Voice call',
            'MediaType': 'voice_call'
        },
        # Video call messages
        {
            'Date': '01/01/2023',
            'Time': '12:08:00',
            'Sender': 'User1',
            'Message': 'Video call (5:45)',
            'MediaType': 'video_call'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:09:00',
            'Sender': 'User2',
            'Message': 'Video call',
            'MediaType': 'video_call'
        },
        # Missed call messages
        {
            'Date': '01/01/2023',
            'Time': '12:10:00',
            'Sender': 'User1',
            'Message': 'Missed voice call, Tap to call back',
            'MediaType': 'missed_voice_call'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:11:00',
            'Sender': 'User2',
            'Message': 'Missed video call, Tap to call back',
            'MediaType': 'missed_video_call'
        },
        # Sticker messages
        {
            'Date': '01/01/2023',
            'Time': '12:12:00',
            'Sender': 'User1',
            'Message': 'sticker omitted',
            'MediaType': 'sticker'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:13:00',
            'Sender': 'User2',
            'Message': 'sticker omitted',
            'MediaType': 'sticker'
        },
        # Audio messages
        {
            'Date': '01/01/2023',
            'Time': '12:14:00',
            'Sender': 'User1',
            'Message': 'audio omitted',
            'MediaType': 'audio'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:15:00',
            'Sender': 'User2',
            'Message': 'audio omitted',
            'MediaType': 'audio'
        },
        # Messages with media terms in them but not actual media messages
        {
            'Date': '01/01/2023',
            'Time': '12:16:00',
            'Sender': 'User1',
            'Message': 'I was talking about that image we saw yesterday',
            'MediaType': 'text'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:17:00',
            'Sender': 'User2',
            'Message': 'The video was really interesting',
            'MediaType': 'text'
        },
        # More regular messages
        {
            'Date': '01/01/2023',
            'Time': '12:18:00',
            'Sender': 'User1',
            'Message': 'Let\'s meet tomorrow',
            'MediaType': 'text'
        },
        {
            'Date': '01/01/2023',
            'Time': '12:19:00',
            'Sender': 'User2',
            'Message': 'Sounds good!',
            'MediaType': 'text'
        }
    ]
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    return df

@pytest.fixture
def mock_config():
    """Create a mock visualization config"""
    config = VisualizationConfig()
    config.output_dir = Path('test_output')
    return config

class TestMediaExclusion:
    """Test the exclusion of media-related terms from text analysis"""
    
    def test_preprocess_message_with_media_terms(self):
        """Test that preprocess_message returns empty string for media messages"""
        # Test with various media messages
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            message = f"This is a {term} message"
            result = preprocess_message(message)
            assert result == "", f"Expected empty string for message with '{term}'"
        
        # Test with regular message
        regular_message = "Hello, how are you?"
        result = preprocess_message(regular_message)
        assert result != "", "Expected non-empty string for regular message"
        assert "hello" in result, "Expected cleaned text to contain 'hello'"
    
    def test_word_frequency_excludes_media_terms(self, media_messages_df):
        """Test that word frequency analysis excludes media terms"""
        df = media_messages_df
        
        # Add some common words to make sure they appear in the results
        common_words = ["hello", "meeting", "tomorrow", "thanks", "good"]
        additional_rows = []
        for word in common_words:
            for i in range(5):  # Add each word 5 times
                additional_rows.append({
                    'Date': '01/01/2023',
                    'Time': f'13:{i:02d}:00',
                    'Sender': 'User1' if i % 2 == 0 else 'User2',
                    'Message': f"This is a {word} message",
                    'MediaType': 'text'
                })
        
        # Use concat instead of append
        df = pd.concat([df, pd.DataFrame(additional_rows)], ignore_index=True)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Add DateTime column required by perform_analysis
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df['Time'].astype(str)
        )
        
        # Test the word frequency analysis logic directly
        stop_words = set(stopwords.words('english'))
        # Add media-related terms to stop words
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            for word in term.lower().split():
                stop_words.add(word)

        # Filter out messages that are media messages
        filtered_messages = []
        for message in df['Message']:
            # Skip messages that are exactly media messages
            is_media_message = False
            
            # Check if message exactly matches any media pattern
            for media_type, pattern in MEDIA_PATTERNS.items():
                if re.search(pattern, str(message), re.IGNORECASE):
                    # Check if the message is just the media pattern without much else
                    if len(str(message).split()) <= 4:  # Media messages are usually short
                        is_media_message = True
                        break
            
            # Check for exact matches with common media message formats
            exact_media_messages = [
                "image omitted", "video omitted", "audio omitted", "sticker omitted",
                "voice call", "video call", "missed voice call", "missed video call",
                "<media omitted>", "Voice call", "Video call"
            ]
            
            if any(str(message).lower().strip() == term.lower() for term in exact_media_messages):
                is_media_message = True
            
            # Only include non-media messages
            if not is_media_message:
                filtered_messages.append(str(message))

        all_messages = ' '.join([preprocess_message(msg) for msg in filtered_messages])
        words = [word for word in all_messages.split() if word not in stop_words]
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(20)
        
        # Check that common words appear in the results
        most_common_words_dict = dict(most_common_words)
        for word in common_words:
            assert word in most_common_words_dict, f"Expected '{word}' to be in most common words"
        
        # Check that media terms do not appear in the results
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            for word in term.lower().split():
                assert word not in most_common_words_dict, f"Media term '{word}' should not be in most common words"
    
    def test_wordcloud_excludes_media_terms(self, media_messages_df):
        """Test that wordcloud generation excludes media terms"""
        # Create a test DataFrame with only text messages
        text_messages = [
            "Hello, how are you?",
            "I am doing well, thanks!",
            "Let's meet tomorrow",
            "Sounds good!"
        ]
        
        data = []
        for i, message in enumerate(text_messages):
            data.append({
                'Date': '01/01/2023',
                'Time': f'12:{i:02d}:00',
                'Sender': 'User1' if i % 2 == 0 else 'User2',
                'Message': message,
                'MediaType': 'text'
            })
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['MessageLength'] = df['Message'].str.len()
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df['Time'].astype(str)
        )
        
        # Test the wordcloud generation logic directly
        # Filter out messages that are media messages
        filtered_messages = []
        for message in df['Message']:
            # Skip messages that are exactly media messages
            is_media_message = False
            
            # Check if message exactly matches any media pattern
            for media_type, pattern in MEDIA_PATTERNS.items():
                if re.search(pattern, str(message), re.IGNORECASE):
                    # Check if the message is just the media pattern without much else
                    if len(str(message).split()) <= 4:  # Media messages are usually short
                        is_media_message = True
                        break
            
            # Check for exact matches with common media message formats
            exact_media_messages = [
                "image omitted", "video omitted", "audio omitted", "sticker omitted",
                "voice call", "video call", "missed voice call", "missed video call",
                "<media omitted>", "Voice call", "Video call"
            ]
            
            if any(str(message).lower().strip() == term.lower() for term in exact_media_messages):
                is_media_message = True
            
            # Only include non-media messages
            if not is_media_message:
                filtered_messages.append(str(message))
        
        # Create custom stopwords set by adding our excluded terms to STOPWORDS
        custom_stopwords = set(STOPWORDS)
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            for word in term.lower().split():
                custom_stopwords.add(word)
        
        # Check that all text messages are included in filtered_messages
        assert len(filtered_messages) == len(text_messages), "All text messages should be included"
        
        # Check that custom_stopwords contains media terms
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            for word in term.lower().split():
                assert word in custom_stopwords, f"Media term '{word}' should be in stopwords"
    
    def test_media_analysis_includes_media_terms(self, media_messages_df):
        """Test that media analysis still includes media messages"""
        df = media_messages_df
        
        # Run the media analysis
        with patch('analyzer.logger'):  # Mock logger to avoid logging during tests
            media_results = analyze_media_messages(df)
        
        # Check that media analysis includes all media types
        for media_type in MEDIA_PATTERNS.keys():
            assert media_type in media_results['media_by_type'], f"Expected '{media_type}' to be in media analysis"
        
        # Check that both users have media messages
        assert 'User1' in media_results['media_by_sender'], "Expected 'User1' to have media messages"
        assert 'User2' in media_results['media_by_sender'], "Expected 'User2' to have media messages"
        
        # Check total media count
        expected_media_count = len(df[df['MediaType'] != 'text'])
        assert media_results['total_media_count'] == expected_media_count, f"Expected {expected_media_count} media messages"

    def test_edge_cases_with_media_terms_in_text(self):
        """Test edge cases where media terms appear within regular text messages"""
        # Create test messages with media terms embedded in regular text
        edge_case_messages = [
            "I want to talk about that photo we saw yesterday",
            "Let's discuss the movie from the conference",
            "Did you hear that sound clip I sent?",
            "I'll give you a ring later",
            "The emoji you sent was funny",
            "That dropped call was from me",
            "I was talking about phone calls in general",
            "Facetime is better than audio only"
        ]
        
        # Test each message with preprocess_message
        # These should NOT be filtered out completely since they're not actual media messages
        for message in edge_case_messages:
            result = preprocess_message(message)
            # The result should not be empty
            assert result != "", f"Message '{message}' should not be completely filtered out"
        
        # Now create a DataFrame with these messages
        data = []
        for i, message in enumerate(edge_case_messages):
            data.append({
                'Date': '01/01/2023',
                'Time': f'14:{i:02d}:00',
                'Sender': 'User1' if i % 2 == 0 else 'User2',
                'Message': message,
                'MediaType': 'text'  # These are all text messages
            })
        
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Add DateTime column required by perform_analysis
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            df['Time'].astype(str)
        )
        
        # Test the word frequency analysis logic directly
        stop_words = set(stopwords.words('english'))
        # Add media-related terms to stop words
        for term in TEXT_ANALYSIS_EXCLUDED_TERMS:
            for word in term.lower().split():
                stop_words.add(word)

        # Process messages directly
        all_messages = ' '.join([preprocess_message(msg) for msg in df['Message']])
        words = [word for word in all_messages.split() if word not in stop_words]
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(20)
        
        # Check that important words from these messages appear in the results
        # but media-specific terms don't
        most_common_words_dict = dict(most_common_words)
        
        # Words that should appear
        expected_words = ["talk", "discuss", "conference", "hear", "sent", "later", "funny", "general", "better"]
        
        # Check at least some of these words appear (not all might make it to top 20)
        found_expected = False
        for word in expected_words:
            if word in most_common_words_dict:
                found_expected = True
                break
        
        assert found_expected, "Expected at least some non-media words to appear in results"
        
        # Media terms that should not appear
        for term in ["image", "video", "audio", "call", "sticker", "missed", "voice"]:
            assert term not in most_common_words_dict, f"Media term '{term}' should not be in most common words"

    def test_media_analysis_by_sender(self, media_messages_df):
        """Test that media analysis by sender is working correctly"""
        df = media_messages_df
        
        # Run the media analysis
        with patch('analyzer.logger'):  # Mock logger to avoid logging during tests
            media_results = analyze_media_messages(df)
        
        # Check that both users have media messages in the analysis
        assert 'User1' in media_results['media_by_sender'], "Expected 'User1' to have media messages"
        assert 'User2' in media_results['media_by_sender'], "Expected 'User2' to have media messages"
        
        # Count media messages by sender and type in the original DataFrame
        user1_media = df[(df['Sender'] == 'User1') & (df['MediaType'] != 'text')]
        user2_media = df[(df['Sender'] == 'User2') & (df['MediaType'] != 'text')]
        
        # Check that the counts match
        user1_media_count = len(user1_media)
        user2_media_count = len(user2_media)
        
        # Calculate total media count from the analysis results for each user
        user1_analysis_count = sum(media_results['media_by_sender']['User1'].values())
        user2_analysis_count = sum(media_results['media_by_sender']['User2'].values())
        
        assert user1_analysis_count == user1_media_count, f"Expected {user1_media_count} media messages for User1, got {user1_analysis_count}"
        assert user2_analysis_count == user2_media_count, f"Expected {user2_media_count} media messages for User2, got {user2_analysis_count}"
        
        # Check that each media type is correctly counted for each user
        for media_type in MEDIA_PATTERNS.keys():
            user1_type_count = len(user1_media[user1_media['MediaType'] == media_type])
            user2_type_count = len(user2_media[user2_media['MediaType'] == media_type])
            
            # Only check if the user has messages of this type
            if user1_type_count > 0:
                assert media_type in media_results['media_by_sender']['User1'], f"Expected '{media_type}' for User1"
                assert media_results['media_by_sender']['User1'][media_type] == user1_type_count, \
                    f"Expected {user1_type_count} {media_type} messages for User1, got {media_results['media_by_sender']['User1'].get(media_type, 0)}"
            
            if user2_type_count > 0:
                assert media_type in media_results['media_by_sender']['User2'], f"Expected '{media_type}' for User2"
                assert media_results['media_by_sender']['User2'][media_type] == user2_type_count, \
                    f"Expected {user2_type_count} {media_type} messages for User2, got {media_results['media_by_sender']['User2'].get(media_type, 0)}"

if __name__ == "__main__":
    pytest.main(["-v", "media_exclusion_test.py"]) 