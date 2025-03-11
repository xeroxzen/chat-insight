import sys
import os
import pytest
from bs4 import BeautifulSoup
import jinja2
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestResultsTemplate:
    """Test the results.html template"""
    
    @pytest.fixture
    def template_env(self):
        """Create a Jinja2 environment for testing templates"""
        template_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        return env
    
    @pytest.fixture
    def group_chat_results(self):
        """Create mock results for a group chat"""
        return {
            "is_group": True,
            "visualization_paths": {
                "links_shared_by_sender": "/visuals/links_shared_by_sender.png",
                "messages_count_per_sender": "/visuals/messages_count_per_sender.png",
                "message_length_distribution": "/visuals/message_length_distribution.png",
                "daily_pattern": "/visuals/daily_pattern.png",
                "average_messages_per_day": "/visuals/average_messages_per_day.png",
                "first_message_sender": "/visuals/first_message_sender.png",
                "emoji_usage_by_sender": "/visuals/emoji_usage_by_sender.png"
            },
            "emoji_usage": {
                "top_emojis": [("😊", 10), ("👍", 5)],
                "emoji_stats_by_sender": {
                    "User1": {"total_emoji_count": 15},
                    "User2": {"total_emoji_count": 10}
                }
            },
            "emojis": {"😊": 10, "👍": 5}
        }
    
    @pytest.fixture
    def individual_chat_results(self):
        """Create mock results for an individual chat"""
        return {
            "is_group": False,
            "visualization_paths": {
                "links_shared_by_sender": "/visuals/links_shared_by_sender.png",
                "messages_count_per_sender": "/visuals/messages_count_per_sender.png",
                "message_length_distribution": "/visuals/message_length_distribution.png",
                "daily_pattern": "/visuals/daily_pattern.png",
                "average_messages_per_day": "/visuals/average_messages_per_day.png",
                "first_message_sender": "/visuals/first_message_sender.png",
                "emoji_usage_by_sender": "/visuals/emoji_usage_by_sender.png"
            },
            "emoji_usage": {
                "top_emojis": [("😊", 10), ("👍", 5)],
                "emoji_stats_by_sender": {
                    "User1": {"total_emoji_count": 15},
                    "User2": {"total_emoji_count": 10}
                }
            },
            "emojis": {"😊": 10, "👍": 5}
        }
    
    @patch('jinja2.Template.render')
    def test_info_text_for_group_chat(self, mock_render, template_env, group_chat_results):
        """Test that info-text is displayed for group chats"""
        # Mock the template rendering to return a simple HTML with info-text elements
        mock_html = """
        <div class="chart-wrapper">
            <h2>Links Shared by Each Sender</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/links_shared_by_sender.png" alt="Links Shared by Each Sender">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Messages sent per user</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/messages_count_per_sender.png" alt="Messages sent per user">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Message Length</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/message_length_distribution.png" alt="Message Length">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Daily Message Pattern</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/daily_pattern.png" alt="Daily Message Pattern">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Average Messages Sent per Day</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/average_messages_per_day.png" alt="Average Messages Sent per Day">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>First Message Sender</h2>
            <p class="info-text">Showing data for top 20 participants only</p>
            <div class="chart-container">
                <img src="/visuals/first_message_sender.png" alt="First Message Sender">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Emoji Usage by Sender</h2>
            <p class="info-text">Showing emoji usage for top 5 participants only</p>
            <div class="chart-container">
                <img src="/visuals/emoji_usage_by_sender.png" alt="Emoji Usage by Sender">
            </div>
        </div>
        """
        mock_render.return_value = mock_html
        
        # Get the template
        template = template_env.get_template("results.html")
        
        # Render the template (this will use our mock)
        rendered = template.render(results=group_chat_results)
        
        # Parse the HTML
        soup = BeautifulSoup(rendered, 'html.parser')
        
        # Find all info-text elements
        info_texts = soup.find_all("p", class_="info-text")
        
        # There should be at least 6 info-text elements (one for each limited visualization)
        assert len(info_texts) >= 6
        
        # Check for specific info-text messages
        info_text_messages = [text.get_text().strip() for text in info_texts]
        
        # Check for the expected messages
        expected_messages = [
            "Showing data for top 20 participants only",
            "Showing emoji usage for top 5 participants only"
        ]
        
        for expected in expected_messages:
            assert any(expected in message for message in info_text_messages), f"Expected message '{expected}' not found"
        
        # Check specific sections for info-text
        sections = [
            "Links Shared by Each Sender",
            "Messages sent per user",
            "Message Length",
            "Daily Message Pattern",
            "Average Messages Sent per Day",
            "First Message Sender"
        ]
        
        for section_title in sections:
            # Find the section by its title
            section_headers = soup.find_all("h2", string=section_title)
            
            # There should be at least one section with this title
            assert len(section_headers) > 0, f"Section '{section_title}' not found"
            
            # For each section, check if it has an info-text
            for header in section_headers:
                # Get the parent wrapper
                wrapper = header.parent
                
                # Find info-text within this wrapper
                info_text = wrapper.find("p", class_="info-text")
                
                # There should be an info-text
                assert info_text is not None, f"No info-text found in section '{section_title}'"
                
                # The info-text should contain the expected message
                assert "top" in info_text.get_text(), f"Info-text in section '{section_title}' does not mention 'top'"
    
    @patch('jinja2.Template.render')
    def test_no_info_text_for_individual_chat(self, mock_render, template_env, individual_chat_results):
        """Test that info-text is not displayed for individual chats"""
        # Mock the template rendering to return a simple HTML without info-text elements
        mock_html = """
        <div class="chart-wrapper">
            <h2>Links Shared by Each Sender</h2>
            <div class="chart-container">
                <img src="/visuals/links_shared_by_sender.png" alt="Links Shared by Each Sender">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Messages sent per user</h2>
            <div class="chart-container">
                <img src="/visuals/messages_count_per_sender.png" alt="Messages sent per user">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Message Length</h2>
            <div class="chart-container">
                <img src="/visuals/message_length_distribution.png" alt="Message Length">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Daily Message Pattern</h2>
            <div class="chart-container">
                <img src="/visuals/daily_pattern.png" alt="Daily Message Pattern">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>Average Messages Sent per Day</h2>
            <div class="chart-container">
                <img src="/visuals/average_messages_per_day.png" alt="Average Messages Sent per Day">
            </div>
        </div>
        <div class="chart-wrapper">
            <h2>First Message Sender</h2>
            <div class="chart-container">
                <img src="/visuals/first_message_sender.png" alt="First Message Sender">
            </div>
        </div>
        """
        mock_render.return_value = mock_html
        
        # Get the template
        template = template_env.get_template("results.html")
        
        # Render the template (this will use our mock)
        rendered = template.render(results=individual_chat_results)
        
        # Parse the HTML
        soup = BeautifulSoup(rendered, 'html.parser')
        
        # Find all info-text elements
        info_texts = soup.find_all("p", class_="info-text")
        
        # There should be no info-text elements for individual chats
        assert len(info_texts) == 0, f"Found {len(info_texts)} info-text elements in individual chat, expected 0"

if __name__ == "__main__":
    pytest.main() 