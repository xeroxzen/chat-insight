import os
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_mock_results(is_group=True):
    """Create mock results for testing the template"""
    return {
        "is_group": is_group,
        "participant_count": 5,
        "total_messages": 1000,
        "analysis_period": ("2023-01-01", "2023-12-31"),
        "first_message_date": datetime.strptime("2023-01-01", "%Y-%m-%d"),
        "last_message_date": datetime.strptime("2023-12-31", "%Y-%m-%d"),
        "avg_messages_per_day": 25.5,
        "top_participants": {
            "User1": 500,
            "User2": 300,
            "User3": 200
        },
        "most_active_day": datetime.strptime("2023-06-15", "%Y-%m-%d"),
        "most_active_time": 20,
        "emojis": {"😊": 10, "👍": 5},
        "links": {"http://example.com": 2},
        "sentiment_analysis": {"positive": 60, "neutral": 30, "negative": 10},
        "group_dynamics": {
            "most_active_member": "User1",
            "least_active_member": "User3",
            "peak_hours": {"User1": 20, "User2": 21, "User3": 19}
        },
        "interaction_network": [
            {"Sender": "User1", "Next_Sender": "User2", "weight": 50},
            {"Sender": "User2", "Next_Sender": "User1", "weight": 40},
            {"Sender": "User1", "Next_Sender": "User3", "weight": 30}
        ],
        "media_analysis": {
            "total_media_count": 50,
            "media_by_type": {"image": 30, "video": 15, "audio": 5},
            "media_sharing_patterns": {
                "total_media_count": 50,
                "media_by_type": {"image": 30, "video": 15, "audio": 5},
                "most_shared_media_type": {
                    "User1": {"type": "image", "count": 20},
                    "User2": {"type": "video", "count": 10}
                }
            }
        },
        "visualization_paths": {
            "wordcloud": "/visuals/user123/wordcloud.png",
            "average_messages_per_day": "/visuals/user123/average_messages_per_day.png",
            "first_message_sender": "/visuals/user123/first_message_sender.png",
            "emoji_usage_by_sender": "/visuals/user123/emoji_usage_by_sender.png",
            "sentiment_counts": "/visuals/user123/sentiment_counts.png",
            "links_shared_by_sender": "/visuals/user123/links_shared_by_sender.png",
            "top_emojis": "/visuals/user123/top_emojis.png",
            "messages_count_per_sender": "/visuals/user123/messages_count_per_sender.png",
            "message_activity_over_time": "/visuals/user123/message_activity_over_time.png",
            "message_activity_heatmap": "/visuals/user123/message_activity_heatmap.png",
            "message_length_distribution": "/visuals/user123/message_length_distribution.png",
            "response_time_distribution": "/visuals/user123/response_time_distribution.png",
            "daily_pattern": "/visuals/user123/daily_pattern.png",
            "group_participation": "/visuals/user123/group_participation.png",
            "group_interaction_network": "/visuals/user123/group_interaction_network.png",
            "media_dashboard": "/visuals/user123/media_dashboard.png",
            "media_types_distribution": "/visuals/user123/media_types_distribution.png",
            "media_by_sender": "/visuals/user123/media_by_sender.png",
            "media_over_time": "/visuals/user123/media_over_time.png",
            "call_patterns": "/visuals/user123/call_patterns.png",
            "call_duration_by_type": "/visuals/user123/call_duration_by_type.png",
            "avg_call_duration_by_type": "/visuals/user123/avg_call_duration_by_type.png",
            "media_sharing_patterns": "/visuals/user123/media_sharing_patterns.png"
        }
    }

def render_template():
    """Render the results template with mock data"""
    # Set up Jinja2 environment
    template_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    
    # Load the template
    template = env.get_template("results.html")
    
    # Create mock results
    results = create_mock_results(is_group=True)
    
    # Render the template
    html = template.render(results=results)
    
    # Save the rendered HTML to a file
    with open("test_results.html", "w") as f:
        f.write(html)
    
    print(f"Template rendered to test_results.html")
    
    # Print the visualization paths for debugging
    print("\nVisualization paths:")
    for key, path in results["visualization_paths"].items():
        print(f"{key}: {path}")
    
    # Check if group-specific paths are included
    group_paths = ["group_participation", "group_interaction_network"]
    for path in group_paths:
        if path in results["visualization_paths"]:
            print(f"\nGroup path found: {path} = {results['visualization_paths'][path]}")
        else:
            print(f"\nGroup path missing: {path}")

if __name__ == "__main__":
    render_template() 