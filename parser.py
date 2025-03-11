import pandas as pd
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Media message patterns - expanded with more variations to catch all formats
MEDIA_PATTERNS = {
    'image': r'image omitted|‎image omitted|<Media omitted>|<image omitted>',
    'video': r'video omitted|‎video omitted|<Media omitted>|<video omitted>',
    'audio': r'audio omitted|‎audio omitted|<Media omitted>|<audio omitted>',
    'voice_call': r'Voice call|‎Voice call|Voice call, ‎Answered on other device|voice call',
    'video_call': r'Video call|‎Video call|‎Video call, ‎Answered on other device|video call',
    'missed_voice_call': r'Missed voice call, Tap to call back|‎Missed voice call, ‎Click to call back|Voice call, No answer|Silenced voice call|missed voice call',
    'missed_video_call': r'Missed video call, Tap to call back|‎Missed video call, ‎Click to call back|Video call, No answer|Silenced video call|missed video call',
    'sticker': r'sticker omitted|‎Sticker omitted|<Media omitted>|<sticker omitted>'
}

def parse_chat_log(txt_file_path: str, csv_file_path: str) -> None:
    """
    Parse a WhatsApp chat log text file and convert it to CSV format.
    
    Args:
        txt_file_path (str): Path to the input text file
        csv_file_path (str): Path to save the output CSV file
    """
    try:
        # Reading the text file
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        logger.info("Read %d lines from %s", len(lines), txt_file_path)
        
        # Lists to store parsed data
        dates = []
        times = []
        senders = []
        messages = []
        media_types = []
        
        # Parsing each line
        current_message = ""
        current_sender = ""
        current_date = None
        current_time = None
        
        for i, line in enumerate(lines):
            try:
                # Expected format: [DD/MM/YYYY, HH:mm:ss] Sender: Message
                if line.strip() and line.startswith('['):
                    # If we have a message in progress, save it first
                    if current_message and current_sender and current_date and current_time:
                        # Detect media type with case-insensitive matching
                        media_type = 'text'  # Default type
                        for m_type, pattern in MEDIA_PATTERNS.items():
                            if re.search(pattern, current_message, re.IGNORECASE):
                                media_type = m_type
                                break
                        
                        # Detect call duration if present
                        call_duration_match = re.search(r'(voice|video) call \((\d+):(\d+)\)', current_message, re.IGNORECASE)
                        if call_duration_match:
                            call_type = call_duration_match.group(1).lower()
                            media_type = f"{call_type}_call"
                        
                        # Store the completed message
                        dates.append(current_date)
                        times.append(current_time)
                        senders.append(current_sender)
                        messages.append(current_message)
                        media_types.append(media_type)
                    
                    # Start a new message
                    # Splitting datetime and message content
                    datetime_str = line[1:line.index(']')]
                    content = line[line.index(']')+2:]
                    
                    # Parsing datetime - handling both YYYY and YY formats
                    try:
                        # Trying full year format first (DD/MM/YYYY)
                        date_time = datetime.strptime(datetime_str, '%d/%m/%Y, %H:%M:%S')
                    except ValueError:
                        try:
                            # Try two-digit year format (DD/MM/YY)
                            date_time = datetime.strptime(datetime_str, '%d/%m/%y, %H:%M:%S')
                        except ValueError as e:
                            logger.warning("Could not parse date: %s", datetime_str)
                            raise e
                    
                    # Splitting sender and message
                    sender_message = content.split(':', 1)
                    if len(sender_message) < 2:
                        # This might be a system message or continuation
                        continue
                        
                    current_sender = sender_message[0].strip()
                    current_message = sender_message[1].strip()
                    current_date = date_time.strftime('%d/%m/%Y')
                    current_time = date_time.strftime('%H:%M:%S')
                    
                elif line.strip():
                    # This is a continuation of the previous message
                    if current_message:
                        current_message += " " + line.strip()
                
            except (ValueError, IndexError) as e:
                logger.warning("Error parsing line %d: %s\nError: %s", i+1, line.strip(), str(e))
                continue
        
        # Don't forget to add the last message
        if current_message and current_sender and current_date and current_time:
            # Detect media type
            media_type = 'text'  # Default type
            for m_type, pattern in MEDIA_PATTERNS.items():
                if re.search(pattern, current_message, re.IGNORECASE):
                    media_type = m_type
                    break
            
            # Detect call duration if present
            call_duration_match = re.search(r'(voice|video) call \((\d+):(\d+)\)', current_message, re.IGNORECASE)
            if call_duration_match:
                call_type = call_duration_match.group(1).lower()
                media_type = f"{call_type}_call"
            
            # Store the completed message
            dates.append(current_date)
            times.append(current_time)
            senders.append(current_sender)
            messages.append(current_message)
            media_types.append(media_type)
        
        logger.info("Successfully parsed %d messages", len(dates))
        
        if not dates:
            raise ValueError("No messages were successfully parsed from the file")
            
        # Creating DataFrame with capitalized column names and formatted dates
        df = pd.DataFrame({
            'Date': dates,  # Already formatted as DD/MM/YYYY strings
            'Time': times,  # Already formatted as HH:MM:SS strings
            'Sender': senders,
            'Message': messages,
            'MediaType': media_types
        })
        
        # Log media type counts for debugging
        media_counts = df['MediaType'].value_counts()
        logger.info("Media type counts: %s", media_counts.to_dict())
        
        # Log media counts by sender for debugging
        for sender in df['Sender'].unique():
            sender_media = df[df['Sender'] == sender]['MediaType'].value_counts()
            logger.info("Media counts for %s: %s", sender, sender_media.to_dict())
        
        # Saving to CSV
        df.to_csv(csv_file_path, index=False)
        logger.info("Saved parsed data to %s", csv_file_path)
        
    except Exception as e:
        logger.error("Error processing file: %s", str(e))
        raise