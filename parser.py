import pandas as pd
from datetime import datetime
import logging
import re
import os

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

# Common date formats in WhatsApp exports across different regions
DATE_FORMATS = [
    # European/Standard formats
    {'pattern': '%d/%m/%Y, %H:%M:%S', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2}'},
    {'pattern': '%d/%m/%y, %H:%M:%S', 'regex': r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}'},
    # US formats
    {'pattern': '%m/%d/%Y, %H:%M:%S', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2}'},
    {'pattern': '%m/%d/%y, %H:%M:%S', 'regex': r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}'},
    # ISO-like formats
    {'pattern': '%Y-%m-%d, %H:%M:%S', 'regex': r'\d{4}-\d{1,2}-\d{1,2}, \d{1,2}:\d{2}:\d{2}'},
    {'pattern': '%d-%m-%Y, %H:%M:%S', 'regex': r'\d{1,2}-\d{1,2}-\d{4}, \d{1,2}:\d{2}:\d{2}'},
    {'pattern': '%d-%m-%y, %H:%M:%S', 'regex': r'\d{1,2}-\d{1,2}-\d{2}, \d{1,2}:\d{2}:\d{2}'},
    # Formats with 12-hour clock
    {'pattern': '%d/%m/%Y, %I:%M:%S %p', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M'},
    {'pattern': '%m/%d/%Y, %I:%M:%S %p', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M'},
    # Formats without seconds
    {'pattern': '%d/%m/%Y, %H:%M', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}'},
    {'pattern': '%m/%d/%Y, %H:%M', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}'},
    # WhatsApp format without brackets (seen in error log)
    {'pattern': '%d/%m/%Y, %H:%M', 'regex': r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}'},
]

def parse_chat_log(txt_file_path: str, csv_file_path: str) -> None:
    """
    Parse a WhatsApp chat log text file and convert it to CSV format.
    
    Args:
        txt_file_path (str): Path to the input text file
        csv_file_path (str): Path to save the output CSV file
    """
    try:
        # Check if file exists
        if not os.path.exists(txt_file_path):
            raise FileNotFoundError(f"The file {txt_file_path} does not exist")
            
        # Check if file is empty
        if os.path.getsize(txt_file_path) == 0:
            raise ValueError(f"The file {txt_file_path} is empty")
        
        # Try different encodings if utf-8 fails
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        lines = None
        
        for encoding in encodings_to_try:
            try:
                with open(txt_file_path, 'r', encoding=encoding) as file:
                    lines = file.readlines()
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with {encoding} encoding, trying next...")
        
        if lines is None:
            raise ValueError(f"Could not read the file with any of the attempted encodings")
        
        logger.info("Read %d lines from %s", len(lines), txt_file_path)
        
        # Determine if the chat format uses brackets or not
        has_brackets = False
        no_brackets = False
        date_format = None
        
        # Check the first 20 non-empty lines to determine format
        checked_lines = 0
        for i in range(min(100, len(lines))):
            if not lines[i].strip():
                continue
                
            checked_lines += 1
            if checked_lines > 20:
                break
                
            # Check for bracketed format: [DD/MM/YYYY, HH:MM]
            if lines[i].startswith('['):
                has_brackets = True
                try:
                    datetime_str = lines[i][1:lines[i].index(']')]
                    for fmt in DATE_FORMATS:
                        if re.match(fmt['regex'], datetime_str):
                            try:
                                datetime.strptime(datetime_str, fmt['pattern'])
                                date_format = fmt['pattern']
                                logger.info(f"Detected bracketed date format: {date_format}")
                                break
                            except ValueError:
                                continue
                    if date_format:
                        break
                except (ValueError, IndexError):
                    continue
            
            # Check for non-bracketed format: DD/MM/YYYY, HH:MM - 
            elif re.match(r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}', lines[i]):
                no_brackets = True
                try:
                    # Extract date/time part (before the " - ")
                    parts = lines[i].split(' - ', 1)
                    if len(parts) >= 1:
                        datetime_str = parts[0].strip()
                        for fmt in DATE_FORMATS:
                            if re.match(fmt['regex'], datetime_str):
                                try:
                                    datetime.strptime(datetime_str, fmt['pattern'])
                                    date_format = fmt['pattern']
                                    logger.info(f"Detected non-bracketed date format: {date_format}")
                                    break
                                except ValueError:
                                    continue
                        if date_format:
                            break
                except (ValueError, IndexError):
                    continue
        
        if not date_format:
            # Sample the first few lines for debugging
            sample_lines = "\n".join(lines[:20])
            logger.error(f"Could not detect date format. Sample of file content:\n{sample_lines}")
            raise ValueError("Could not detect the date format in the chat log. Please check if this is a valid WhatsApp chat export.")
        
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
        
        # Track parsing statistics for better error reporting
        total_lines = len(lines)
        parsed_lines = 0
        skipped_lines = 0
        error_lines = 0
        
        for i, line in enumerate(lines):
            try:
                # Check if this is a new message line
                is_new_message = False
                
                if has_brackets and line.strip() and line.startswith('['):
                    is_new_message = True
                elif no_brackets and re.match(r'\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}', line):
                    is_new_message = True
                
                if is_new_message:
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
                        parsed_lines += 1
                    
                    # Start a new message
                    try:
                        if has_brackets:
                            # Bracketed format: [DD/MM/YYYY, HH:MM] Sender: Message
                            datetime_str = line[1:line.index(']')]
                            content = line[line.index(']')+2:]
                        else:
                            # Non-bracketed format: DD/MM/YYYY, HH:MM - Sender: Message
                            parts = line.split(' - ', 1)
                            if len(parts) < 2:
                                logger.debug(f"Skipping invalid line format at line {i+1}: {line.strip()}")
                                skipped_lines += 1
                                continue
                                
                            datetime_str = parts[0].strip()
                            content = parts[1].strip()
                        
                        # Parsing datetime with detected format
                        date_time = datetime.strptime(datetime_str, date_format)
                        
                        # Splitting sender and message
                        sender_message = content.split(':', 1)
                        if len(sender_message) < 2:
                            # This might be a system message or notification
                            logger.debug(f"Skipping system message at line {i+1}: {line.strip()}")
                            skipped_lines += 1
                            continue
                            
                        current_sender = sender_message[0].strip()
                        current_message = sender_message[1].strip()
                        current_date = date_time.strftime('%d/%m/%Y')
                        current_time = date_time.strftime('%H:%M:%S')
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing message format at line {i+1}: {line.strip()}\nError: {str(e)}")
                        error_lines += 1
                        continue
                    
                elif line.strip():
                    # This is a continuation of the previous message
                    if current_message:
                        current_message += " " + line.strip()
                
            except Exception as e:
                logger.warning(f"Unexpected error parsing line {i+1}: {line.strip()}\nError: {str(e)}")
                error_lines += 1
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
            parsed_lines += 1
        
        logger.info("Parsing statistics: Total lines: %d, Parsed messages: %d, Skipped: %d, Errors: %d", 
                   total_lines, parsed_lines, skipped_lines, error_lines)
        
        if not dates:
            # Provide more detailed error information
            sample_lines = "\n".join(lines[:20])
            logger.error(f"No messages were successfully parsed. Sample of file content:\n{sample_lines}")
            raise ValueError(
                "No messages were successfully parsed from the file. This could be due to:\n"
                "1. The file is not a WhatsApp chat export\n"
                "2. The chat format is different from what's expected\n"
                "3. The file contains only system messages\n"
                "Please check if this is a valid WhatsApp chat export."
            )
            
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