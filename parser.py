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

# Common WhatsApp system messages to ignore
SYSTEM_MESSAGE_PATTERNS = [
    r"Messages and calls are end-to-end encrypted",
    r"Tap to learn more",
    r"This chat is with a business account",
    r"Click to learn more",
    r".*security code changed.*",
    r".*end-to-end encrypted.*",
    r".*changed their phone number.*",
    r".*changed the group description.*",
    r".*changed the subject.*",
    r".*added.*to this group.*",
    r".*left.*",
    r".*removed.*",
    r".*created group.*",
    r".*changed the group icon.*",
    r".*deleted this message.*",
    r".*changed the group name.*",
    r".*changed their phone number to.*",
    r".*This message was deleted.*",
    r".*registered as a business account.*",
    r".*changed the group settings.*",
]

# Compile the patterns for better performance
SYSTEM_MESSAGE_REGEX = re.compile('|'.join(SYSTEM_MESSAGE_PATTERNS), re.IGNORECASE)

# Updated date formats with more flexible patterns
DATE_FORMATS = [
    # Format with brackets - 24-hour time with seconds
    {
        'pattern': '%d/%m/%Y, %H:%M:%S',
        'regex': r'\[(\d{1,2}/\d{1,2}/\d{4},\s*\d{2}:\d{2}:\d{2})\]',
        'example': '[31/01/2021, 20:43:49]'
    },
    # Format without brackets - 24-hour time
    {
        'pattern': '%d/%m/%Y, %H:%M',
        'regex': r'(\d{1,2}/\d{1,2}/\d{4},\s*\d{2}:\d{2})',
        'example': '31/01/2021, 20:43'
    },
    # US format with AM/PM
    {
        'pattern': '%d/%m/%y, %I:%M %p',
        'regex': r'(\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*[AP]M)',
        'example': '31/01/21, 8:43 PM'
    }
]

def detect_date_format(datetime_str: str) -> dict:
    """
    Detect the correct date format from the string.
    
    Args:
        datetime_str (str): The datetime string to analyze
        
    Returns:
        dict: The matching date format dictionary or None if no match
    """
    for date_format in DATE_FORMATS:
        if re.match(date_format['regex'], datetime_str):
            return date_format
    return None

def find_chat_file(input_path: str) -> str:
    """
    Find the chat file whether it's a direct .txt file or inside a folder.
    
    Args:
        input_path (str): Path to either a .txt file or a folder containing the chat
    
    Returns:
        str: Path to the actual chat .txt file
    """
    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        return input_path
    
    if os.path.isdir(input_path):
        # Look for .txt files in the directory
        txt_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
        if txt_files:
            # Prefer files with 'chat' in the name
            chat_files = [f for f in txt_files if 'chat' in f.lower()]
            if chat_files:
                return os.path.join(input_path, chat_files[0])
            return os.path.join(input_path, txt_files[0])
    
    raise FileNotFoundError(f"No chat file found at {input_path}")

def parse_datetime_line(line: str) -> tuple:
    """
    Parse a line to extract datetime and remaining content.
    
    Args:
        line (str): A line from the chat log
    
    Returns:
        tuple: (datetime_obj, remaining_content, format_used) or (None, None, None) if parsing fails
    """
    line = line.strip()
    
    # Try with square brackets
    bracket_match = re.match(r'\[(.*?)\](.*)', line)
    if bracket_match:
        datetime_str = bracket_match.group(1).strip()
        content = bracket_match.group(2).strip()
        full_datetime = f"[{datetime_str}]"
    else:
        # Try with hyphen separator
        parts = line.split(' - ', 1)
        if len(parts) == 2:
            datetime_str = parts[0].strip()
            content = parts[1].strip()
            full_datetime = datetime_str
        else:
            return None, None, None
    
    # Detect the date format
    date_format = detect_date_format(full_datetime)
    if not date_format:
        return None, None, None
        
    try:
        # Parse the datetime string
        if '[' in full_datetime:
            datetime_str = datetime_str  # Keep the original string without brackets
        
        # Extract day, month, year from the datetime string
        date_parts = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', datetime_str)
        if not date_parts:
            return None, None, None
            
        first_num, second_num, year = map(int, date_parts.groups())
        
        # Determine if the date is in DD/MM or MM/DD format
        # Rule: If first number is > 12, it must be a day
        # If second number is > 12, it must be a day
        # Otherwise, assume DD/MM format
        if first_num > 12:
            # Must be DD/MM format
            day, month = first_num, second_num
        elif second_num > 12:
            # Must be MM/DD format
            month, day = first_num, second_num
        else:
            # Assume DD/MM format as per standard
            day, month = first_num, second_num
            
        # Reconstruct the datetime string in DD/MM format
        time_part = datetime_str[datetime_str.find(','):]  # Get everything after the date
        formatted_date = f"{day:02d}/{month:02d}/{year}{time_part}"
        
        # Parse with the correct format
        datetime_obj = datetime.strptime(formatted_date, date_format['pattern'])
        
        # Validate the date is not in the future
        if datetime_obj > datetime.now():
            # If date is in future, try swapping day/month
            day, month = month, day
            formatted_date = f"{day:02d}/{month:02d}/{year}{time_part}"
            datetime_obj = datetime.strptime(formatted_date, date_format['pattern'])
            
            # If still in future, this might be an error
            if datetime_obj > datetime.now():
                logger.warning(f"Date appears to be in the future: {formatted_date}")
        
        return datetime_obj, content, date_format
                
    except Exception as e:
        logger.debug(f"Failed to parse date '{datetime_str}': {str(e)}")
        return None, None, None

def format_date(datetime_obj: datetime, original_format: dict) -> str:
    """
    Format the date consistently in DD/MM/YYYY format.
    
    Args:
        datetime_obj (datetime): The datetime object to format
        original_format (dict): The original format dictionary
        
    Returns:
        str: The formatted date string in DD/MM/YYYY format
    """
    return datetime_obj.strftime('%d/%m/%Y')

def is_system_message(message: str) -> bool:
    """
    Check if a message is a system message that should be ignored.
    
    Args:
        message (str): The message to check
        
    Returns:
        bool: True if the message is a system message, False otherwise
    """
    return bool(SYSTEM_MESSAGE_REGEX.match(message))

def parse_message_content(content: str) -> tuple:
    """
    Parse the content part of a message to extract sender and message.
    Skip system messages.
    
    Args:
        content (str): The content part of the message
    
    Returns:
        tuple: (sender, message) or (None, None) for system messages
    """
    # First check if it's a system message
    if is_system_message(content):
        return None, None
        
    # Try different separator patterns
    for separator in [':', ' - ']:
        parts = content.split(separator, 1)
        if len(parts) == 2:
            sender, message = parts[0].strip(), parts[1].strip()
            # Double check the message part isn't a system message
            if not is_system_message(message):
                return sender, message
    
    # If no valid message format found, return None
    return None, None

def parse_chat_log(txt_file_path: str, csv_file_path: str) -> None:
    """
    Parse a WhatsApp chat log text file and convert it to CSV format.
    
    Args:
        txt_file_path (str): Path to the input text file or folder
        csv_file_path (str): Path to save the output CSV file
    """
    try:
        # Find the actual chat file
        actual_file_path = find_chat_file(txt_file_path)
        logger.info(f"Processing chat file: {actual_file_path}")
        
        # Check if file is empty
        if os.path.getsize(actual_file_path) == 0:
            raise ValueError(f"The file {actual_file_path} is empty")
        
        # Try different encodings if utf-8 fails
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        lines = None
        
        for encoding in encodings_to_try:
            try:
                with open(actual_file_path, 'r', encoding=encoding) as file:
                    lines = file.readlines()
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with {encoding} encoding, trying next...")
        
        if lines is None:
            raise ValueError(f"Could not read the file with any of the attempted encodings")
        
        # Lists to store parsed data
        dates = []
        times = []
        senders = []
        messages = []
        media_types = []
        
        # Track parsing statistics
        total_lines = len(lines)
        parsed_lines = 0
        skipped_lines = 0
        error_lines = 0
        
        # Debug: Print first few lines
        logger.debug("First few lines of the file:")
        for i in range(min(5, len(lines))):
            logger.debug(f"Line {i+1}: {lines[i].strip()}")
        
        current_message = None
        current_sender = None
        current_datetime = None
        current_format = None
        
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as a new message
                datetime_obj, content, format_used = parse_datetime_line(line)
                
                if datetime_obj:
                    # This is a new message
                    
                    # Save the previous message if exists and not a system message
                    if current_message is not None and current_sender is not None:
                        media_type = 'text'
                        for m_type, pattern in MEDIA_PATTERNS.items():
                            if re.search(pattern, current_message, re.IGNORECASE):
                                media_type = m_type
                                break
                        
                        dates.append(format_date(current_datetime, current_format))
                        times.append(current_datetime.strftime('%H:%M:%S'))
                        senders.append(current_sender)
                        messages.append(current_message)
                        media_types.append(media_type)
                        parsed_lines += 1
                    
                    # Parse the new message
                    current_datetime = datetime_obj
                    current_format = format_used
                    current_sender, current_message = parse_message_content(content)
                    
                else:
                    # This is a continuation of the previous message
                    if current_message is not None and current_sender is not None:
                        current_message += " " + line
                        logger.debug(f"Added continuation line: {line}")
                    else:
                        skipped_lines += 1
                        logger.debug(f"Skipped line {i+1}: {line}")
            
            except Exception as e:
                logger.warning(f"Unexpected error at line {i+1}: {line}\nError: {str(e)}")
                error_lines += 1
                continue
        
        # Don't forget to add the last message if it's not a system message
        if current_message is not None and current_sender is not None:
            media_type = 'text'
            for m_type, pattern in MEDIA_PATTERNS.items():
                if re.search(pattern, current_message, re.IGNORECASE):
                    media_type = m_type
                    break
            
            dates.append(format_date(current_datetime, current_format))
            times.append(current_datetime.strftime('%H:%M:%S'))
            senders.append(current_sender)
            messages.append(current_message)
            media_types.append(media_type)
            parsed_lines += 1
        
        if not dates:
            logger.error("No messages were parsed from the file!")
            sample_lines = "\n".join(lines[:10])
            logger.error(f"Sample of file content:\n{sample_lines}")
            raise ValueError("No messages were parsed from the file. Please check the format.")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'Date': dates,
            'Time': times,
            'Sender': senders,
            'Message': messages,
            'MediaType': media_types
        })
        
        # Convert dates to datetime using DD/MM/YYYY format and validate
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Check for dates in the future and try to fix them
        future_dates = df['Date'] > datetime.now()
        if future_dates.any():
            logger.warning(f"Found {future_dates.sum()} dates in the future. Attempting to fix...")
            
            # For future dates, try swapping day and month
            future_dates_df = df[future_dates].copy()
            future_dates_df['Date'] = future_dates_df['Date'].apply(
                lambda x: x.replace(day=x.month, month=x.day) if x > datetime.now() else x
            )
            
            # Update the original dataframe with fixed dates
            df.loc[future_dates, 'Date'] = future_dates_df['Date']
        
        # Sort by date and time
        df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
        df = df.sort_values('DateTime')
        df = df.drop('DateTime', axis=1)
        
        # Save to CSV with formatted dates in DD/MM/YYYY format
        df.to_csv(csv_file_path, index=False, date_format='%d/%m/%Y')
        
        logger.info(f"Successfully parsed {parsed_lines} messages. Skipped: {skipped_lines}, Errors: {error_lines}")
        logger.info(f"Found {len(df['Sender'].unique())} unique senders: {', '.join(df['Sender'].unique())}")
        
    except Exception as e:
        logger.error("Error processing file: %s", str(e))
        raise