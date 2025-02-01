import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_chat_log(txt_file_path: str, csv_file_path: str) -> None:
    """
    Parse a WhatsApp chat log text file and convert it to CSV format.
    
    Args:
        txt_file_path (str): Path to the input text file
        csv_file_path (str): Path to save the output CSV file
    """
    try:
        # Read the text file
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        logger.info(f"Read {len(lines)} lines from {txt_file_path}")
        
        # Lists to store parsed data
        dates = []
        times = []
        senders = []
        messages = []
        
        # Parse each line
        for i, line in enumerate(lines):
            try:
                # Expected format: [DD/MM/YYYY, HH:mm:ss] Sender: Message
                if not line.strip() or not line.startswith('['): 
                    continue
                    
                # Split datetime and message content
                datetime_str = line[1:line.index(']')]
                content = line[line.index(']')+2:]
                
                # Parse datetime - handle both YYYY and YY formats
                try:
                    # Try full year format first (DD/MM/YYYY)
                    date_time = datetime.strptime(datetime_str, '%d/%m/%Y, %H:%M:%S')
                except ValueError:
                    try:
                        # Try two-digit year format (DD/MM/YY)
                        date_time = datetime.strptime(datetime_str, '%d/%m/%y, %H:%M:%S')
                    except ValueError as e:
                        logger.warning(f"Could not parse date: {datetime_str}")
                        raise e
                
                # Split sender and message
                sender_message = content.split(':', 1)
                if len(sender_message) < 2:
                    continue
                    
                sender = sender_message[0].strip()
                message = sender_message[1].strip()
                
                # Store parsed data - format date as DD/MM/YYYY
                dates.append(date_time.strftime('%d/%m/%Y'))
                times.append(date_time.strftime('%H:%M:%S'))
                senders.append(sender)
                messages.append(message)
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {i+1}: {line.strip()}\nError: {str(e)}")
                continue
        
        logger.info(f"Successfully parsed {len(dates)} messages")
        
        if not dates:
            raise ValueError("No messages were successfully parsed from the file")
            
        # Create DataFrame with capitalized column names and formatted dates
        df = pd.DataFrame({
            'Date': dates,  # Already formatted as DD/MM/YYYY strings
            'Time': times,  # Already formatted as HH:MM:SS strings
            'Sender': senders,
            'Message': messages
        })
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        logger.info(f"Saved parsed data to {csv_file_path}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise