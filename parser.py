import csv
import re
import pandas as pd

def parse_chat_log(txt_file_path, csv_file_path):
    pattern = re.compile(r"\[(\d{1,2}/\d{1,2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.+?): (.+)")
    data = []

    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        lines = txt_file.readlines()

    current_message = ""
    current_date = ""
    current_time = ""
    current_sender = ""

    for line in lines:
        match = pattern.match(line)
        if match:
            if current_message:
                data.append([current_date, current_time, current_sender, current_message])
            current_date, current_time, current_sender, current_message = match.groups()
        else:
            current_message += "\n" + line.strip()

    if current_message:
        data.append([current_date, current_time, current_sender, current_message])

    df = pd.DataFrame(data, columns=["Date", "Time", "Sender", "Message"])
    df.to_csv(csv_file_path, index=False)