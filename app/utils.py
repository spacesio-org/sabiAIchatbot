# app/utils.py

import csv
import os

def read_csv(file_path: str):
    """Reads a CSV file and returns its contents as a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def write_to_file(file_path: str, data: str):
    """Writes data to a specified file."""
    with open(file_path, mode='w', encoding='utf-8') as file:
        file.write(data)
