import csv
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# Data models
class OrderData(BaseModel):
    name: str
    order_details: str
    address: str
    timestamp: str

class ReturnData(BaseModel):
    name: str
    order_number: str
    reason: str
    timestamp: str

class IssueData(BaseModel):
    name: str
    issue_description: str
    timestamp: str

class CallbackData(BaseModel):
    name: str
    phone_number: str
    reason: str
    timestamp: str

class TrackOrderData(BaseModel):
    name: str
    order_number: str
    timestamp: str

def ensure_directory_exists(file_path: str):
    """Ensures the directory exists for the given file path."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_new_order(name: str, order_details: str, address: str):
    """Saves new order details to CSV file."""
    file_path = 'customerrecords/neworder.csv'
    ensure_directory_exists(file_path)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'name', 'order_details', 'address'])
    
    # Append new order
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, name, order_details, address])

def save_return_request(name: str, order_number: str, reason: str):
    """Saves return request details to CSV file."""
    file_path = 'customerrecords/orderreturns.csv'
    ensure_directory_exists(file_path)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'name', 'order_number', 'reason'])
    
    # Append new return request
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, name, order_number, reason])

def save_issue_report(name: str, issue_description: str):
    """Saves issue report details to CSV file."""
    file_path = 'customerrecords/issues.csv'
    ensure_directory_exists(file_path)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'name', 'issue_description'])
    
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, name, issue_description])

def save_callback_request(name: str, phone_number: str):
    """Saves callback request details to CSV file."""
    file_path = 'customerrecords/customerrecords.csv'
    ensure_directory_exists(file_path)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'name', 'phone_number', 'reason'])
    
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, name, phone_number, "Requested Callback"])

def save_track_order(name: str, order_number: str):
    """Saves order tracking details to CSV file."""
    file_path = 'customerrecords/trackorder.csv'
    ensure_directory_exists(file_path)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'name', 'order_number'])
    
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, name, order_number])

# API endpoints
@router.get("/sabineworders", response_model=List[OrderData])
async def get_new_orders():
    """Fetch all new orders from CSV"""
    orders = []
    file_path = 'customerrecords/neworder.csv'
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                orders.append(OrderData(**row))
    return orders

@router.get("/sabireturns", response_model=List[ReturnData])
async def get_returns():
    """Fetch all returns from CSV"""
    returns = []
    file_path = 'customerrecords/orderreturns.csv'
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                returns.append(ReturnData(**row))
    return returns

@router.get("/sabiissues", response_model=List[IssueData])
async def get_issues():
    """Returns all issue reports as JSON."""
    try:
        file_path = 'customerrecords/issues.csv'
        if not os.path.exists(file_path):
            return []
        
        issues = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                issues.append(IssueData(**row))
        return issues
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sabicallbacks", response_model=List[CallbackData])
async def get_callbacks():
    """Returns all callback requests as JSON."""
    try:
        file_path = 'customerrecords/customerrecords.csv'
        if not os.path.exists(file_path):
            return []
        
        callbacks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('reason') == "Requested Callback":
                    callbacks.append(CallbackData(**row))
        return callbacks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sabitracking", response_model=List[TrackOrderData])
async def get_tracking():
    """Fetch all tracking requests from CSV"""
    tracking = []
    file_path = 'customerrecords/trackorder.csv'
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                tracking.append(TrackOrderData(**row))
    return tracking 