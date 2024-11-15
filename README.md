# Sabi Chatbot System

A multi-service chatbot system that handles customer interactions for Sabi Market, Trace, and Katsu Bank services.

## Table of Contents

- [System Overview](#system-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)

## System Overview

The Sabi Chatbot System is a FastAPI-based application that provides intelligent chatbot functionality for three distinct services:

- Sabi Market: E-commerce and order management
- Trace: Product tracking and logistics
- Katsu Bank: Banking services

The system uses OpenAI's language models for natural language processing and maintains separate training documents for each service.

## Features

- Multi-service support (Sabi Market, Trace, Katsu Bank)
- Real-time chat interface
- Document upload system for training data
- Customer record management
- Order tracking
- Return processing
- Callback request handling
- Automated response improvement system

## Prerequisites

- Python 3.8+
- OpenAI API key
- Virtual environment tool (venv recommended)
- Modern web browser
- Git

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cbotsabi.git
cd cbotsabi

```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. Install dependencies:

```bash
pip install -r requirements.txt

```

4. Create .env file:

```bash
OPENAI_API_KEY=your_api_key_here

```

5. Create required directories:

```bash
mkdir -p documents/sabiMarket documents/trace documents/katsu

```

6. Start the application:

```bash
python app/main.py

```

## API Documentation

### Main Endpoints

1. Chat Interface

```python
POST /chatbot
```

Request body:

```json
{
  "query": "string",
  "name": "string",
  "app": "string",
  "address": "string"
}
```

2. Document Upload

```python
POST /upload-document
```

Form data:

- app: string (sabi/trace/katsu)
- document: file (.txt)

3. Customer Records

```python
GET /sabi/sabineworders
GET /sabi/sabireturns
GET /sabi/sabitracking
GET /sabi/sabicallbacks
```

### Web Interfaces

- Chat UI: `/ui`
- Document Upload: `/upload`
- Customer Records: `/records`

## System Architecture

### Components

1. Frontend Layer

- HTML/CSS/JavaScript interface
- Real-time chat functionality
- Document upload interface
- Customer records dashboard

2. API Layer

- FastAPI application
- Request handling and routing
- Response processing
- File management

3. Service Layer

- Sabi Market service
- Trace service
- Katsu Bank service
- Document processing

4. Data Layer

- Document storage
- Customer records
- Training data
- Feedback system

### Data Flow

1. User Input Flow:

```
User Input → Frontend → API → Service Layer → OpenAI Processing → Response
```

2. Document Training Flow:

```
Upload → Document Processing → Vector Storage → Training Data Update
```

3. Customer Records Flow:

```
Database → API Endpoints → Frontend Dashboard
```

## Directory Structure

```
cbotsabi/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── chatbot_api.py
│   ├── models.py
│   └── utils.py
├── services/
│   ├── sabi_service.py
│   ├── trace_service.py
│   └── katsu_service.py
├── templates/
│   ├── index.html
│   ├── document_upload.html
│   └── customer_records.html
├── documents/
│   ├── sabiMarket/
│   ├── trace/
│   └── katsu/
├── scripts/
│   └── improve_responses.py
├── requirements.txt
└── .env
```
