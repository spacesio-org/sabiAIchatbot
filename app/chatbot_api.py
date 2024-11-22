from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from services.sabi_service import handle_sabi_query
from services.trace_service import handle_trace_query
from services.katsu_service import handle_katsu_query
from functions.sabi_functions import router as sabi_router
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import shutil
import csv
from datetime import datetime
from typing import Optional
from app.models import FeedbackData
from scripts.improve_responses import update_training_data, improve_response


# Load environment variables
load_dotenv()

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Sabi's Chatbot API",
    description="""
    # Sabi Market Chatbot API

    This API provides chatbot functionality for Sabi Market's WhatsApp integration.

    ## Features
    * Order Processing: Place new orders with item quantities
    * Order Returns: Process return requests with order numbers
    * Issue Reporting: Submit customer issues and complaints
    * Callback Requests: Schedule customer service callbacks
    * Order Tracking: Track order status using order numbers
    * Document Upload: Upload training documents for each app

    ## Request Format
    All requests should include:
    - `app`: Service identifier ("sabi", "trace", or "katsu")
    - `name`: Customer's name
    - `query`: The customer's message
    - `address`: Delivery address (optional)
    - `phone_number`: Contact number (optional)

    ## Response Format
    All responses include:
    - `answer`: The chatbot's response message

    ## Document Upload
    Upload training documents using the `/upload-document` endpoint:
    - Supported apps: sabi, trace, katsu
    - File format: .txt files
    - Documents are stored in app-specific folders
    """,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Chatbot",
            "description": "Main chatbot interaction endpoints"
        },
        {
            "name": "Document Management",
            "description": "Upload and manage training documents"
        },
        {
            "name": "Sabi Market Operations",
            "description": "Data retrieval endpoints for customer records"
        },
        {
            "name": "Status",
            "description": "API health check endpoints"
        }
    ]
)

# Add this after initializing the FastAPI app
app.include_router(
    sabi_router,
    prefix="/sabi",
    tags=["Sabi Market Operations"],
    responses={404: {"description": "Not found"}},
)

# Add after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced request model with better documentation
class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The user's query or request",
        example="I want to order Milo (5 tins), Rice (2 bags)"
    )
    name: str = Field(
        ..., 
        description="Customer's name",
        example="John Doe"
    )
    app: str = Field(
        ..., 
        description="App choice (sabi, trace, or katsu)",
        example="sabi"
    )
    address: str = Field(
        ...,
        description="Customer's delivery address",
        example="123 Main Street, Lagos"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "I want to order Milo (5 tins), Rice (2 bags)",
                "name": "John Doe",
                "app": "sabi",
                "address": "123 Main Street, Lagos"
            }
        }

# Enhanced response model
class QueryResponse(BaseModel):
    answer: str = Field(
        ..., 
        description="The chatbot's response",
        example="Thank you for your order! We've saved your details and will process it right away."
    )

@app.post("/chatbot", 
    response_model=QueryResponse,
    tags=["Chatbot"],
    summary="Process a chatbot query",
    description="""
    Process a user query and return an appropriate response.
    
    Supported operations:
    - Place new orders
    - Request returns
    - Report issues
    - Request callbacks
    - Track orders
    
    The response will vary based on the type of query and the selected app.
    """
)
async def get_chatbot_response(query_request: QueryRequest):
    """
    Process a chatbot query with the following steps:
    1. Validate the request
    2. Route to appropriate service (Sabi, Trace, or Katsu)
    3. Process the query
    4. Return the response
    
    If an error occurs, returns a 500 error with details.
    """
    try:
        # Get initial response
        if query_request.app.lower() == "sabi":
            initial_answer = await handle_sabi_query(query_request.query, query_request.name, query_request.address)
        elif query_request.app.lower() == "trace":
            initial_answer = await handle_trace_query(query_request.query, query_request.name)
        elif query_request.app.lower() == "katsu":
            initial_answer = await handle_katsu_query(query_request.query, query_request.name)
        else:
            return QueryResponse(answer="Invalid app specified.")

        # Improve response if not a system message
        if not any(msg in initial_answer for msg in [
            "Thank you for submitting",
            "Please provide",
            "To process your"
        ]):
            improved_answer = await improve_response(
                query_request.app,
                query_request.query,
                initial_answer
            )
            return QueryResponse(answer=improved_answer)

        return QueryResponse(answer=initial_answer)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/", 
    tags=["Status"],
    summary="Check API Status",
    description="Returns the current status of the API"
)
async def read_root():
    """Check if the API is running."""
    return {
        "status": "online",
        "message": "AI Chatbot API is up and running!",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


# Initialize templates
templates = Jinja2Templates(directory="templates")

# UI route
@app.get("/ui", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/records", response_class=HTMLResponse)
async def customer_records(request: Request):
    return templates.TemplateResponse("customer_records.html", {"request": request})

@app.post("/upload-document",
    tags=["Document Management"],
    summary="Upload Training Document",
    description="""
    Upload a text document for training the chatbot.
    
    The document will be stored in the appropriate app folder:
    - sabi: documents/sabiMarket
    - trace: documents/trace
    - katsu: documents/katsu
    
    Only .txt files are supported.
    """
)
async def upload_document(
    app: str = Form(..., description="Target app (sabi, trace, or katsu)"),
    document: UploadFile = File(..., description="Text document to upload (.txt)")
):
    """Upload a document for training the chatbot"""
    try:
        app_folders = {
            "sabi": "documents/sabiMarket",
            "trace": "documents/trace",
            "katsu": "documents/katsu"
        }
        
        folder = app_folders.get(app.lower())
        if not folder:
            raise HTTPException(status_code=400, detail="Invalid application specified")
            
        if not document.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
            
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, document.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
            
        return {
            "message": f"Document successfully uploaded to {folder}",
            "filename": document.filename,
            "app": app
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/upload", response_class=HTMLResponse)
async def upload_ui(request: Request):
    return templates.TemplateResponse("document_upload.html", {"request": request})

@app.post("/feedback")
async def save_feedback(feedback: FeedbackData):
    try:
        feedback_file = f'feedback/{feedback.app}_feedback.csv'
        os.makedirs('feedback', exist_ok=True)
        
        is_new_file = not os.path.exists(feedback_file)
        mode = 'a'  # Always append
        
        with open(feedback_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(['timestamp', 'query', 'response', 'rating', 'comment'])
            writer.writerow([
                feedback.timestamp,
                feedback.query.replace(',', ' '),
                feedback.response.replace(',', ' '),
                '👍' if feedback.rating else '👎',
                feedback.comment or ""
            ])
        
        # Update training data only for positive feedback
        if feedback.rating:
            update_training_data(feedback.app, feedback.query, feedback.response)
        
        return {"status": "success"}
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        return {"status": "error", "detail": str(e)}