from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class QueryRequest(BaseModel):
    app: str  # The app from which the user is making the request
    name: str  # The user's name
    address: str  # The user's delivery address
    phone_number: Optional[str]  # Optional: The user's phone number
    query: str  # The query the user is asking
    order_details: Optional[str]  # Optional: Order details if applicable

    class Config:
        json_schema_extra = {
            "example": {
                "app": "sabi",
                "name": "John Doe",
                "address": "123 Main St",
                "query": "I want to order 5 bags of rice"
            }
        }

class QueryResponse(BaseModel):
    answer: str  # The response from the chatbot (e.g., confirmation message, order details)

class FeedbackData(BaseModel):
    query: str
    response: str
    rating: bool  # True for positive, False for negative
    comment: Optional[str] = ""
    timestamp: str
    app: str  # sabi/trace/katsu

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are your delivery times?",
                "response": "We deliver within 24-48 hours.",
                "rating": True,  # 👍
                "comment": "",
                "timestamp": "2024-03-19T10:30:00.000Z",
                "app": "sabi"
            }
        }
