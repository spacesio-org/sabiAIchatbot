from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    app: str  # The app from which the user is making the request
    name: str  # The user's name
    address: str  # The user's delivery address
    phone_number: Optional[str]  # Optional: The user's phone number
    query: str  # The query the user is asking
    order_details: Optional[str]  # Optional: Order details if applicable

class QueryResponse(BaseModel):
    answer: str  # The response from the chatbot (e.g., confirmation message, order details)
