# app/main.py

import uvicorn
from fastapi import FastAPI
from app.chatbot_api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)