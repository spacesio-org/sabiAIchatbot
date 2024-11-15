# app/main.py

import uvicorn
from fastapi import FastAPI
from chatbot_api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
