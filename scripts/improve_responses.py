import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
from datetime import datetime
import csv

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model - using GPT-3.5-turbo for better performance/cost ratio
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=300,
    api_key=openai_api_key
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Prompt template for response improvement
IMPROVEMENT_PROMPT = PromptTemplate(
    input_variables=["query", "original_response", "feedback_data"],
    template="""
    You are improving chatbot responses for an e-commerce platform. 
    
    Original Query: {query}
    Original Response: {original_response}
    
    Previous feedback and successful responses:
    {feedback_data}
    
    Please generate an improved response that:
    1. Maintains accuracy and relevance
    2. Uses a consistent, professional tone
    3. Includes all necessary information
    4. Is concise yet complete
    5. Follows the patterns of highly-rated responses
    
    Improved response:
    """
)

def load_feedback_data(app: str) -> list:
    """Load and parse feedback data for a specific app"""
    feedback_file = f'feedback/{app}_feedback.csv'
    if not os.path.exists(feedback_file):
        return []
    
    feedback_responses = []
    try:
        with open(feedback_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feedback_responses.append({
                    'query': row['query'],
                    'response': row['response'],
                    'rating': row['rating']
                })
    except Exception as e:
        print(f"Error loading feedback data: {e}")
    
    return feedback_responses

async def improve_response(app: str, query: str, original_response: str) -> str:
    """Generate an improved response based on feedback data"""
    try:
        # Load relevant feedback data
        feedback_data = load_feedback_data(app)
        
        # Format feedback data for prompt
        formatted_feedback = "\n".join([
            f"Query: {f['query']}\n"
            f"Successful Response: {f['response']}\n"
            f"Rating: {f['rating']}\n"
            for f in feedback_data[-5:]  # Use last 5 successful responses
        ])
        
        # Create improvement chain using LCEL syntax
        chain = IMPROVEMENT_PROMPT | llm
        
        # Generate improved response
        result = await chain.ainvoke({
            "query": query,
            "original_response": original_response,
            "feedback_data": formatted_feedback
        })
        
        # Save the improved response for future learning
        save_improved_response(app, query, result.content)
        
        return result.content.strip()
        
    except Exception as e:
        print(f"Error improving response: {e}")
        return original_response  # Fallback to original response if improvement fails

def save_improved_response(app: str, query: str, response: str):
    """Save improved responses for future training"""
    improved_file = f'training_data/{app}_improved_responses.jsonl'
    os.makedirs('training_data', exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "improved_response": response
    }
    
    with open(improved_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def update_training_data(app: str, query: str, response: str):
    """Update training data with new successful interactions"""
    training_file = f'training_data/{app}_training.jsonl'
    os.makedirs('training_data', exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    }
    
    with open(training_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def analyze_feedback(app: str):
    """Analyze feedback data to identify patterns in successful responses"""
    feedback_data = load_feedback_data(app)
    if not feedback_data:
        print(f"No feedback data found for {app}")
        return
    
    positive_feedback = [f for f in feedback_data if f['rating'] == '👍']
    negative_feedback = [f for f in feedback_data if f['rating'] == '👎']
    
    print(f"Analysis for {app}:")
    print(f"Total feedback entries: {len(feedback_data)}")
    print(f"Positive feedback: {len(positive_feedback)}")
    print(f"Negative feedback: {len(negative_feedback)}")

if __name__ == "__main__":
    apps = ['sabi', 'trace', 'katsu']
    for app in apps:
        analyze_feedback(app)
        update_training_data(app) 