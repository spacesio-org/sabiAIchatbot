import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model and embedding model
llm = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Function to handle Trace queries
async def handle_trace_query(query, user_name):
    # Load Trace-specific documents
    docs_path = 'documents/trace'
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return "I apologize, but I don't have enough information about TRACE at the moment. Please try again later or contact support."
    
    try:
        # Load and combine all documents
        documents = []
        for filename in os.listdir(docs_path):
            if filename.endswith('.txt'):
                with open(os.path.join(docs_path, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        
        combined_docs = "\n\n".join(documents)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(combined_docs)
        
        if not text_chunks:
            return "I apologize, but I don't have enough information to answer your question about TRACE. Please contact support for assistance."

        # Create FAISS index
        vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(text_chunks))]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        )

        # Get response
        result = qa_chain.invoke({"query": query})
        return result["result"]

    except Exception as e:
        print(f"Error in trace service: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support."
