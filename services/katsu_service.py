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

async def handle_katsu_query(query, user_name): 
    # Load Katsu-specific documents and process the query
    documents = load_documents_for_app()
    text_chunks = limit_content_size(documents)

    # Create FAISS index from text chunks
    faiss_index = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(text_chunks))]
    )
    
    retriever = faiss_index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    result = qa_chain.invoke(query)
    return result.get('result', 'Sorry, no result found.')

def load_documents_for_app():
    """Load all documents from the Katsu folder"""
    folder = "documents/katsu"
    if not os.path.exists(folder):
        return []
        
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                
    return documents

def limit_content_size(content_list, max_tokens=2000):
    """Limit content size while preserving complete Q&A pairs"""
    if not content_list:
        return []
        
    # Join all content with newlines to preserve formatting
    combined_content = "\n".join(content_list)
    
    # Split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split text into chunks
    texts = text_splitter.split_text(combined_content)
    
    return texts
