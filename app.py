# import faiss
# import pickle
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI, Form, Request, UploadFile, File
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# import numpy as np
# import uvicorn
# import json
# from typing import List, Dict, Optional
# from datetime import datetime
# import subprocess
# import re
# import PyPDF2
# import io
# import atexit
# import shutil

# INDEX_DIR = Path("index")
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# app = FastAPI(title="Simple RAG Chatbot")

# def cleanup_on_exit():
#     shutil.rmtree(INDEX_DIR, ignore_errors=True)
#     print("🗑️ Index deleted")

# atexit.register(cleanup_on_exit)

# # Global state
# sentence_model = None
# index = None
# texts = None
# metadata = None

# # Store conversation history per session
# conversations: Dict[str, List[Dict]] = {}

# def initialize_components():
#     global sentence_model, index, texts, metadata
    
#     if sentence_model is not None:
#         return
    
#     print("🚀 Loading model and index...")
#     sentence_model = SentenceTransformer(EMBED_MODEL_NAME)
    
#     if not (INDEX_DIR / "faiss.index").exists():
#         print("📦 Creating new empty index...")
#         INDEX_DIR.mkdir(parents=True, exist_ok=True)
#         d = 384
#         index = faiss.IndexFlatL2(d)
#         texts = []
#         metadata = []
#         save_index()
#     else:
#         index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
#         with open(INDEX_DIR / "metadata.pkl", "rb") as f:
#             metadata = pickle.load(f)
#         with open(INDEX_DIR / "texts.pkl", "rb") as f:
#             texts = pickle.load(f)
    
#     print("✅ Loaded successfully!")

# def retrieve_relevant_chunks(query: str, top_k: int = 4) -> List[Dict]:
#     """Retrieve relevant document chunks"""
#     q_emb = sentence_model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     D, I = index.search(q_emb, top_k)
    
#     chunks = []
#     for score, idx in zip(D[0], I[0]):
#         if idx == -1:
#             continue
#         chunks.append({
#             "text": texts[idx],
#             "source": metadata[idx].get("source", "unknown"),
#             "chunk_id": metadata[idx].get("chunk_id", idx),
#             "score": float(score)
#         })
    
#     return chunks


# def build_simple_prompt(query: str, chunks: List[Dict], history: List[Dict]) -> str:
#     """Build a clean, simple prompt"""
    
#     # Add recent history (last 2 exchanges)
#     history_text = ""
#     if history:
#         recent = history[-4:]
#         for msg in recent:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             history_text += f"{role}: {msg['content']}\n"
    
#     # Add document context
#     context_text = ""
#     if chunks:
#         for i, chunk in enumerate(chunks, 1):
#             context_text += f"\n[Document {i}]: {chunk['text']}\n"
    
#     # Improved prompt with clear boundaries
#     prompt = f"""You are a helpful AI assistant. Answer questions clearly and naturally.

# CONVERSATION:
# {history_text if history_text else "No previous conversation."}

# DOCUMENTS:
# {context_text if context_text else "No documents provided."}

# QUESTION: {query}

# IMPORTANT RULES:
# - Write ONLY your answer in plain text
# - NO code, NO functions, NO programming syntax
# - NO markdown formatting like ``` or ***
# - Just natural conversational text
# - Keep answers concise (under 200 words)
# - If using documents, mention them naturally

# ANSWER (plain text only):
# """
    
#     return prompt


# def call_ollama(prompt: str) -> str:
#     """Call Ollama and get clean response"""
#     try:
#         result = subprocess.run(
#             ["ollama", "run", "llama3.2:3b"],
#             input=prompt,
#             text=True,
#             capture_output=True,
#             timeout=30
#         )
        
#         response = result.stdout.strip()
        
#         # Clean the response
#         response = clean_response(response)
        
#         return response
        
#     except subprocess.TimeoutExpired:
#         return "Sorry, the request timed out. Please try again."
#     except Exception as e:
#         print(f"Error calling Ollama: {e}")
#         return "Sorry, there was an error generating the response."


# def clean_response(text: str) -> str:
#     """Clean the response - simple approach"""
#     import re
    
#     # Remove lines that are clearly code (contain specific markers)
#     lines = []
#     for line in text.split('\n'):
#         line = line.strip()
#         # Skip obvious code lines
#         if any(marker in line for marker in ['function(', 'const ', '=> {', 'getElementById', '```']):
#             continue
#         if line:
#             lines.append(line)
    
#     text = '\n'.join(lines)
    
#     # Remove prompt labels if they appear at start
#     for label in ['ANSWER:', 'Answer:', 'ANSWER (plain text only):']:
#         if text.startswith(label):
#             text = text[len(label):].strip()
    
#     # Clean up spacing
#     text = re.sub(r'\n{3,}', '\n\n', text)
    
#     return text.strip()


# def extract_text_from_pdf(pdf_file) -> str:
#     """Extract text from uploaded PDF"""
#     try:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         print(f"Error extracting PDF: {e}")
#         return ""


# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#     """Split text into overlapping chunks"""
#     words = text.split()
#     chunks = []
    
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         if chunk.strip():
#             chunks.append(chunk)
    
#     return chunks


# def add_document_to_index(filename: str, text: str):
#     """Add new document to the FAISS index"""
#     global index, texts, metadata, sentence_model
    
#     # Chunk the text
#     chunks = chunk_text(text)
    
#     if not chunks:
#         return 0
    
#     # Encode chunks
#     embeddings = sentence_model.encode(chunks, convert_to_numpy=True)
#     faiss.normalize_L2(embeddings)
    
#     # Add to index
#     index.add(embeddings)
    
#     # Update texts and metadata
#     for i, chunk in enumerate(chunks):
#         texts.append(chunk)
#         metadata.append({
#             "source": filename,
#             "chunk_id": len(texts) - 1,
#             "upload_date": datetime.now().isoformat()
#         })
    
#     # Save updated index
#     save_index()
    
#     return len(chunks)


# def save_index():
#     """Save FAISS index and metadata to disk"""
#     global index, texts, metadata
    
#     try:
#         faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
#         with open(INDEX_DIR / "texts.pkl", "wb") as f:
#             pickle.dump(texts, f)
#         with open(INDEX_DIR / "metadata.pkl", "wb") as f:
#             pickle.dump(metadata, f)
#         print("✅ Index saved successfully")
#     except Exception as e:
#         print(f"❌ Error saving index: {e}")


# # Models
# class ChatRequest(BaseModel):
#     message: str
#     session_id: Optional[str] = "default"
#     top_k: Optional[int] = 4
#     use_rag: Optional[bool] = True


# class ChatResponse(BaseModel):
#     response: str
#     sources: List[Dict]
#     session_id: str
#     timestamp: str


# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     """Main chat endpoint"""
#     initialize_components()
    
#     # Get or create conversation history
#     if request.session_id not in conversations:
#         conversations[request.session_id] = []
    
#     conversation_history = conversations[request.session_id]
    
#     # Retrieve relevant chunks if RAG is enabled
#     chunks = []
#     if request.use_rag:
#         chunks = retrieve_relevant_chunks(request.message, request.top_k)
    
#     # Build prompt
#     prompt = build_simple_prompt(request.message, chunks, conversation_history)
    
#     # Generate response
#     response = call_ollama(prompt)
    
#     # Update conversation history
#     conversation_history.append({
#         "role": "user",
#         "content": request.message,
#         "timestamp": datetime.now().isoformat()
#     })
#     conversation_history.append({
#         "role": "assistant",
#         "content": response,
#         "timestamp": datetime.now().isoformat()
#     })
    
#     # Keep only last 10 messages
#     if len(conversation_history) > 10:
#         conversations[request.session_id] = conversation_history[-10:]
    
#     return ChatResponse(
#         response=response,
#         sources=chunks,
#         session_id=request.session_id,
#         timestamp=datetime.now().isoformat()
#     )


# @app.post("/clear-history")
# def clear_history(session_id: str = Form("default")):
#     """Clear conversation history"""
#     if session_id in conversations:
#         conversations[session_id] = []
#     return {"message": "History cleared", "session_id": session_id}


# @app.get("/history/{session_id}")
# def get_history(session_id: str):
#     """Get conversation history"""
#     return {
#         "session_id": session_id,
#         "history": conversations.get(session_id, [])
#     }


# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     """Upload and index a new document"""
#     initialize_components()
    
#     # Validate file type
#     if not file.filename.endswith('.pdf'):
#         return {
#             "success": False,
#             "message": "Only PDF files are supported"
#         }
    
#     try:
#         # Read PDF
#         contents = await file.read()
#         pdf_file = io.BytesIO(contents)
        
#         # Extract text
#         text = extract_text_from_pdf(pdf_file)
        
#         if not text.strip():
#             return {
#                 "success": False,
#                 "message": "Could not extract text from PDF"
#             }
        
#         # Add to index
#         num_chunks = add_document_to_index(file.filename, text)
        
#         return {
#             "success": True,
#             "message": f"Successfully indexed {file.filename}",
#             "chunks_added": num_chunks,
#             "total_chunks": len(texts)
#         }
        
#     except Exception as e:
#         print(f"Upload error: {e}")
#         return {
#             "success": False,
#             "message": f"Error processing file: {str(e)}"
#         }


# @app.get("/documents")
# def list_documents():
#     """List all indexed documents"""
#     initialize_components()
    
#     # Get unique document names
#     docs = {}
#     for meta in metadata:
#         source = meta.get("source", "unknown")
#         if source not in docs:
#             docs[source] = {
#                 "name": source,
#                 "chunks": 0,
#                 "upload_date": meta.get("upload_date", "unknown")
#             }
#         docs[source]["chunks"] += 1
    
#     return {
#         "total_documents": len(docs),
#         "total_chunks": len(texts),
#         "documents": list(docs.values())
#     }


# # Legacy endpoint for compatibility
# @app.post("/ask")
# def ask(query: str = Form(...), top_k: int = Form(4)):
#     """Legacy endpoint"""
#     request = ChatRequest(message=query, top_k=top_k)
#     result = chat(request)
#     return {
#         "query": query,
#         "results": result.sources,
#         "answer": result.response
#     }


# # Static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")


# @app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import atexit
import shutil
from pathlib import Path

from routes.chat_routes import router as chat_router
from config import INDEX_DIR

app = FastAPI(title="Simple RAG Chatbot")

# Cleanup on exit
def cleanup_on_exit():
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    print("🗑️ Index deleted")

atexit.register(cleanup_on_exit)

# Include routes
app.include_router(chat_router)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)