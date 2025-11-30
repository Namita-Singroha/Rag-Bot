from fastapi import APIRouter, Form, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import io

from services import vector_service, llm_service, file_service

router = APIRouter()

conversations: Dict[str, List[Dict]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    top_k: Optional[int] = 4
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    session_id: str
    timestamp: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    vector_service.initialize()
    
    if request.session_id not in conversations:
        conversations[request.session_id] = []
    
    conversation_history = conversations[request.session_id]
    
    chunks = []
    if request.use_rag:
        chunks = vector_service.retrieve_chunks(request.message, request.top_k)
    
    prompt = llm_service.build_prompt(request.message, chunks, conversation_history)
    response = llm_service.generate_response(prompt)
    
    conversation_history.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    })
    conversation_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    if len(conversation_history) > 10:
        conversations[request.session_id] = conversation_history[-10:]
    
    return ChatResponse(
        response=response,
        sources=chunks,
        session_id=request.session_id,
        timestamp=datetime.now().isoformat()
    )

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    vector_service.initialize()
    
    if not file.filename.endswith('.pdf'):
        return {"success": False, "message": "Only PDF files are supported"}
    
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        text = file_service.extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            return {"success": False, "message": "Could not extract text from PDF"}
        
        chunks = file_service.chunk_text(text)
        num_chunks = vector_service.add_to_index(file.filename, chunks)
        
        return {
            "success": True,
            "message": f"Successfully indexed {file.filename}",
            "chunks_added": num_chunks,
            "total_chunks": len(vector_service.texts)
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

@router.post("/clear-history")
def clear_history(session_id: str = Form("default")):
    if session_id in conversations:
        conversations[session_id] = []
    return {"message": "History cleared", "session_id": session_id}

@router.get("/history/{session_id}")
def get_history(session_id: str):
    return {"session_id": session_id, "history": conversations.get(session_id, [])}

@router.get("/documents")
def list_documents():
    vector_service.initialize()
    return vector_service.get_stats()

@router.post("/ask")
def ask(query: str = Form(...), top_k: int = Form(4)):
    request = ChatRequest(message=query, top_k=top_k)
    result = chat(request)
    return {"query": query, "results": result.sources, "answer": result.response}