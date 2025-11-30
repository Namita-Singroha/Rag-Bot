import PyPDF2
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks