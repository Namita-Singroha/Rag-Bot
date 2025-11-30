import subprocess
import re
from typing import List, Dict
from config import LLM_MODEL

def build_prompt(query: str, chunks: List[Dict], history: List[Dict]) -> str:
    history_text = ""
    if history:
        recent = history[-4:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    
    context_text = ""
    if chunks:
        for i, chunk in enumerate(chunks, 1):
            context_text += f"\n[Document {i}]: {chunk['text']}\n"
    
    prompt = f"""You are a helpful AI assistant. Answer questions clearly and naturally.

CONVERSATION:
{history_text if history_text else "No previous conversation."}

DOCUMENTS:
{context_text if context_text else "No documents provided."}

QUESTION: {query}

IMPORTANT RULES:
- Write ONLY your answer in plain text
- NO code, NO functions, NO programming syntax
- NO markdown formatting like ``` or ***
- Just natural conversational text
- Keep answers concise (under 200 words)
- If using documents, mention them naturally

ANSWER (plain text only):
"""
    return prompt

def generate_response(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL, "--verbose"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=30
        )
        response = result.stdout.strip()
        return clean_response(response)
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "Sorry, there was an error generating the response."

def clean_response(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if any(marker in line for marker in ['function(', 'const ', '=> {', 'getElementById', '```']):
            continue
        if line:
            lines.append(line)
    
    text = '\n'.join(lines)
    
    for label in ['ANSWER:', 'Answer:', 'ANSWER (plain text only):']:
        if text.startswith(label):
            text = text[len(label):].strip()
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()