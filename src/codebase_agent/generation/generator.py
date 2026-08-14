"""LLM Generator component connecting to local Ollama server (FR-5.4, FR-8.1, NFR-6)."""

import json
from typing import List, Optional
import requests

from codebase_agent.retrieval.models import RetrievedChunk


SYSTEM_PROMPT = """You are a senior software engineering assistant answering questions about a software repository.
Use ONLY the provided code snippets below as evidence to answer the user's question.
Do NOT fabricate file paths, function signatures, or line numbers not present in the context.
If the context does not contain enough information to answer the question accurately, explicitly state:
"I do not have sufficient context in the provided codebase to answer this question."
"""


class LLMGenerator:
    """Invokes local Ollama LLM to synthesize answers grounded in retrieved code context."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        ollama_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

    def format_context_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Formats code chunks and line metadata into prompt context block."""
        context_parts: List[str] = [f"User Question: {query}\n", "Retrieved Code Context:"]

        for idx, chunk in enumerate(chunks, 1):
            header = f"\n--- Snippet [{idx}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (Symbol: {chunk.symbol_name}) ---"
            context_parts.append(header)
            context_parts.append(chunk.document)

        return "\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> str:
        """
        Calls local Ollama API (/api/chat) with grounded system prompt (FR-5.4).
        Catches connection errors if Ollama is not running (NFR-6).
        """
        context_text = self.format_context_prompt(query, chunks)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context_text}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        chat_url = f"{self.ollama_url}/api/chat"

        try:
            response = requests.post(chat_url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                msg = data.get("message", {}).get("content", "")
                return msg.strip()
            elif response.status_code == 404:
                return (
                    f"Error: Ollama model '{self.model_name}' was not found. "
                    f"Please pull it first by running: 'ollama pull {self.model_name}'"
                )
            else:
                return f"Ollama API returned HTTP status {response.status_code}: {response.text}"

        except requests.exceptions.ConnectionError:
            return (
                f"Ollama server is not running locally at {self.ollama_url}.\n"
                f"Please start Ollama service and ensure the model is pulled:\n"
                f"  1. Start Ollama: 'ollama serve'\n"
                f"  2. Pull model:  'ollama pull {self.model_name}'"
            )
        except Exception as ex:
            return f"Error communicating with local LLM generator: {str(ex)}"
