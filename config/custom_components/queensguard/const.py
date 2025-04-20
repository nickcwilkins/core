"""Constants for the Queen's Guard integration."""

from __future__ import annotations

DOMAIN = "queensguard"

# Configuration constants
CONF_WEAVIATE_URL = "weaviate_url"
CONF_WEAVIATE_API_KEY = "weaviate_api_key"  # Optional API key
CONF_OLLAMA_URL = "ollama_url"
CONF_EMBEDDING_MODEL = "embedding_model"  # Renamed from CONF_MODEL for clarity
CONF_LLM_MODEL = "llm_model"  # Added for the main LLM model if needed, or assume one is selected elsewhere? Prompt implies only embedding model selection in config flow. Let's stick to embedding model for now.

DEFAULT_NAME = "Queen's Guard"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 10  # seconds for API connection tests

# Weaviate constants
WEAVIATE_MEMORY_COLLECTION = "QueenGuardMemories"

# Embedding models suitable for text embedding (used for Weaviate RAG)
EMBEDDING_MODELS = [
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
    "mxbai-embed-small",
]
