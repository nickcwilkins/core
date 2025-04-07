"""Constants for the Queen's Guard integration."""

from __future__ import annotations

DOMAIN = "queensguard"

# Configuration constants
CONF_CHROMA_URL = "chroma_url"
CONF_OLLAMA_URL = "ollama_url"
CONF_MODEL = "model"
DEFAULT_NAME = "Queen's Guard"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 10  # seconds for API connection tests

# Embedding models suitable for text embedding
EMBEDDING_MODELS = [
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
    "mxbai-embed-small",
]
