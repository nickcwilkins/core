"""Utility functions for QueensGuard integration."""

from urllib.parse import urlparse

import ollama
import weaviate

from homeassistant.util.ssl import get_default_context


async def create_ollama_client(
    url: str,
) -> ollama.AsyncClient:
    """Create Ollama client."""
    return ollama.AsyncClient(host=url, verify=get_default_context())


async def create_weaviate_client(
    url: str, api_key: str | None
) -> weaviate.WeaviateAsyncClient:
    """Create Weaviate client."""
    auth_credentials = weaviate.auth.Auth.api_key(api_key) if api_key else None
    parsed_url = urlparse(url)
    client: weaviate.WeaviateAsyncClient = weaviate.use_async_with_local(
        host=parsed_url.hostname,
        port=parsed_url.port or 8080,
        auth_credentials=auth_credentials,
    )
    await client.connect()
    return client
