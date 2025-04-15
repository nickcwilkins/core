"""Queen's Guard integration for Home Assistant."""

from __future__ import annotations

import logging
from typing import cast
from urllib.parse import urlparse

import chromadb
from chromadb.api.async_api import AsyncClientAPI
from ollama import AsyncClient

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import async_register_api
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.ssl import get_default_context

from .api import QueensGuardAPI
from .const import CONF_CHROMA_URL, CONF_MODEL, CONF_OLLAMA_URL, DOMAIN
from .embeddings import EmbeddingManager

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Queen's Guard integration."""
    return True


async def _create_chroma_client(url: str) -> AsyncClientAPI:
    """Create ChromaDB client in an executor to avoid blocking I/O in the event loop."""
    parsed_url = urlparse(url)
    return await chromadb.AsyncHttpClient(
        host=parsed_url.hostname or "localhost",
        port=parsed_url.port or 8000,
        ssl=parsed_url.scheme == "https",
    )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Queen's Guard from a config entry."""
    chroma_url = cast(str, entry.data.get(CONF_CHROMA_URL))
    ollama_url = cast(str, entry.data.get(CONF_OLLAMA_URL))
    model = cast(str, entry.data.get(CONF_MODEL))

    _LOGGER.info("Setting up Queen's Guard with model: %s", model)

    # Initialize ChromaDB client in the executor to avoid blocking I/O
    chroma_client = await _create_chroma_client(chroma_url)

    # Initialize Ollama client
    ollama_client = AsyncClient(host=ollama_url, verify=get_default_context())

    # Create the embedding manager with the client instances and model
    embedding_manager = EmbeddingManager(
        hass=hass,
        chroma_client=chroma_client,
        ollama_client=ollama_client,
        model=model,
    )

    # Set up embedding listeners
    await embedding_manager.async_setup()

    # Create the API instance with a reference to the embedding manager
    api = QueensGuardAPI(hass, chroma_url, ollama_url, embedding_manager)

    # Register the API with Home Assistant's LLM system
    unregister_api = async_register_api(hass, api)

    # Store both components in the hass data registry
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "api": api,
        "embedding_manager": embedding_manager,
        "chroma_client": chroma_client,
        "ollama_client": ollama_client,
        "unregister_api": unregister_api,
    }

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    components = hass.data[DOMAIN][entry.entry_id]

    # Unload embedding manager
    await components["embedding_manager"].async_unload()

    # Unload API
    await components["api"].async_unload()

    # Unregister API from LLM system
    components["unregister_api"]()

    # Clean up data
    hass.data[DOMAIN].pop(entry.entry_id)
    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)

    return True
