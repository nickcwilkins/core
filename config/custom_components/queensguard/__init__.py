"""Queen's Guard integration for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import async_register_api
from homeassistant.helpers.typing import ConfigType

from .api import QueensGuardAPI
from .const import CONF_CHROMA_URL, CONF_OLLAMA_URL, DOMAIN
from .embeddings import EmbeddingManager

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Queen's Guard integration."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Queen's Guard from a config entry."""
    chroma_url = entry.data.get(CONF_CHROMA_URL)
    ollama_url = entry.data.get(CONF_OLLAMA_URL)

    # Create the embedding manager
    embedding_manager = EmbeddingManager(hass, chroma_url, ollama_url)

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
