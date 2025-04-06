"""Queen's Guard integration for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .api import QueensGuardAPI
from .const import CONF_CHROMA_URL, CONF_OLLAMA_URL, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Queen's Guard integration."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Queen's Guard from a config entry."""
    chroma_url = entry.data.get(CONF_CHROMA_URL)
    ollama_url = entry.data.get(CONF_OLLAMA_URL)

    # Create the API instance
    api = QueensGuardAPI(hass, chroma_url, ollama_url)

    # Set up the API's embedding listeners
    await api.async_setup_embedding_listeners()

    # Store the API instance in the hass data registry
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = api

    # Register the API with Home Assistant's LLM system
    # In Home Assistant 2025+, the LLM API is registered via a direct call
    service_name = "register_api"
    if hass.services.has_service("llm", service_name):
        await hass.services.async_call(
            domain="llm",
            service=service_name,
            service_data={"api": api},
            blocking=True,
        )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    api = hass.data[DOMAIN][entry.entry_id]
    await api.async_unload()

    hass.data[DOMAIN].pop(entry.entry_id)
    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)

    # Unregister the API with Home Assistant's LLM system
    service_name = "unregister_api"
    if hass.services.has_service("llm", service_name):
        await hass.services.async_call(
            domain="llm",
            service=service_name,
            service_data={"api_id": api.id},
            blocking=True,
        )

    return True
