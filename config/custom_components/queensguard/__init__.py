"""Queen's Guard integration for Home Assistant."""

from __future__ import annotations

import logging
from typing import cast

import weaviate

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.llm import async_register_api
from homeassistant.helpers.typing import ConfigType

from .api import QueensGuardAPI
from .const import (
    CONF_EMBEDDING_MODEL,
    CONF_OLLAMA_URL,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_URL,
    DOMAIN,
)
from .memories import MemoryManager
from .util import create_ollama_client, create_weaviate_client

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = []  # No platforms needed for now


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Queen's Guard integration."""
    # Ensure domain is setup
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Queen's Guard from a config entry."""
    weaviate_url = cast(str, entry.data[CONF_WEAVIATE_URL])
    weaviate_api_key = entry.data.get(CONF_WEAVIATE_API_KEY)
    ollama_url = cast(str, entry.data[CONF_OLLAMA_URL])
    embedding_model = cast(str, entry.data[CONF_EMBEDDING_MODEL])

    _LOGGER.info(
        "Setting up Queen's Guard with Weaviate: %s, Ollama: %s, Embedding Model: %s",
        weaviate_url,
        ollama_url,
        embedding_model,
    )

    # Initialize Weaviate client
    weaviate_client = await create_weaviate_client(weaviate_url, weaviate_api_key)
    if not await weaviate_client.is_ready():
        _LOGGER.error("Weaviate instance at %s is not ready", weaviate_url)
        await weaviate_client.close()
        raise ConfigEntryNotReady(f"Weaviate instance at {weaviate_url} is not ready")

    # Initialize Ollama client
    ollama_client = await create_ollama_client(ollama_url)

    # Create the memory manager
    memory_manager = MemoryManager(
        hass=hass,
        weaviate_client=weaviate_client,
        ollama_client=ollama_client,
        embedding_model=embedding_model,
    )

    try:
        # Set up memory manager (e.g., create collection if needed)
        await memory_manager.async_setup()
    except Exception as err:
        _LOGGER.error("Failed to setup Memory Manager: %s", err)
        # Ensure client is closed on setup failure
        await weaviate_client.close()
        raise ConfigEntryNotReady(f"Failed to setup Memory Manager: {err}") from err

    # Create the API instance with a reference to the memory manager
    api = QueensGuardAPI(hass, memory_manager)

    # Register the API with Home Assistant's LLM system
    unregister_api = async_register_api(hass, api)

    # Store components in the hass data registry
    hass.data[DOMAIN][entry.entry_id] = {
        "api": api,
        "memory_manager": memory_manager,
        "weaviate_client": weaviate_client,
        "ollama_client": ollama_client,
        "unregister_api": unregister_api,
    }

    # Set up options listener
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if entry.entry_id not in hass.data[DOMAIN]:
        return True  # Already unloaded

    data = hass.data[DOMAIN].pop(entry.entry_id)

    # Unregister API from LLM system
    if "unregister_api" in data:
        data["unregister_api"]()

    # Unload memory manager (add async_unload if needed)
    # if "memory_manager" in data:
    #     await data["memory_manager"].async_unload()

    # Close Weaviate client
    if "weaviate_client" in data:
        weaviate_client: weaviate.WeaviateClient = data["weaviate_client"]
        await hass.async_add_executor_job(weaviate_client.close)
        _LOGGER.debug("Closed Weaviate client")

    # Clean up domain data if empty
    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)

    _LOGGER.info("Successfully unloaded Queen's Guard integration")
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
