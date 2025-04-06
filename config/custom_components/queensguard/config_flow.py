"""Config flow for Queen's Guard integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp
import chromadb
import ollama
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.util.ssl import get_default_context

from .const import CONF_CHROMA_URL, CONF_OLLAMA_URL, DOMAIN

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds for API connection tests


async def _test_chroma_connection(url: str) -> str | None:
    """Test connection to ChromaDB.

    Returns None if connection successful, error code string otherwise.
    """

    try:
        # parse the url
        parsed_url = urlparse(url)
        chromadb.HttpClient(host=parsed_url.hostname, port=parsed_url.port)

    except TimeoutError:
        _LOGGER.exception("Timeout connecting to ChromaDB at %s", url)
        return "timeout"
    except aiohttp.ClientError as err:
        _LOGGER.exception("Error connecting to ChromaDB at %s", url)
        return str(err)
    except ValueError as err:
        _LOGGER.exception("Error connecting to ChromaDB at %s", url)
        return str(err)
    except Exception as err:
        _LOGGER.exception("Unexpected exception connecting to ChromaDB at %s", url)
        return str(err)
    else:
        return None


async def _test_ollama_connection(url: str) -> str | None:
    """Test connection to Ollama.

    Returns None if connection successful, error code string otherwise.
    """
    try:
        # Create ollama client with the provided URL
        client = ollama.AsyncClient(host=url, verify=get_default_context())

        # Test the connection by listing available models
        async with asyncio.timeout(DEFAULT_TIMEOUT):
            response = await client.list()

        # Verify the response contains models data
        if "models" not in response:
            _LOGGER.error("Ollama API returned unexpected data: %s", response)
            return "invalid_response"

    except TimeoutError:
        _LOGGER.error("Timeout connecting to Ollama at %s", url)
        return "timeout"
    except ollama.ResponseError as err:
        _LOGGER.error("Ollama API error: %s", err)
        return "cannot_connect"
    except Exception:  # pylint: disable=broad-except
        _LOGGER.exception("Unexpected exception connecting to Ollama at %s", url)
        return "unknown"
    else:
        return None


class QueensGuardConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Queen's Guard integration."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> QueensGuardOptionsFlowHandler:
        """Get the options flow for this handler."""
        return QueensGuardOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            chroma_url = user_input[CONF_CHROMA_URL]
            ollama_url = user_input[CONF_OLLAMA_URL]

            # Check if this combination of URLs is already configured
            await self.async_set_unique_id(f"{chroma_url}_{ollama_url}")
            self._abort_if_unique_id_configured()

            _LOGGER.debug("Testing connection to ChromaDB at %s", chroma_url)
            # Test connections to both services
            chroma_error = await self._test_chroma_connection(chroma_url)
            if chroma_error:
                errors[CONF_CHROMA_URL] = chroma_error

            _LOGGER.debug("Testing connection to Ollama at %s", ollama_url)
            ollama_error = await self._test_ollama_connection(ollama_url)
            if ollama_error:
                errors[CONF_OLLAMA_URL] = ollama_error

            if not errors:
                _LOGGER.info("Successfully connected to ChromaDB and Ollama")
                return self.async_create_entry(
                    data={
                        CONF_CHROMA_URL: chroma_url,
                        CONF_OLLAMA_URL: ollama_url,
                    },
                )

        # If there is no user input or there were errors, show the form again
        schema = vol.Schema(
            {
                vol.Required(CONF_CHROMA_URL): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Required(CONF_OLLAMA_URL): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )


class QueensGuardOptionsFlowHandler(OptionsFlow):
    """Handle Queen's Guard options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.config_flow = QueensGuardConfigFlow()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            chroma_url = user_input[CONF_CHROMA_URL]
            ollama_url = user_input[CONF_OLLAMA_URL]

            # Test connections to both services if URLs changed
            if chroma_url != self.config_entry.data.get(CONF_CHROMA_URL):
                chroma_error = await _test_chroma_connection(chroma_url)
                if chroma_error:
                    errors[CONF_CHROMA_URL] = chroma_error

            if ollama_url != self.config_entry.data.get(CONF_OLLAMA_URL):
                ollama_error = await _test_ollama_connection(ollama_url)
                if ollama_error:
                    errors[CONF_OLLAMA_URL] = ollama_error

            if not errors:
                # Create a new data dict with updated values
                data = {**self.config_entry.data, **user_input}
                return self.async_create_entry(title="", data=data)

        # Get current values from entry
        chroma_url = self.config_entry.data.get(CONF_CHROMA_URL, "")
        ollama_url = self.config_entry.data.get(CONF_OLLAMA_URL, "")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CHROMA_URL, default=chroma_url): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.URL)
                    ),
                    vol.Required(CONF_OLLAMA_URL, default=ollama_url): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.URL)
                    ),
                }
            ),
            errors=errors,
        )
