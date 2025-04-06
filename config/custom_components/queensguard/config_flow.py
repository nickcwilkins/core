"""Config flow for Queen's Guard integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import chromadb
import ollama
import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
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

from .const import CONF_CHROMA_URL, CONF_OLLAMA_URL, DEFAULT_NAME, DOMAIN

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds for API connection tests


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

    def __init__(self) -> None:
        """Initialize config flow."""
        self.url: str | None = None
        self.model: str | None = None
        self.ollama_client: ollama.AsyncClient | None = None
        self.download_task: asyncio.Task | None = None
        self.chroma_client: chromadb.Client | None = None

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
                _LOGGER.debug("Successfully connected to ChromaDB and Ollama")
                return self.async_create_entry(
                    title=DEFAULT_NAME,
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
            data_schema=schema,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure connection settings."""
        errors: dict[str, str] = {}
        entry = self._get_reconfigure_entry()

        if user_input is not None:
            chroma_url = user_input[CONF_CHROMA_URL]
            ollama_url = user_input[CONF_OLLAMA_URL]

            # Check if this combination of URLs is already configured in another entry
            new_unique_id = f"{chroma_url}_{ollama_url}"
            await self.async_set_unique_id(new_unique_id)
            self._abort_if_unique_id_mismatch()

            # Test connections to both services
            chroma_error = await self._test_chroma_connection(chroma_url)
            if chroma_error:
                errors[CONF_CHROMA_URL] = chroma_error

            ollama_error = await self._test_ollama_connection(ollama_url)
            if ollama_error:
                errors[CONF_OLLAMA_URL] = ollama_error

            if not errors:
                _LOGGER.debug("Successfully connected to ChromaDB and Ollama")
                data = {
                    CONF_CHROMA_URL: chroma_url,
                    CONF_OLLAMA_URL: ollama_url,
                }
                return self.async_update_reload_and_abort(
                    entry, data=data, unique_id=new_unique_id
                )

        # Get current values from entry
        chroma_url = entry.data.get(CONF_CHROMA_URL, "")
        ollama_url = entry.data.get(CONF_OLLAMA_URL, "")

        schema = vol.Schema(
            {
                vol.Required(CONF_CHROMA_URL, default=chroma_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Required(CONF_OLLAMA_URL, default=ollama_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
            }
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    async def _test_chroma_connection(self, url: str) -> str | None:
        """Test connection to ChromaDB.

        Returns None if connection successful, error code string otherwise.
        """
        try:
            # parse the url
            parsed_url = urlparse(url)
            self.chroma_client = chromadb.HttpClient(
                host=parsed_url.hostname, port=parsed_url.port
            )
        except TimeoutError:
            _LOGGER.exception("Timeout connecting to ChromaDB at %s", url)
            return "timeout"
        except ValueError:
            _LOGGER.exception("Invalid URL format for ChromaDB at %s", url)
            return "invalid_url"
        except Exception:
            _LOGGER.exception("Unexpected exception connecting to ChromaDB at %s", url)
            return "unknown"
        else:
            return None

    async def _test_ollama_connection(self, url: str) -> str | None:
        """Test connection to Ollama.

        Returns None if connection successful, error code string otherwise.
        """
        try:
            # Create ollama client with the provided URL
            self.ollama_client = ollama.AsyncClient(
                host=url, verify=get_default_context()
            )

            # Test the connection by listing available models
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                response = await self.ollama_client.list()

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
        except ConnectionError as err:
            _LOGGER.error("Ollama connection error: %s", err)
            return "connection_error"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception connecting to Ollama at %s", url)
            return "unknown"
        else:
            return None


class QueensGuardOptionsFlowHandler(OptionsFlow):
    """Handle Queen's Guard options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        return await self.async_step_reconfigure()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure the integration."""
        # Start the reconfigure flow by transferring to the ConfigFlow reconfigure step
        return self.hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": SOURCE_RECONFIGURE,
                "entry_id": self.config_entry.entry_id,
            },
            data=self.config_entry.data,
        )
