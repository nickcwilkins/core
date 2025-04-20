"""Config flow for Queen's Guard integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

import ollama
from ollama import ResponseError
import voluptuous as vol
import weaviate

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_EMBEDDING_MODEL,
    CONF_OLLAMA_URL,
    CONF_WEAVIATE_API_KEY,
    CONF_WEAVIATE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NAME,
    DEFAULT_TIMEOUT,
    DOMAIN,
    EMBEDDING_MODELS,
)
from .memories import MemoryManager
from .util import create_ollama_client, create_weaviate_client

_LOGGER = logging.getLogger(__name__)


@dataclass
class QueensGuardData:
    """Data for QueensGuard integration."""

    ollama: ollama.AsyncClient
    weaviate: weaviate.WeaviateClient
    memory_manager: MemoryManager


class QueensGuardConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Queen's Guard integration."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> QueensGuardOptionsFlowHandler:
        """Get the options flow for this handler."""
        return QueensGuardOptionsFlowHandler()

    def __init__(self) -> None:
        """Initialize config flow."""
        self.weaviate_url: str | None = None
        self.weaviate_api_key: str | None = None
        self.ollama_url: str | None = None
        self.embedding_model: str | None = None
        self.ollama_client: ollama.AsyncClient | None = None
        self.download_task: asyncio.Task | None = None
        self.weaviate_client: weaviate.WeaviateClient | None = None
        self.downloaded_models: set[str] = set()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.weaviate_url = user_input.get(CONF_WEAVIATE_URL, self.weaviate_url)
            self.weaviate_api_key = user_input.get(CONF_WEAVIATE_API_KEY)
            self.ollama_url = user_input.get(CONF_OLLAMA_URL, self.ollama_url)
            self.embedding_model = user_input.get(
                CONF_EMBEDDING_MODEL, self.embedding_model
            )

            # If we have URLs but no model yet, proceed to connection testing
            if self.weaviate_url and self.ollama_url and not self.embedding_model:
                _LOGGER.debug("Testing connection to Weaviate at %s", self.weaviate_url)
                # Test connections to both services
                weaviate_error = await self._test_weaviate_connection(
                    self.weaviate_url, self.weaviate_api_key
                )
                if weaviate_error:
                    errors["base"] = weaviate_error  # Use 'base' for general errors

                _LOGGER.debug("Testing connection to Ollama at %s", self.ollama_url)
                ollama_error, models = await self._test_ollama_connection(
                    self.ollama_url
                )
                if ollama_error:
                    errors[CONF_OLLAMA_URL] = ollama_error
                else:
                    self.downloaded_models = models

                if not errors:
                    _LOGGER.debug("Successfully connected to Weaviate and Ollama")
                    # Proceed to model selection
                    return await self.async_step_select_model()

            # If we have all info, create the entry
            elif self.weaviate_url and self.ollama_url and self.embedding_model:
                return self.async_create_entry(
                    title=DEFAULT_NAME,
                    data={
                        CONF_WEAVIATE_URL: self.weaviate_url,
                        CONF_WEAVIATE_API_KEY: self.weaviate_api_key,
                        CONF_OLLAMA_URL: self.ollama_url,
                        CONF_EMBEDDING_MODEL: self.embedding_model,
                    },
                )

        # If there is no user input or there were errors, show the form again
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_WEAVIATE_URL,
                    default=self.weaviate_url if self.weaviate_url else None,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
                # vol.Optional(CONF_WEAVIATE_API_KEY): TextSelector(
                #     TextSelectorConfig(type=TextSelectorType.PASSWORD)
                # ),
                vol.Required(
                    CONF_OLLAMA_URL,
                    default=self.ollama_url if self.ollama_url else None,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_select_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the embedding model to use."""
        if user_input is not None:
            self.embedding_model = user_input[CONF_EMBEDDING_MODEL]

            if self.embedding_model not in self.downloaded_models:
                # Ollama server needs to download model first
                return await self.async_step_download()

            # Model is already downloaded, create entry
            return self.async_create_entry(
                title=DEFAULT_NAME,
                data={
                    CONF_WEAVIATE_URL: self.weaviate_url,
                    CONF_WEAVIATE_API_KEY: self.weaviate_api_key,
                    CONF_OLLAMA_URL: self.ollama_url,
                    CONF_EMBEDDING_MODEL: self.embedding_model,
                },
            )

        # Show models that have been downloaded first, followed by all known models
        models_to_list = [
            SelectOptionDict(label=f"{m} (downloaded)", value=m)
            for m in sorted(self.downloaded_models)
            if any(m.startswith(known_model) for known_model in EMBEDDING_MODELS)
        ] + [
            SelectOptionDict(label=m, value=f"{m}:latest")
            for m in sorted(EMBEDDING_MODELS)
            if m not in self.downloaded_models
        ]

        model_step_schema = vol.Schema(
            {
                vol.Required(
                    CONF_EMBEDDING_MODEL,
                    default=DEFAULT_EMBEDDING_MODEL
                    if DEFAULT_EMBEDDING_MODEL in self.downloaded_models
                    else f"{DEFAULT_EMBEDDING_MODEL}:latest",
                ): SelectSelector(
                    SelectSelectorConfig(options=models_to_list, custom_value=True)
                ),
            }
        )

        return self.async_show_form(
            step_id="select_model",
            data_schema=model_step_schema,
        )

    async def async_step_download(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step to wait for Ollama server to download a model."""
        assert self.embedding_model is not None
        assert self.ollama_client is not None

        if self.download_task is None:
            # Tell Ollama server to pull the model.
            # The task will block until the model and metadata are fully downloaded.
            self.download_task = self.hass.async_create_background_task(
                self.ollama_client.pull(self.embedding_model),
                f"Downloading {self.embedding_model}",
            )

        try:
            await self.download_task
        except Exception:
            _LOGGER.exception("Unexpected error while downloading model")
            return self.async_show_progress_done(next_step_id="failed")
        finally:
            self.download_task = None

        return self.async_show_progress_done(next_step_id="finish")

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step after model downloading has succeeded."""
        assert self.weaviate_url is not None
        assert self.ollama_url is not None
        assert self.embedding_model is not None

        return self.async_create_entry(
            title=DEFAULT_NAME,
            data={
                CONF_WEAVIATE_URL: self.weaviate_url,
                CONF_WEAVIATE_API_KEY: self.weaviate_api_key,
                CONF_OLLAMA_URL: self.ollama_url,
                CONF_EMBEDDING_MODEL: self.embedding_model,
            },
        )

    async def async_step_failed(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step after model downloading has failed."""
        return self.async_abort(reason="download_failed")

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure connection settings."""
        errors: dict[str, str] = {}
        entry = self.current_entry
        assert entry  # Should exist for reconfigure

        if user_input is not None:
            weaviate_url = user_input[CONF_WEAVIATE_URL]
            weaviate_api_key = user_input.get(CONF_WEAVIATE_API_KEY)
            ollama_url = user_input[CONF_OLLAMA_URL]
            embedding_model = user_input.get(
                CONF_EMBEDDING_MODEL,
                entry.data.get(CONF_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL),
            )

            # Test connections to both services
            weaviate_error = await self._test_weaviate_connection(
                weaviate_url, weaviate_api_key
            )
            if weaviate_error:
                errors["base"] = weaviate_error

            ollama_error, models = await self._test_ollama_connection(ollama_url)
            if ollama_error:
                errors[CONF_OLLAMA_URL] = ollama_error

            # Ensure the model is available or can be downloaded
            if not ollama_error and embedding_model not in models:
                # We'll need to download - but that happens in a separate step
                self.weaviate_url = weaviate_url
                self.weaviate_api_key = weaviate_api_key
                self.ollama_url = ollama_url
                self.embedding_model = embedding_model
                return await self.async_step_download()

            if not errors:
                _LOGGER.debug("Successfully connected to Weaviate and Ollama")
                data = {
                    CONF_WEAVIATE_URL: weaviate_url,
                    CONF_WEAVIATE_API_KEY: weaviate_api_key,
                    CONF_OLLAMA_URL: ollama_url,
                    CONF_EMBEDDING_MODEL: embedding_model,
                }
                return self.async_update_reload_and_abort(entry, data=data)

        # Get current values from entry
        weaviate_url = entry.data.get(CONF_WEAVIATE_URL, "")
        weaviate_api_key = entry.data.get(CONF_WEAVIATE_API_KEY)
        ollama_url = entry.data.get(CONF_OLLAMA_URL, "")
        embedding_model = entry.data.get(CONF_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)

        schema = vol.Schema(
            {
                vol.Required(CONF_WEAVIATE_URL, default=weaviate_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Optional(
                    CONF_WEAVIATE_API_KEY, default=weaviate_api_key
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                vol.Required(CONF_OLLAMA_URL, default=ollama_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Required(
                    CONF_EMBEDDING_MODEL, default=embedding_model
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
            }
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )

    async def _test_weaviate_connection(
        self, url: str, api_key: str | None
    ) -> str | None:
        """Test connection to Weaviate.

        Returns None if connection successful, error code string otherwise.
        """

        try:
            client = await create_weaviate_client(url, api_key)
            # Test connection by checking readiness
            if not await client.is_ready():
                _LOGGER.error("Weaviate instance at %s is not ready", url)
                await client.close()
                return "cannot_connect"
            await client.close()

        except ValueError:
            _LOGGER.exception("Invalid URL format for Weaviate at %s", url)
            return "invalid_url"
        except Exception:
            _LOGGER.exception("Unexpected exception connecting to Weaviate at %s", url)
            return "unknown"
        else:
            return None

    async def _test_ollama_connection(self, url: str) -> tuple[str | None, set[str]]:
        """Test connection to Ollama.

        Returns a tuple of (error_code, downloaded_models).
        Error is None if connection successful, error code string otherwise.
        """
        downloaded_models: set[str] = set()

        try:
            # Create ollama client with the provided URL
            self.ollama_client = await create_ollama_client(url)

            # Test the connection by listing available models
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                response = await self.ollama_client.list()

            # Verify the response contains models data
            if "models" not in response:
                _LOGGER.error("Ollama API returned unexpected data: %s", response)
                return "invalid_response", downloaded_models

            # Get list of downloaded models
            downloaded_models = {
                model_info["model"] for model_info in response.get("models", [])
            }

        except TimeoutError:
            _LOGGER.error("Timeout connecting to Ollama at %s", url)
            return "timeout", downloaded_models
        except ResponseError as err:
            _LOGGER.error("Ollama API error: %s", err)
            return "cannot_connect", downloaded_models
        except ConnectionError as err:
            _LOGGER.error("Ollama connection error: %s", err)
            return "connection_error", downloaded_models
        else:
            return None, downloaded_models


class QueensGuardOptionsFlowHandler(OptionsFlow):
    """Handle Queen's Guard options."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        # Options flow should redirect to reconfigure to allow changing URLs/API key
        return await self.async_step_reconfigure()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure the integration."""
        # Start the reconfigure flow by transferring to the ConfigFlow reconfigure step
        return await self.hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": SOURCE_RECONFIGURE,
                "entry_id": self.config_entry.entry_id,
            },
            data=self.config_entry.data,
        )
