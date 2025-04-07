"""Config flow for Queen's Guard integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import chromadb
import ollama
from ollama import ResponseError
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
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.util.ssl import get_default_context

from .const import (
    CONF_CHROMA_URL,
    CONF_MODEL,
    CONF_OLLAMA_URL,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_TIMEOUT,
    DOMAIN,
    EMBEDDING_MODELS,
)

_LOGGER = logging.getLogger(__name__)


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
        self.chroma_url: str | None = None
        self.ollama_url: str | None = None
        self.model: str | None = None
        self.ollama_client: ollama.AsyncClient | None = None
        self.download_task: asyncio.Task | None = None
        self.chroma_client: chromadb.Client | None = None
        self.downloaded_models: set[str] = set()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.chroma_url = user_input.get(CONF_CHROMA_URL, self.chroma_url)
            self.ollama_url = user_input.get(CONF_OLLAMA_URL, self.ollama_url)
            self.model = user_input.get(CONF_MODEL, self.model)

            # If we have URLs but no model yet, proceed to connection testing
            if self.chroma_url and self.ollama_url and not self.model:
                # Check if this combination of URLs is already configured
                await self.async_set_unique_id(f"{self.chroma_url}_{self.ollama_url}")
                self._abort_if_unique_id_configured()

                _LOGGER.debug("Testing connection to ChromaDB at %s", self.chroma_url)
                # Test connections to both services
                chroma_error = await self._test_chroma_connection(self.chroma_url)
                if chroma_error:
                    errors[CONF_CHROMA_URL] = chroma_error

                _LOGGER.debug("Testing connection to Ollama at %s", self.ollama_url)
                ollama_error, models = await self._test_ollama_connection(
                    self.ollama_url
                )
                if ollama_error:
                    errors[CONF_OLLAMA_URL] = ollama_error
                else:
                    self.downloaded_models = models

                if not errors:
                    _LOGGER.debug("Successfully connected to ChromaDB and Ollama")
                    # Proceed to model selection
                    return await self.async_step_select_model()

            # If we have all info, create the entry
            elif self.chroma_url and self.ollama_url and self.model:
                return self.async_create_entry(
                    title=DEFAULT_NAME,
                    data={
                        CONF_CHROMA_URL: self.chroma_url,
                        CONF_OLLAMA_URL: self.ollama_url,
                        CONF_MODEL: self.model,
                    },
                )

        # If there is no user input or there were errors, show the form again
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_CHROMA_URL,
                    default=self.chroma_url if self.chroma_url else None,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
                vol.Required(
                    CONF_OLLAMA_URL,
                    default=self.ollama_url if self.ollama_url else None,
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_select_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the embedding model to use."""
        if user_input is not None:
            self.model = user_input[CONF_MODEL]

            if self.model not in self.downloaded_models:
                # Ollama server needs to download model first
                return await self.async_step_download()

            # Model is already downloaded, create entry
            return self.async_create_entry(
                title=DEFAULT_NAME,
                data={
                    CONF_CHROMA_URL: self.chroma_url,
                    CONF_OLLAMA_URL: self.ollama_url,
                    CONF_MODEL: self.model,
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
                    CONF_MODEL,
                    default=DEFAULT_MODEL
                    if DEFAULT_MODEL in self.downloaded_models
                    else f"{DEFAULT_MODEL}:latest",
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
        assert self.model is not None
        assert self.ollama_client is not None

        if self.download_task is None:
            # Tell Ollama server to pull the model.
            # The task will block until the model and metadata are fully downloaded.
            self.download_task = self.hass.async_create_background_task(
                self.ollama_client.pull(self.model),
                f"Downloading {self.model}",
            )

        if self.download_task.done():
            if err := self.download_task.exception():
                _LOGGER.exception("Unexpected error while downloading model: %s", err)
                return self.async_show_progress_done(next_step_id="failed")

            return self.async_show_progress_done(next_step_id="finish")

        return self.async_show_progress(
            step_id="download",
            progress_action="download",
            progress_task=self.download_task,
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step after model downloading has succeeded."""
        assert self.chroma_url is not None
        assert self.ollama_url is not None
        assert self.model is not None

        return self.async_create_entry(
            title=DEFAULT_NAME,
            data={
                CONF_CHROMA_URL: self.chroma_url,
                CONF_OLLAMA_URL: self.ollama_url,
                CONF_MODEL: self.model,
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
        entry = self._get_reconfigure_entry()

        if user_input is not None:
            chroma_url = user_input[CONF_CHROMA_URL]
            ollama_url = user_input[CONF_OLLAMA_URL]
            model = user_input.get(
                CONF_MODEL, entry.data.get(CONF_MODEL, DEFAULT_MODEL)
            )

            # Check if this combination of URLs is already configured in another entry
            new_unique_id = f"{chroma_url}_{ollama_url}"
            await self.async_set_unique_id(new_unique_id)
            self._abort_if_unique_id_mismatch()

            # Test connections to both services
            chroma_error = await self._test_chroma_connection(chroma_url)
            if chroma_error:
                errors[CONF_CHROMA_URL] = chroma_error

            ollama_error, models = await self._test_ollama_connection(ollama_url)
            if ollama_error:
                errors[CONF_OLLAMA_URL] = ollama_error

            # Ensure the model is available or can be downloaded
            if not ollama_error and model not in models:
                # We'll need to download - but that happens in a separate step
                self.chroma_url = chroma_url
                self.ollama_url = ollama_url
                self.model = model
                self.ollama_client = ollama.AsyncClient(
                    host=ollama_url, verify=get_default_context()
                )
                return await self.async_step_download()

            if not errors:
                _LOGGER.debug("Successfully connected to ChromaDB and Ollama")
                data = {
                    CONF_CHROMA_URL: chroma_url,
                    CONF_OLLAMA_URL: ollama_url,
                    CONF_MODEL: model,
                }
                return self.async_update_reload_and_abort(
                    entry, data=data, unique_id=new_unique_id
                )

        # Get current values from entry
        chroma_url = entry.data.get(CONF_CHROMA_URL, "")
        ollama_url = entry.data.get(CONF_OLLAMA_URL, "")
        model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)

        schema = vol.Schema(
            {
                vol.Required(CONF_CHROMA_URL, default=chroma_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Required(CONF_OLLAMA_URL, default=ollama_url): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
                vol.Required(CONF_MODEL, default=model): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT)
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
            # Parse the url
            parsed_url = urlparse(url)
            self.chroma_client = chromadb.HttpClient(
                host=parsed_url.hostname or "localhost",
                port=parsed_url.port or 8000,
                ssl=parsed_url.scheme == "https",
            )
            # Test connection by retrieving heartbeat
            self.chroma_client.heartbeat()

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

    async def _test_ollama_connection(self, url: str) -> tuple[str | None, set[str]]:
        """Test connection to Ollama.

        Returns a tuple of (error_code, downloaded_models).
        Error is None if connection successful, error code string otherwise.
        """
        downloaded_models: set[str] = set()

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
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception connecting to Ollama at %s", url)
            return "unknown", downloaded_models
        else:
            return None, downloaded_models


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
