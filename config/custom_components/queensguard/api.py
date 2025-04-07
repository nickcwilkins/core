"""API for Queen's Guard integration."""

from __future__ import annotations

from functools import cache, partial
import logging
from typing import Any

from slugify import slugify

from homeassistant.components.calendar import DOMAIN as CALENDAR_DOMAIN
from homeassistant.components.cover import (
    SERVICE_CLOSE_COVER as INTENT_CLOSE_COVER,
    SERVICE_OPEN_COVER as INTENT_OPEN_COVER,
)
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.helpers.llm import (
    API,
    APIInstance,
    CalendarGetEventsTool,
    GetHomeStateTool,
    IntentTool,
    LLMContext,
    ScriptTool,
    Tool,
    ToolInput,
    _get_exposed_entities,
)
from homeassistant.util.json import JsonObjectType

_LOGGER = logging.getLogger(__name__)

# Intent constant not imported from elsewhere
INTENT_GET_WEATHER = "GetWeather"

QUEENS_GUARD_API_ID = "queens_guard"


class QueensGuardEmbeddingTool(Tool):
    """Tool to retrieve relevant context from embeddings."""

    name = "retrieve_context"
    description = "Retrieve relevant context from stored embeddings based on user query"

    def __init__(self, api: QueensGuardAPI) -> None:
        """Initialize the embedding tool."""
        self.api = api

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Retrieve context from embeddings."""
        query = llm_context.user_prompt
        if not query:
            return {
                "success": False,
                "message": "No query provided",
            }

        try:
            context_results = await self.api.async_retrieve_relevant_context(query)

        except HomeAssistantError as err:
            _LOGGER.error("Error retrieving context: %s", err)
            return {
                "success": False,
                "message": f"Error retrieving context: {err}",
            }
        else:
            return {
                "success": True,
                "results": context_results,
            }


class QueensGuardAPI(API):
    """API exposing Queen's Guard RAG capabilities to LLMs."""

    # Copy of the intents to ignore from AssistAPI
    IGNORE_INTENTS = {
        intent.INTENT_GET_TEMPERATURE,
        INTENT_GET_WEATHER,
        INTENT_OPEN_COVER,  # deprecated
        INTENT_CLOSE_COVER,  # deprecated
        intent.INTENT_GET_STATE,
        intent.INTENT_NEVERMIND,
        intent.INTENT_TOGGLE,
        intent.INTENT_GET_CURRENT_DATE,
        intent.INTENT_GET_CURRENT_TIME,
        intent.INTENT_RESPOND,
    }

    def __init__(
        self,
        hass: HomeAssistant,
        chroma_url: str,
        ollama_url: str,
        embedding_manager=None,
    ) -> None:
        """Initialize the Queen's Guard API."""
        super().__init__(
            hass=hass,
            id=QUEENS_GUARD_API_ID,
            name="Queen's Guard",
        )
        self.embedding_manager = embedding_manager
        self.chroma_url = chroma_url
        self.ollama_url = ollama_url
        self.cached_slugify = cache(partial(slugify, separator="_", lowercase=False))
        _LOGGER.info("Initialized Queen's Guard API")

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""
        if llm_context.assistant:
            _LOGGER.debug(
                "Getting exposed entities for assistant: %s", llm_context.assistant
            )
            exposed_entities: dict | None = await self._async_get_api_prompt(
                llm_context
            )
        else:
            _LOGGER.debug("No assistant specified, skipping exposed entities")
            exposed_entities = None

        _LOGGER.debug("Creating API instance with tools")
        return APIInstance(
            api=self,
            api_prompt=self._get_api_prompt(),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, exposed_entities),
        )

    async def _async_get_api_prompt(self, llm_context: LLMContext) -> dict | None:
        """Get API prompt with relevant context from embeddings if available."""
        if not llm_context.user_prompt:
            _LOGGER.debug("No user prompt available for context retrieval")
            return _get_exposed_entities(
                self.hass, llm_context.assistant, include_state=False
            )

        _LOGGER.debug("Fetching relevant context for API prompt")
        return _get_exposed_entities(
            self.hass, llm_context.assistant, include_state=False
        )

    @callback
    def _get_api_prompt(self) -> str:
        """Return the prompt for the API."""
        return (
            "Queen's Guard provides advanced RAG capabilities to enhance your responses. "
            "Use the retrieve_context tool to get relevant information from stored embeddings "
            "when answering user questions about their smart home.\n\n"
            "The tool will search for context related to any entities mentioned in the "
            "user's question, their current states, and historical values.\n\n"
            "When controlling Home Assistant always call the intent tools. "
            "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
            "When controlling a device, prefer passing just name and domain. "
            "When controlling an area, prefer passing just area name and domain."
        )

    @callback
    def _async_get_tools(
        self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> list[Tool]:
        """Return a list of LLM tools."""
        # Start with our unique RAG tool
        tools: list[Tool] = [QueensGuardEmbeddingTool(self)]

        # Add all the same tools from AssistAPI
        ignore_intents = self.IGNORE_INTENTS
        if not llm_context.device_id or not async_device_supports_timers(
            self.hass, llm_context.device_id
        ):
            ignore_intents = ignore_intents | {
                intent.INTENT_START_TIMER,
                intent.INTENT_CANCEL_TIMER,
                intent.INTENT_INCREASE_TIMER,
                intent.INTENT_DECREASE_TIMER,
                intent.INTENT_PAUSE_TIMER,
                intent.INTENT_UNPAUSE_TIMER,
                intent.INTENT_TIMER_STATUS,
            }

        intent_handlers = [
            intent_handler
            for intent_handler in intent.async_get(self.hass)
            if intent_handler.intent_type not in ignore_intents
        ]

        exposed_domains: set[str] | None = None
        if exposed_entities is not None:
            exposed_domains = {
                info["domain"] for info in exposed_entities["entities"].values()
            }

            intent_handlers = [
                intent_handler
                for intent_handler in intent_handlers
                if intent_handler.platforms is None
                or intent_handler.platforms & exposed_domains
            ]

        tools.extend(
            [
                IntentTool(
                    self.cached_slugify(intent_handler.intent_type), intent_handler
                )
                for intent_handler in intent_handlers
            ]
        )

        if exposed_entities:
            if exposed_entities[CALENDAR_DOMAIN]:
                names = []
                for info in exposed_entities[CALENDAR_DOMAIN].values():
                    names.extend(info["names"].split(", "))
                tools.append(CalendarGetEventsTool(names))

            tools.extend(
                ScriptTool(self.hass, script_entity_id)
                for script_entity_id in exposed_entities[SCRIPT_DOMAIN]
            )

        if exposed_domains:
            tools.append(GetHomeStateTool())

        return tools

    async def async_retrieve_relevant_context(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant context based on user query."""
        if self.embedding_manager is None:
            raise HomeAssistantError("Embedding manager is not initialized")

        _LOGGER.debug("Retrieving relevant context for query: %s", query)
        try:
            return await self.embedding_manager.async_retrieve_relevant_context(query)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error retrieving context: %s", err)
            raise HomeAssistantError(f"Failed to retrieve context: {err}") from err

    async def async_unload(self) -> None:
        """Unload the API and clean up resources."""
        _LOGGER.debug("Unloading Queen's Guard API")
        # Nothing specific to clean up here, embedding manager is handled separately
