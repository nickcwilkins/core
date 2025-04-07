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
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    floor_registry as fr,
    intent,
)
from homeassistant.helpers.llm import (
    API,
    APIInstance,
    CalendarGetEventsTool,
    GetHomeStateTool,
    IntentTool,
    LLMContext,
    ScriptTool,
    Tool,
    _get_exposed_entities,
)
from homeassistant.util import yaml as yaml_util

from .embeddings import EmbeddingManager

_LOGGER = logging.getLogger(__name__)

# Intent constant not imported from elsewhere
INTENT_GET_WEATHER = "GetWeather"

QUEENS_GUARD_API_ID = "queensguard"


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
        embedding_manager: EmbeddingManager = None,
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
            exposed_entities: dict | None = await self._async_get_api_prompt_entities(
                llm_context
            )
        else:
            _LOGGER.debug("No assistant specified, skipping exposed entities")
            exposed_entities = None

        _LOGGER.debug("Creating API instance with tools")
        return APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(llm_context, exposed_entities),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, exposed_entities),
            custom_serializer=None,
        )

    async def _async_get_api_prompt_entities(
        self, llm_context: LLMContext
    ) -> dict | None:
        """Get API prompt entities with relevant context from embeddings if available."""
        base_entities = _get_exposed_entities(
            self.hass, llm_context.assistant, include_state=False
        )

        if not llm_context.user_prompt or not self.embedding_manager:
            _LOGGER.debug("No user prompt or embedding manager, using default entities")
            return base_entities

        try:
            _LOGGER.debug("Retrieving relevant context for user query")
            context_results = (
                await self.embedding_manager.async_retrieve_relevant_context(
                    llm_context.user_prompt
                )
            )

            _LOGGER.debug("Retrieved %d context results", len(context_results))

        except Exception as err:
            _LOGGER.error("Error retrieving context: %s", err)
            raise
        else:
            return base_entities

    @callback
    def _async_get_api_prompt(
        self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> str:
        """Return the prompt for the API."""
        if not exposed_entities or not exposed_entities["entities"]:
            return "Only if the user wants to control a device, tell them to expose entities to their voice assistant in Home Assistant."

        return "\n".join(
            [
                *self._async_get_preamble(llm_context),
                *self._async_get_exposed_entities_prompt(llm_context, exposed_entities),
            ]
        )

    @callback
    def _async_get_preamble(self, llm_context: LLMContext) -> list[str]:
        """Return the preamble for the API."""
        prompt = [
            (
                "Queen's Guard provides advanced RAG capabilities to enhance your responses. "
                "When controlling Home Assistant always call the intent tools. "
                "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
                "When controlling a device, prefer passing just name and domain. "
                "When controlling an area, prefer passing just area name and domain."
            )
        ]

        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)

            extra = "and all generic commands like 'turn on the lights' should target this area."

        if floor and area:
            prompt.append(f"You are in area {area.name} (floor {floor.name}) {extra}")
        elif area:
            prompt.append(f"You are in area {area.name} {extra}")
        else:
            prompt.append(
                "When a user asks to turn on all devices of a specific type, "
                "ask user to specify an area, unless there is only one device of that type."
            )

        if not llm_context.device_id or not async_device_supports_timers(
            self.hass, llm_context.device_id
        ):
            prompt.append("This device is not able to start timers.")

        return prompt

    @callback
    def _async_get_exposed_entities_prompt(
        self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> list[str]:
        """Return the prompt for the API for exposed entities."""
        prompt = []

        if exposed_entities and exposed_entities["entities"]:
            prompt.append(
                "An overview of the areas and the devices in this smart home:"
            )
            prompt.append(yaml_util.dump(list(exposed_entities["entities"].values())))

        return prompt

    @callback
    def _async_get_tools(
        self, llm_context: LLMContext, exposed_entities: dict | None
    ) -> list[Tool]:
        """Return a list of LLM tools."""
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

        tools: list[Tool] = [
            IntentTool(self.cached_slugify(intent_handler.intent_type), intent_handler)
            for intent_handler in intent_handlers
        ]

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
