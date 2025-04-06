"""API for Queen's Guard integration."""

from __future__ import annotations

from collections.abc import Callable
import datetime
from functools import cache, partial
import logging
from typing import Any

import aiohttp
from slugify import slugify

from homeassistant.components.calendar import DOMAIN as CALENDAR_DOMAIN
from homeassistant.components.cover import (
    SERVICE_CLOSE_COVER as INTENT_CLOSE_COVER,
    SERVICE_OPEN_COVER as INTENT_OPEN_COVER,
)
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, intent
from homeassistant.helpers.event import async_track_state_change_event
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
MAX_CONTEXT_RESULTS = 5


class QueensGuardEmbeddingTool(Tool):
    """Tool to retrieve relevant context from embeddings."""

    name = "retrieve_context"
    description = "Retrieve relevant context from stored embeddings based on user query"

    def __init__(self, api: QueensGuardAPI) -> None:
        """Initialize the embedding tool."""
        super().__init__()
        self.api = api

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Retrieve context from embeddings."""
        query = tool_input.get("query")
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
    ) -> None:
        """Initialize the Queen's Guard API."""
        super().__init__(
            hass=hass,
            id=QUEENS_GUARD_API_ID,
            name="Queen's Guard",
        )
        self.chroma_url = chroma_url
        self.ollama_url = ollama_url
        self._embedding_update_remove_callbacks: list[Callable[[], None]] = []
        self.session: aiohttp.ClientSession | None = None
        self.cached_slugify = cache(partial(slugify, separator="_", lowercase=False))

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""
        if llm_context.assistant:
            exposed_entities: dict | None = _get_exposed_entities(
                self.hass, llm_context.assistant, include_state=False
            )
        else:
            exposed_entities = None

        return APIInstance(
            api=self,
            api_prompt=self._get_api_prompt(),
            llm_context=llm_context,
            tools=self._async_get_tools(llm_context, exposed_entities),
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

    async def async_setup_embedding_listeners(self) -> None:
        """Set up listeners to update embeddings when entity states change."""
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        # Get all entities exposed to voice assistants
        entity_registry = er.async_get(self.hass)
        exposed_entities = [
            entity_id
            for entity_id, entry in entity_registry.entities.items()
            if entry.exposed_to_voice_assistant
        ]

        if not exposed_entities:
            _LOGGER.warning("No entities exposed to voice assistants found")
            return

        _LOGGER.info(
            "Setting up embedding listeners for %s exposed entities",
            len(exposed_entities),
        )

        # Set up state change listener for exposed entities
        remove_listener = async_track_state_change_event(
            self.hass,
            exposed_entities,
            self._handle_entity_state_change,
        )
        self._embedding_update_remove_callbacks.append(remove_listener)

        # Generate initial embeddings for all exposed entities
        for entity_id in exposed_entities:
            state = self.hass.states.get(entity_id)
            if state:
                await self.async_store_embedding(entity_id, state)

    async def _handle_entity_state_change(self, event: Event) -> None:
        """Handle entity state change events."""
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")

        if not new_state:
            return

        await self.async_store_embedding(entity_id, new_state)

    async def async_store_embedding(self, entity_id: str, state: State) -> None:
        """Generate and store embeddings for an entity state."""
        if not self.session:
            _LOGGER.error("HTTP session not initialized")
            return

        try:
            # Create text representation of the entity state
            timestamp = state.last_updated or datetime.datetime.now()
            state_text = (
                f"Entity: {entity_id}\n"
                f"State: {state.state}\n"
                f"Last Updated: {timestamp.isoformat()}\n"
            )

            if state.attributes:
                state_text += "Attributes:\n"
                for key, value in state.attributes.items():
                    state_text += f"- {key}: {value}\n"

            # Generate embedding using Ollama
            embedding = await self._generate_embedding(state_text)
            if not embedding:
                return

            # Store in ChromaDB
            await self._store_in_chroma(entity_id, state_text, embedding)
            _LOGGER.debug("Stored embedding for %s", entity_id)

        except aiohttp.ClientError as err:
            _LOGGER.error("Network error storing embedding for %s: %s", entity_id, err)
        except HomeAssistantError as err:
            _LOGGER.error("Error storing embedding for %s: %s", entity_id, err)

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using Ollama."""
        if not self.session:
            return None

        try:
            async with self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            ) as resp:
                if resp.status != 200:
                    _LOGGER.error(
                        "Error generating embedding: %s - %s",
                        resp.status,
                        await resp.text(),
                    )
                    return None

                data = await resp.json()
                return data.get("embedding")

        except aiohttp.ClientError as err:
            _LOGGER.error("Error communicating with Ollama: %s", err)
            return None

    async def _store_in_chroma(
        self, entity_id: str, text: str, embedding: list[float]
    ) -> None:
        """Store embedding in ChromaDB."""
        if not self.session:
            return

        # Generate a unique ID for this entry based on entity_id and timestamp
        doc_id = f"{entity_id}_{datetime.datetime.now().timestamp()}"

        try:
            # ChromaDB collection name
            collection_name = "home_assistant_entities"

            # Check if collection exists
            async with self.session.get(
                f"{self.chroma_url}/api/v1/collections/{collection_name}"
            ) as resp:
                if resp.status == 404:
                    # Create collection
                    await self.session.post(
                        f"{self.chroma_url}/api/v1/collections",
                        json={"name": collection_name},
                    )

            # Add document to collection
            async with self.session.post(
                f"{self.chroma_url}/api/v1/collections/{collection_name}/add",
                json={
                    "ids": [doc_id],
                    "embeddings": [embedding],
                    "metadatas": [{"entity_id": entity_id}],
                    "documents": [text],
                },
            ) as resp:
                if resp.status != 200:
                    _LOGGER.error(
                        "Error storing in ChromaDB: %s - %s",
                        resp.status,
                        await resp.text(),
                    )

        except aiohttp.ClientError as err:
            _LOGGER.error("Error communicating with ChromaDB: %s", err)

    async def async_retrieve_relevant_context(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant context based on user query."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []

            # Query ChromaDB for similar documents
            collection_name = "home_assistant_entities"

            async with self.session.post(
                f"{self.chroma_url}/api/v1/collections/{collection_name}/query",
                json={
                    "query_embeddings": [query_embedding],
                    "n_results": MAX_CONTEXT_RESULTS,
                },
            ) as resp:
                if resp.status != 200:
                    _LOGGER.error(
                        "Error querying ChromaDB: %s - %s",
                        resp.status,
                        await resp.text(),
                    )
                    return []

                data = await resp.json()
                results = []

                # Process results
                for i, doc in enumerate(data.get("documents", [[]])[0]):
                    metadata = data.get("metadatas", [[]])[0][i]
                    distance = data.get("distances", [[]])[0][i]

                    results.append(
                        {
                            "document": doc,
                            "entity_id": metadata.get("entity_id"),
                            "relevance_score": 1.0
                            - distance,  # Convert distance to relevance score
                        }
                    )

                return results

        except aiohttp.ClientError as err:
            _LOGGER.error("Error retrieving context: %s", err)
            return []

    async def async_unload(self) -> None:
        """Unload the API and clean up resources."""
        for remove_callback in self._embedding_update_remove_callbacks:
            remove_callback()
        self._embedding_update_remove_callbacks.clear()

        if self.session:
            await self.session.close()
            self.session = None
