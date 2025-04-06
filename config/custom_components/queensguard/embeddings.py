"""Embeddings manager for Queen's Guard integration."""

from __future__ import annotations

from collections.abc import Callable
import datetime
import logging
from typing import Any

import aiohttp

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.core import Event, HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_track_state_change_event

_LOGGER = logging.getLogger(__name__)

COLLECTION_NAME = "home_assistant_entities"
MAX_CONTEXT_RESULTS = 5


class EmbeddingManager:
    """Manager for handling entity embeddings."""

    def __init__(
        self,
        hass: HomeAssistant,
        chroma_url: str,
        ollama_url: str,
    ) -> None:
        """Initialize the embedding manager."""
        self.hass = hass
        self.chroma_url = chroma_url
        self.ollama_url = ollama_url
        self._embedding_update_remove_callbacks: list[Callable[[], None]] = []
        self.session: aiohttp.ClientSession | None = None

    async def async_setup(self) -> None:
        """Set up the embedding manager."""
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        # Get all entities exposed to voice assistants
        entity_registry = er.async_get(self.hass)
        exposed_entities = [
            entity_id
            for entity_id, entry in entity_registry.entities.items()
            if async_should_expose(self.hass, "conversation", entity_id)
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
            # Check if collection exists
            async with self.session.get(
                f"{self.chroma_url}/api/v1/collections/{COLLECTION_NAME}"
            ) as resp:
                if resp.status == 404:
                    # Create collection
                    await self.session.post(
                        f"{self.chroma_url}/api/v1/collections",
                        json={"name": COLLECTION_NAME},
                    )

            # Add document to collection
            async with self.session.post(
                f"{self.chroma_url}/api/v1/collections/{COLLECTION_NAME}/add",
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
            async with self.session.post(
                f"{self.chroma_url}/api/v1/collections/{COLLECTION_NAME}/query",
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
        """Unload the embedding manager and clean up resources."""
        for remove_callback in self._embedding_update_remove_callbacks:
            remove_callback()
        self._embedding_update_remove_callbacks.clear()

        if self.session:
            await self.session.close()
            self.session = None
