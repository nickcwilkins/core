"""Embeddings manager for Queen's Guard integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Literal, cast

from chromadb import Metadata
from chromadb.api import AsyncClientAPI
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import Embedding, OneOrMany
from chromadb.errors import ChromaError
import numpy as np
import ollama
from ollama import EmbeddingsResponse

from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.helpers.area_registry import (
    EVENT_AREA_REGISTRY_UPDATED,
    EventAreaRegistryUpdatedData,
)
from homeassistant.helpers.entity_registry import (
    EVENT_ENTITY_REGISTRY_UPDATED,
    EventEntityRegistryUpdatedData,
)
from homeassistant.helpers.floor_registry import (
    EVENT_FLOOR_REGISTRY_UPDATED,
    EventFloorRegistryUpdatedData,
)

_LOGGER = logging.getLogger(__name__)

ENTITY_COLLECTION_NAME = "home_assistant_entities"
ATTRIBUTE_COLLECTION_NAME = "home_assistant_attributes"

type EntityMetadataKey = Literal[
    "entity_id", "last_updated_timestamp", "area_id", "floor_id", "device_id"
]

type EntityMetadata = Mapping[EntityMetadataKey, str | int | float | bool | None]


@dataclass
class ContextResult:
    """Context result containing which entities and attributes are relevant."""

    entities: dict[str, EntityMetadata]
    attributes: set[str]

    def __init__(
        self, entities: dict[str, EntityMetadata], attributes: set[str]
    ) -> None:
        """Initialize the context result."""
        self.entities = entities
        self.attributes = attributes


class EmbeddingManager:
    """Manager for handling entity embeddings."""

    def __init__(
        self,
        hass: HomeAssistant,
        chroma_client: AsyncClientAPI,
        ollama_client: ollama.AsyncClient,
        model: str,
    ) -> None:
        """Initialize the embedding manager."""
        self.hass = hass
        self.chroma_client = chroma_client
        self.ollama_client = ollama_client
        self.model = model
        self._embedding_update_remove_callbacks: list[Callable[[], None]] = []
        self._entity_collection: AsyncCollection | None = None
        self._attr_collection: AsyncCollection | None = None

        self._tracked_attributes: dict[str, set[str | int | float | bool]] = {}

    async def _list_collections(self) -> Sequence[AsyncCollection]:
        """List ChromaDB collections."""
        return await self.chroma_client.list_collections()

    async def _add_to_collection(
        self,
        collection: AsyncCollection,
        ids: list[str],
        embeddings: OneOrMany[Embedding],
        metadatas: OneOrMany[Metadata],
        documents: list[str],
    ) -> None:
        """Add to ChromaDB collection."""
        await collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    async def async_setup(self) -> None:
        """Set up the embedding manager."""
        # Set up ChromaDB collections
        # Check if entity collection exists, create it if not

        # Delete collections if they exist for debugging
        collections = await self.chroma_client.list_collections()
        for collection in collections:
            await self.chroma_client.delete_collection(collection.name)

        configuration: CreateCollectionConfiguration = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 30,
                "ef_construction": 200,
                "ef_search": 200,
            },
            "embedding_function": None,
        }
        self._entity_collection = await self.chroma_client.get_or_create_collection(
            ENTITY_COLLECTION_NAME, configuration=configuration
        )

        # Check if attribute collection exists, create it if not
        self._attr_collection = await self.chroma_client.get_or_create_collection(
            ATTRIBUTE_COLLECTION_NAME,
            configuration=configuration,
        )
        _LOGGER.debug("ChromaDB collections set up successfully")

        # Get all entities exposed to voice assistants
        await self._async_setup_initial_embeddings()

        # Register event listeners
        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Set up event listeners for registry changes."""
        # Listen for entity state changes
        remove_state_changed = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED, self._async_handle_entity_state_changed
        )
        self._embedding_update_remove_callbacks.append(remove_state_changed)

        # Listen for entity registry changes
        remove_entity_registry = self.hass.bus.async_listen(
            EVENT_ENTITY_REGISTRY_UPDATED, self._async_handle_entity_registry_changed
        )
        self._embedding_update_remove_callbacks.append(remove_entity_registry)

        # Listen for area registry changes
        remove_area_registry = self.hass.bus.async_listen(
            EVENT_AREA_REGISTRY_UPDATED, self._handle_area_registry_changed
        )
        self._embedding_update_remove_callbacks.append(remove_area_registry)

        # Listen for floor registry changes
        remove_floor_registry = self.hass.bus.async_listen(
            EVENT_FLOOR_REGISTRY_UPDATED, self._handle_floor_registry_changed
        )
        self._embedding_update_remove_callbacks.append(remove_floor_registry)

    async def _async_handle_entity_state_changed(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Handle entity state changes."""
        entity_id = event.data["entity_id"]
        new_state = event.data["new_state"]
        is_exposed = async_should_expose(self.hass, CONVERSATION_DOMAIN, entity_id)
        if not is_exposed:
            return

        if new_state:
            _LOGGER.debug("Entity %s state changed to %s", entity_id, new_state.state)
            await self.async_store_entity(entity_id, new_state)
        else:
            await self._async_remove_entity_embeddings(entity_id)

    async def _async_handle_entity_registry_changed(
        self, event: Event[er.EventEntityRegistryUpdatedData]
    ) -> None:
        """Handle entity registry changes."""
        entity_id = event.data["entity_id"]
        if event.data["action"] == "remove":
            await self._async_remove_entity_embeddings(entity_id)
        elif event.data["action"] == "update":
            # Check if the entity is still exposed
            if async_should_expose(self.hass, CONVERSATION_DOMAIN, entity_id):
                entity_state = self.hass.states.get(entity_id)
                if entity_state:
                    # Store the new state embedding
                    await self.async_store_entity(entity_id, entity_state)
                else:
                    _LOGGER.warning(
                        "Entity %s state not found for embedding update", entity_id
                    )
            else:
                await self._async_remove_entity_embeddings(entity_id)
        elif event.data["action"] == "add":
            # Check if the entity is exposed
            if async_should_expose(self.hass, CONVERSATION_DOMAIN, entity_id):
                entity_state = self.hass.states.get(entity_id)
                if entity_state:
                    # Store the new state embedding
                    self.hass.async_create_task(
                        self.async_store_entity(entity_id, entity_state)
                    )
                else:
                    _LOGGER.warning(
                        "Entity %s state not found for embedding update", entity_id
                    )
        else:
            _LOGGER.warning(
                "Unknown action in entity registry event: %s", event.data["action"]
            )

    def _handle_area_registry_changed(
        self, event: Event[EventAreaRegistryUpdatedData]
    ) -> None:
        """Handle area registry changes."""
        area_id = event.data["area_id"]
        if event.data["action"] == "remove" or event.data["action"] == "update":
            # Find all entities associated with this area and update their embeddings
            entity_registry = er.async_get(self.hass)
            for entity_id, entry in entity_registry.entities.items():
                if entry.area_id == area_id and async_should_expose(
                    self.hass, CONVERSATION_DOMAIN, entity_id
                ):
                    entity_state = self.hass.states.get(entity_id)
                    if entity_state:
                        self.hass.async_create_task(
                            self.async_store_entity(entity_id, entity_state)
                        )

    def _handle_floor_registry_changed(
        self, event: Event[EventFloorRegistryUpdatedData]
    ) -> None:
        """Handle floor registry changes."""
        floor_id = event.data["floor_id"]
        if event.data["action"] == "remove" or event.data["action"] == "update":
            # Find all entities associated with this floor and update their embeddings
            entity_registry = er.async_get(self.hass)
            for entity_id, entry in entity_registry.entities.items():
                if entry.area_id == floor_id and async_should_expose(
                    self.hass, CONVERSATION_DOMAIN, entity_id
                ):
                    entity_state = self.hass.states.get(entity_id)
                    if entity_state:
                        self.hass.async_create_task(
                            self.async_store_entity(entity_id, entity_state)
                        )

    async def _async_setup_initial_embeddings(self) -> None:
        """Set up initial embeddings for all exposed entities, areas, and floors."""
        _LOGGER.debug("Setting up initial embeddings for exposed entities")

        if not self._entity_collection or not self._attr_collection:
            _LOGGER.error("ChromaDB collections not initialized")
            return

        # Get all entities currently exposed to voice assistants
        entity_registry = er.async_get(self.hass)
        exposed_entities = {
            entity_id
            for entity_id, entry in entity_registry.entities.items()
            if async_should_expose(self.hass, CONVERSATION_DOMAIN, entity_id)
            and entry.disabled_by is None
        }

        if not exposed_entities:
            _LOGGER.warning("No entities exposed to voice assistants found")
            return

        try:
            # Get entities currently stored in ChromaDB
            entity_result = await self._entity_collection.get()
            stored_entity_ids = set(entity_result.get("ids", []))

            # 1. Find entities in ChromaDB that are no longer exposed and remove them
            entities_to_remove = [
                entity_id
                for entity_id in stored_entity_ids
                if entity_id not in exposed_entities
            ]

            if entities_to_remove:
                await self._entity_collection.delete(ids=entities_to_remove)
                _LOGGER.debug(
                    "Removed %s entities that are no longer exposed",
                    len(entities_to_remove),
                )

            # 2. Find newly exposed entities and add them to the collection
            untracked_entities = exposed_entities - stored_entity_ids
            added_count = len(untracked_entities)
            for entity_id in untracked_entities:
                state = self.hass.states.get(entity_id)
                if state:
                    await self.async_store_entity(entity_id, state)
                else:
                    added_count -= 1

            if untracked_entities:
                _LOGGER.debug("Added %s newly exposed entities", added_count)

            # 3. Check for entities with updated states
            existing_entities = exposed_entities.intersection(stored_entity_ids)
            updated_count = 0

            # Create a mapping of entity_ids to their metadata for quick lookup
            entity_metadata_map = {}
            if entity_result["ids"] and entity_result["metadatas"]:
                for idx, entity_id in enumerate(entity_result["ids"]):
                    if entity_id in existing_entities:
                        entity_metadata_map[entity_id] = entity_result["metadatas"][idx]

            for entity_id in existing_entities:
                state = self.hass.states.get(entity_id)
                if state and entity_id in entity_metadata_map:
                    stored_timestamp = entity_metadata_map[entity_id].get(
                        "last_updated_timestamp"
                    )
                    if stored_timestamp != state.last_updated_timestamp:
                        await self.async_store_entity(entity_id, state)
                        updated_count += 1

            if updated_count > 0:
                _LOGGER.debug(
                    "Updated embeddings for %s entities with changed states",
                    updated_count,
                )

            # After all entities have been processed, process all unique attribute names
            await self._async_process_attribute_names()

        except (ChromaError, ValueError, ConnectionError) as err:
            _LOGGER.error("Error setting up initial embeddings: %s", err)

    async def _async_remove_entity_embeddings(self, entity_id: str) -> None:
        """Remove embeddings for a specific entity."""
        if not self._entity_collection or not self._attr_collection:
            return

        try:
            # Get embeddings for this entity based on metadata
            result = await self._entity_collection.get(where={"entity_id": entity_id})
            attr_result = await self._attr_collection.get(
                where={"entity_id": entity_id}
            )

            if result and result["ids"]:
                # Delete the embeddings
                await self._entity_collection.delete(ids=result["ids"])
                _LOGGER.debug(
                    "Removed %s embeddings for entity %s", len(result["ids"]), entity_id
                )

            if attr_result and attr_result["ids"]:
                # Delete the attribute embeddings
                await self._attr_collection.delete(ids=attr_result["ids"])
                _LOGGER.debug(
                    "Removed %s attribute embeddings for entity %s",
                    len(attr_result["ids"]),
                    entity_id,
                )
        except (ChromaError, ValueError, KeyError) as err:
            _LOGGER.error("Error removing embeddings for entity %s: %s", entity_id, err)

    async def _handle_entity_state_change(
        self, event: Event[EventEntityRegistryUpdatedData]
    ) -> None:
        """Handle entity state change events."""
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")

        if not new_state:
            return

        await self.async_store_entity(entity_id, new_state)

    async def async_store_entity(self, entity_id: str, state: State) -> None:
        """Generate and store embeddings for an entity state."""
        if not self._entity_collection or not self._attr_collection:
            _LOGGER.error("ChromaDB collections not initialized")
            return

        # Check for area
        entity_registry = er.async_get(self.hass)
        entity_entry = entity_registry.async_get(entity_id)
        area_entry: ar.AreaEntry | None = None
        floor_entry: fr.FloorEntry | None = None
        device_entry: dr.DeviceEntry | None = None

        if entity_entry:
            area_id = entity_entry.area_id
            if (device_id := entity_entry.device_id) is not None:
                device_entry = dr.async_get(self.hass).async_get(device_id)

            if not area_id and device_entry and device_entry.area_id:
                area_id = device_entry.area_id

            if entity_entry.area_id:
                area_entry = ar.async_get(self.hass).async_get_area(
                    entity_entry.area_id
                )
                if area_entry:
                    if area_entry.floor_id:
                        floor_entry = fr.async_get(self.hass).async_get_floor(
                            area_entry.floor_id
                        )

            if entity_entry.device_id:
                device_entry = dr.async_get(self.hass).async_get(entity_entry.device_id)

        else:
            _LOGGER.warning("Entity %s not found in registry", entity_id)
            return

        document_text: str
        area_name: str | None = None
        if area_entry:
            if area_entry.aliases:
                area_name = f"{area_entry.name}({', '.join(area_entry.aliases)})"
            else:
                area_name = area_entry.name

        floor_name: str | None = None
        if floor_entry:
            if floor_entry.aliases:
                floor_name = f"{floor_entry.name}({', '.join(floor_entry.aliases)})"
            else:
                floor_name = floor_entry.name

        entity_text: str
        if area_name and not floor_name:
            entity_text = (
                f"{state.name} is a {state.domain} entity located in {area_name}.\n"
            )
        elif area_name and floor_name:
            entity_text = f"{state.name} is a {state.domain} entity located in {area_name} on {floor_name}.\n"
        else:
            entity_text = f"{state.name} is a {state.domain} entity.\n"

        state_text = f"{state.name} is currently {state.state}."

        attribute_text = ""

        filtered_attributes = {
            key: value
            for key, value in state.attributes.items()
            if key
            not in [
                "supported_features",
                "friendly_name",
                "entity_id",
                "icon",
                "name",
            ]
        }

        if filtered_attributes:
            attribute_text = f"{state.name} has the following attributes:\n"
            for key, value in filtered_attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    attribute_text += f"  {key}: {value}\n"

                attr_values = self._tracked_attributes.setdefault(key, set())
                if len(attr_values) < 5 or key in ["device_class"]:
                    if isinstance(value, (str, int, float, bool)):
                        attr_values.add(value)

        # Generate embedding using Ollama
        document_text = f"{entity_text}{state_text}"
        if attribute_text:
            document_text += f"\n{attribute_text}"

        embedding = await self._generate_embedding(document_text)
        if embedding is None:
            return

        metadata: Metadata = {
            "entity_id": entity_id,
            "last_updated_timestamp": state.last_updated_timestamp,
        }
        if area_entry:
            metadata["area_id"] = area_entry.id
        if floor_entry:
            metadata["floor_id"] = floor_entry.floor_id
        if device_entry:
            metadata["device_id"] = device_entry.id

        await self._entity_collection.upsert(
            [entity_id],
            np.array(embedding, dtype=np.float32),
            metadata,
            [document_text],
        )
        _LOGGER.debug("Stored embedding for %s", entity_id)

    async def _async_process_attribute_names(self) -> None:
        """Process and store embeddings for all unique attribute names.

        Each document id in this collection corresponds to the attribute name
        (e.g., "temperature", "humidity"). The document text is 5 lines of text,
        each line containing the attribute name, and a unique value from an entity
        that has this attribute.
        """
        if not self._attr_collection:
            _LOGGER.error("Attribute name collection not initialized")
            return

        # Get current attribute name documents
        result = await self._attr_collection.get()
        existing_names = set()

        if result and result.get("ids"):
            for doc_id in result["ids"]:
                existing_names.add(doc_id)

        # Process only new attribute names
        new_names = set(self._tracked_attributes.keys()) - existing_names

        if not new_names:
            _LOGGER.debug("No new attribute names to process")
            return

        _LOGGER.info("Processing %s new attribute names", len(new_names))

        for attribute_name in new_names:
            attr_text = f"Attribute Name: {attribute_name}"
            embedding = await self._generate_embedding(attr_text)
            if embedding is None:
                continue

            attr_doc_id = f"{attribute_name}"
            await self._attr_collection.upsert(
                [attr_doc_id],
                np.array(embedding, dtype=np.float32),
                {"attribute_name": attribute_name},
                [attr_text],
            )
            _LOGGER.debug("Stored embedding for attribute name %s", attribute_name)

    async def _generate_embedding(self, text: str) -> EmbeddingsResponse | None:
        """Generate embedding for text using Ollama."""
        response = await self.ollama_client.embeddings(
            model=self.model,
            prompt=text,
        )
        return response.get("embedding")

    async def async_retrieve_relevant_context(self, query: str) -> ContextResult | None:
        """Retrieve relevant entities and attributes based on user query."""
        if not self._entity_collection or not self._attr_collection:
            _LOGGER.error("ChromaDB collections not initialized")
            return None

        embedding = await self._generate_embedding(query)
        if not embedding:
            return None
        entity_results = await self._entity_collection.query(
            np.array(embedding, dtype=np.float32),
            include=["metadatas", "distances", "documents"],
        )
        _LOGGER.debug("Found %s relevant entities", len(entity_results["ids"][0]))
        attr_results = await self._attr_collection.query(
            np.array(embedding, dtype=np.float32), include=[]
        )
        _LOGGER.debug("Found %s relevant attributes", len(attr_results["ids"][0]))

        # Process entity results
        if not entity_results["metadatas"]:
            _LOGGER.error("Entity results contained no metadata")
            return None

        entities: dict[str, EntityMetadata] = cast(
            dict[str, EntityMetadata],
            dict(
                zip(
                    entity_results["ids"][0],
                    entity_results["metadatas"][0],
                    strict=False,
                )
            ),
        )
        attributes = set(attr_results["ids"][0])

        return ContextResult(entities, attributes)

    async def async_unload(self) -> None:
        """Unload the embedding manager and clean up resources."""
        for remove_callback in self._embedding_update_remove_callbacks:
            remove_callback()
        self._embedding_update_remove_callbacks.clear()
