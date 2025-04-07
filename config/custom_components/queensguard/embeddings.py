"""Embeddings manager for Queen's Guard integration."""

from __future__ import annotations

from collections.abc import Callable
import datetime
import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
import ollama
from ollama import ResponseError

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.helpers.area_registry import EVENT_AREA_REGISTRY_UPDATED
from homeassistant.helpers.entity_registry import EVENT_ENTITY_REGISTRY_UPDATED
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.floor_registry import EVENT_FLOOR_REGISTRY_UPDATED

_LOGGER = logging.getLogger(__name__)

COLLECTION_NAME = "home_assistant_entities"
MAX_CONTEXT_RESULTS = 5
CLEANUP_INTERVAL = datetime.timedelta(days=7)  # Clean up old embeddings once per week


class EmbeddingManager:
    """Manager for handling entity embeddings."""

    def __init__(
        self,
        hass: HomeAssistant,
        chroma_client: chromadb.Client,
        ollama_client: ollama.AsyncClient,
        model: str,
    ) -> None:
        """Initialize the embedding manager."""
        self.hass = hass
        self.chroma_client = chroma_client
        self.ollama_client = ollama_client
        self.model = model
        self._embedding_update_remove_callbacks: list[Callable[[], None]] = []
        self._collection: Collection | None = None
        self._tracked_entities: set[str] = set()
        self._tracked_areas: set[str] = set()
        self._tracked_floors: set[str] = set()

    def _list_collections(self) -> list[Collection]:
        """List ChromaDB collections in executor to avoid blocking I/O."""
        return self.chroma_client.list_collections()

    def _create_collection(self, name: str) -> Collection:
        """Create ChromaDB collection in executor to avoid blocking I/O."""
        return self.chroma_client.create_collection(name=name)

    def _get_collection(self, name: str) -> Collection:
        """Get ChromaDB collection in executor to avoid blocking I/O."""
        return self.chroma_client.get_collection(name=name)

    def _add_to_collection(
        self,
        collection: Collection,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Add to ChromaDB collection in executor to avoid blocking I/O."""
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def _delete_from_collection(
        self,
        collection: Collection,
        ids: list[str],
    ) -> None:
        """Delete from ChromaDB collection in executor to avoid blocking I/O."""
        collection.delete(ids=ids)

    def _query_collection(
        self,
        collection: Collection,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> QueryResult:
        """Query ChromaDB collection in executor to avoid blocking I/O."""
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )

    async def async_setup(self) -> None:
        """Set up the embedding manager."""
        # Set up ChromaDB collection
        try:
            # Check if collection exists, create it if not
            collections = await self.hass.async_add_executor_job(self._list_collections)

            if not any(coll.name == COLLECTION_NAME for coll in collections):
                _LOGGER.info("Creating new ChromaDB collection: %s", COLLECTION_NAME)
                self._collection = await self.hass.async_add_executor_job(
                    self._create_collection, COLLECTION_NAME
                )
            else:
                _LOGGER.info("Using existing ChromaDB collection: %s", COLLECTION_NAME)
                self._collection = await self.hass.async_add_executor_job(
                    self._get_collection, COLLECTION_NAME
                )
        except (chromadb.errors.ChromaError, ValueError, ConnectionError) as err:
            _LOGGER.error("Error setting up ChromaDB collection: %s", err)
            return

        # Get all entities exposed to voice assistants
        await self._async_setup_initial_embeddings()

        # Register event listeners
        self._setup_event_listeners()

        # Set up periodic cleanup
        cleanup_remove = async_track_time_interval(
            self.hass,
            self._async_cleanup_old_embeddings,
            CLEANUP_INTERVAL,
        )
        self._embedding_update_remove_callbacks.append(cleanup_remove)

    async def _async_setup_initial_embeddings(self) -> None:
        """Set up initial embeddings for all exposed entities, areas, and floors."""
        entity_registry = er.async_get(self.hass)
        exposed_entities = [
            entity_id
            for entity_id, entry in entity_registry.entities.items()
            if async_should_expose(self.hass, "conversation", entity_id)
            and entry.disabled_by is None
        ]

        if not exposed_entities:
            _LOGGER.warning("No entities exposed to voice assistants found")
            return

        _LOGGER.info(
            "Setting up embedding listeners for %s exposed entities with model %s",
            len(exposed_entities),
            self.model,
        )

        # Track entities we're monitoring
        self._tracked_entities.update(exposed_entities)

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

        # Process areas
        area_registry = ar.async_get(self.hass)
        for area in area_registry.async_list_areas():
            await self._async_process_area(area)
            self._tracked_areas.add(area.id)

        # Process floors
        floor_registry = fr.async_get(self.hass)
        for floor in floor_registry.async_list_floors():
            await self._async_process_floor(floor)
            self._tracked_floors.add(floor.floor_id)

    def _setup_event_listeners(self) -> None:
        """Set up event listeners for registry changes."""
        # Listen for entity registry changes
        remove_entity_registry = self.hass.bus.async_listen(
            EVENT_ENTITY_REGISTRY_UPDATED, self._handle_entity_registry_changed
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

    @callback
    def _handle_entity_registry_changed(self, event: Event) -> None:
        """Handle entity registry changes with callback pattern."""
        self.hass.async_create_task(self._async_entity_registry_changed(event))

    async def _async_entity_registry_changed(self, event: Event) -> None:
        """Handle entity registry changes."""
        entity_registry = er.async_get(self.hass)
        data = event.data
        entity_id = data.get("entity_id")

        if not entity_id:
            return

        if data.get("action") == "remove":
            # Remove entity from embeddings if it was tracked
            if entity_id in self._tracked_entities:
                _LOGGER.debug("Entity %s removed, removing from embeddings", entity_id)
                await self._async_remove_entity_embeddings(entity_id)
                self._tracked_entities.discard(entity_id)
            return

        # Get the entity entry
        entry = entity_registry.async_get(entity_id)
        if not entry:
            return

        # Check if the entity's exposure status has changed
        is_exposed = async_should_expose(self.hass, "conversation", entity_id)
        was_tracked = entity_id in self._tracked_entities

        # Check if the entity is disabled
        is_disabled = entry.disabled_by is not None

        if is_exposed and not is_disabled and not was_tracked:
            # Entity is newly exposed - add to tracked list and create embedding
            _LOGGER.debug("Entity %s newly exposed, adding to embeddings", entity_id)
            self._tracked_entities.add(entity_id)
            state = self.hass.states.get(entity_id)
            if state:
                await self.async_store_embedding(entity_id, state)
        elif (not is_exposed or is_disabled) and was_tracked:
            # Entity is no longer exposed - remove from tracked list and remove embeddings
            _LOGGER.debug(
                "Entity %s no longer exposed, removing from embeddings", entity_id
            )
            await self._async_remove_entity_embeddings(entity_id)
            self._tracked_entities.discard(entity_id)

    @callback
    def _handle_area_registry_changed(self, event: Event) -> None:
        """Handle area registry changes with callback pattern."""
        self.hass.async_create_task(self._async_area_registry_changed(event))

    async def _async_area_registry_changed(self, event: Event) -> None:
        """Handle area registry changes."""
        area_registry = ar.async_get(self.hass)
        data = event.data
        area_id = data.get("area_id")

        if not area_id:
            return

        if data.get("action") == "remove":
            # Remove area from embeddings if it was tracked
            if area_id in self._tracked_areas:
                _LOGGER.debug("Area %s removed, removing from embeddings", area_id)
                await self._async_remove_area_embeddings(area_id)
                self._tracked_areas.discard(area_id)
            return

        # Get the area entry
        area = area_registry.async_get_area(area_id)
        if not area:
            return

        # Process new or updated area
        if area_id not in self._tracked_areas:
            _LOGGER.debug("New area %s found, adding to embeddings", area.name)
            self._tracked_areas.add(area_id)
        else:
            _LOGGER.debug("Area %s updated, updating embeddings", area.name)

        await self._async_process_area(area)

    @callback
    def _handle_floor_registry_changed(self, event: Event) -> None:
        """Handle floor registry changes with callback pattern."""
        self.hass.async_create_task(self._async_floor_registry_changed(event))

    async def _async_floor_registry_changed(self, event: Event) -> None:
        """Handle floor registry changes."""
        floor_registry = fr.async_get(self.hass)
        data = event.data
        floor_id = data.get("floor_id")

        if not floor_id:
            return

        if data.get("action") == "remove":
            # Remove floor from embeddings if it was tracked
            if floor_id in self._tracked_floors:
                _LOGGER.debug("Floor %s removed, removing from embeddings", floor_id)
                await self._async_remove_floor_embeddings(floor_id)
                self._tracked_floors.discard(floor_id)
            return

        # Get the floor entry
        floor = floor_registry.async_get_floor(floor_id)
        if not floor:
            return

        # Process new or updated floor
        if floor_id not in self._tracked_floors:
            _LOGGER.debug("New floor %s found, adding to embeddings", floor.name)
            self._tracked_floors.add(floor_id)
        else:
            _LOGGER.debug("Floor %s updated, updating embeddings", floor.name)

        await self._async_process_floor(floor)

    async def _async_process_area(self, area: ar.AreaEntry) -> None:
        """Process an area for embeddings."""
        if not self._collection:
            return

        try:
            # Create text representation of the area
            area_text = f"Area: {area.name}\n"
            if area.aliases:
                area_text += f"Aliases: {', '.join(area.aliases)}\n"

            if area.floor_id:
                floor_registry = fr.async_get(self.hass)
                if floor := floor_registry.async_get_floor(area.floor_id):
                    area_text += f"Floor: {floor.name}\n"

            # Get devices in this area
            device_registry = dr.async_get(self.hass)
            devices_in_area = [
                device
                for device in device_registry.devices.values()
                if device.area_id == area.id
            ]

            if devices_in_area:
                area_text += f"Number of devices: {len(devices_in_area)}\n"
                area_text += "Devices:\n"
                for device in devices_in_area:
                    area_text += f"- {device.name or device.id}\n"

            # Generate embedding for the area
            embedding = await self._generate_embedding(area_text)
            if not embedding:
                return

            # Store in ChromaDB
            doc_id = f"area_{area.id}_{datetime.datetime.now().timestamp()}"

            await self.hass.async_add_executor_job(
                self._add_to_collection,
                self._collection,
                [doc_id],
                [embedding],
                [{"area_id": area.id}],
                [area_text],
            )
            _LOGGER.debug("Stored embedding for area %s", area.name)

        except (chromadb.errors.ChromaError, HomeAssistantError) as err:
            _LOGGER.error("Error storing embedding for area %s: %s", area.name, err)

    async def _async_process_floor(self, floor: fr.FloorEntry) -> None:
        """Process a floor for embeddings."""
        if not self._collection:
            return

        try:
            # Create text representation of the floor
            floor_text = f"Floor: {floor.name}\n"

            # Get areas on this floor
            area_registry = ar.async_get(self.hass)
            areas_on_floor = [
                area
                for area in area_registry.async_list_areas()
                if area.floor_id == floor.floor_id
            ]

            if areas_on_floor:
                floor_text += f"Number of areas: {len(areas_on_floor)}\n"
                floor_text += "Areas:\n"
                for area in areas_on_floor:
                    floor_text += f"- {area.name}\n"

            # Generate embedding for the floor
            embedding = await self._generate_embedding(floor_text)
            if not embedding:
                return

            # Store in ChromaDB
            doc_id = f"floor_{floor.floor_id}_{datetime.datetime.now().timestamp()}"

            await self.hass.async_add_executor_job(
                self._add_to_collection,
                self._collection,
                [doc_id],
                [embedding],
                [{"floor_id": floor.floor_id}],
                [floor_text],
            )
            _LOGGER.debug("Stored embedding for floor %s", floor.name)

        except (chromadb.errors.ChromaError, HomeAssistantError) as err:
            _LOGGER.error("Error storing embedding for floor %s: %s", floor.name, err)

    async def _async_remove_entity_embeddings(self, entity_id: str) -> None:
        """Remove all embeddings for a specific entity."""
        if not self._collection:
            return

        try:
            # Get all embeddings for this entity
            result = self._collection.get(where={"entity_id": entity_id})

            if result and result["ids"]:
                # Delete the embeddings
                await self.hass.async_add_executor_job(
                    self._delete_from_collection,
                    self._collection,
                    result["ids"],
                )
                _LOGGER.debug(
                    "Removed %s embeddings for entity %s", len(result["ids"]), entity_id
                )
        except (chromadb.errors.ChromaError, ValueError, KeyError) as err:
            _LOGGER.error("Error removing embeddings for entity %s: %s", entity_id, err)

    async def _async_remove_area_embeddings(self, area_id: str) -> None:
        """Remove all embeddings for a specific area."""
        if not self._collection:
            return

        try:
            # Get all embeddings for this area
            result = self._collection.get(where={"area_id": area_id})

            if result and result["ids"]:
                # Delete the embeddings
                await self.hass.async_add_executor_job(
                    self._delete_from_collection,
                    self._collection,
                    result["ids"],
                )
                _LOGGER.debug(
                    "Removed %s embeddings for area %s", len(result["ids"]), area_id
                )
        except (chromadb.errors.ChromaError, ValueError, KeyError) as err:
            _LOGGER.error("Error removing embeddings for area %s: %s", area_id, err)

    async def _async_remove_floor_embeddings(self, floor_id: str) -> None:
        """Remove all embeddings for a specific floor."""
        if not self._collection:
            return

        try:
            # Get all embeddings for this floor
            result = self._collection.get(where={"floor_id": floor_id})

            if result and result["ids"]:
                # Delete the embeddings
                await self.hass.async_add_executor_job(
                    self._delete_from_collection,
                    self._collection,
                    result["ids"],
                )
                _LOGGER.debug(
                    "Removed %s embeddings for floor %s", len(result["ids"]), floor_id
                )
        except (chromadb.errors.ChromaError, ValueError, KeyError) as err:
            _LOGGER.error("Error removing embeddings for floor %s: %s", floor_id, err)

    async def _async_cleanup_old_embeddings(
        self, _now: datetime.datetime | None = None
    ) -> None:
        """Clean up old embeddings, keeping only the most recent for each entity, area and floor."""
        if not self._collection:
            return

        _LOGGER.debug("Running scheduled cleanup of old embeddings")

        try:
            # Get all documents
            result = self._collection.get()

            if not result or not result["ids"]:
                return

            # Group by entity_id, area_id, and floor_id
            entities: dict[str, list[tuple[str, float]]] = {}
            areas: dict[str, list[tuple[str, float]]] = {}
            floors: dict[str, list[tuple[str, float]]] = {}

            for i, doc_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i]

                # Parse the timestamp from the ID
                try:
                    if "_" in doc_id and doc_id.rsplit("_", 1)[1]:
                        timestamp = float(doc_id.rsplit("_", 1)[1])
                    else:
                        # Can't parse timestamp, skip this document
                        continue
                except (ValueError, IndexError):
                    continue

                if "entity_id" in metadata:
                    entity_id = metadata["entity_id"]
                    entities.setdefault(entity_id, []).append((doc_id, timestamp))
                elif "area_id" in metadata:
                    area_id = metadata["area_id"]
                    areas.setdefault(area_id, []).append((doc_id, timestamp))
                elif "floor_id" in metadata:
                    floor_id = metadata["floor_id"]
                    floors.setdefault(floor_id, []).append((doc_id, timestamp))

            # Find IDs to delete (keeping only the most recent for each entity/area/floor)
            ids_to_delete = []

            for entity_docs in entities.values():
                if len(entity_docs) <= 1:
                    continue
                # Sort by timestamp descending
                entity_docs.sort(key=lambda x: x[1], reverse=True)
                # Keep the most recent, delete the rest
                ids_to_delete.extend(doc_id for doc_id, _ in entity_docs[1:])

            for area_docs in areas.values():
                if len(area_docs) <= 1:
                    continue
                # Sort by timestamp descending
                area_docs.sort(key=lambda x: x[1], reverse=True)
                # Keep the most recent, delete the rest
                ids_to_delete.extend(doc_id for doc_id, _ in area_docs[1:])

            for floor_docs in floors.values():
                if len(floor_docs) <= 1:
                    continue
                # Sort by timestamp descending
                floor_docs.sort(key=lambda x: x[1], reverse=True)
                # Keep the most recent, delete the rest
                ids_to_delete.extend(doc_id for doc_id, _ in floor_docs[1:])

            if ids_to_delete:
                await self.hass.async_add_executor_job(
                    self._delete_from_collection,
                    self._collection,
                    ids_to_delete,
                )
                _LOGGER.info("Cleaned up %s old embeddings", len(ids_to_delete))

        except (chromadb.errors.ChromaError, ValueError, KeyError) as err:
            _LOGGER.error("Error during embeddings cleanup: %s", err)

    async def _handle_entity_state_change(self, event: Event) -> None:
        """Handle entity state change events."""
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")

        if not new_state:
            return

        await self.async_store_embedding(entity_id, new_state)

    async def async_store_embedding(self, entity_id: str, state: State) -> None:
        """Generate and store embeddings for an entity state."""
        if not self._collection:
            _LOGGER.error("ChromaDB collection not initialized")
            return

        try:
            # Create text representation of the entity state
            timestamp = state.last_updated or datetime.datetime.now()
            state_text = (
                f"Entity: {entity_id}\n"
                f"Name: {state.name}\n"
                f"State: {state.state}\n"
                f"Last Updated: {timestamp.isoformat()}\n"
            )

            if state.attributes:
                state_text += "Attributes:\n"
                for key, value in state.attributes.items():
                    state_text += f"- {key}: {value}\n"

            # Check for area
            entity_registry = er.async_get(self.hass)
            area_registry = ar.async_get(self.hass)
            floor_registry = fr.async_get(self.hass)

            entity_entry = entity_registry.async_get(entity_id)
            if entity_entry:
                area_id = None
                device_id = entity_entry.device_id

                # Get area from entity or device
                if entity_entry.area_id:
                    area_id = entity_entry.area_id
                elif device_id:
                    device_registry = dr.async_get(self.hass)
                    device = device_registry.async_get(device_id)
                    if device and device.area_id:
                        area_id = device.area_id

                # Add area and floor info if available
                if area_id and (area := area_registry.async_get_area(area_id)):
                    state_text += f"Area: {area.name}\n"
                    if area.floor_id and (
                        floor := floor_registry.async_get_floor(area.floor_id)
                    ):
                        state_text += f"Floor: {floor.name}\n"

            # Generate embedding using Ollama
            embedding = await self._generate_embedding(state_text)
            if embedding is None:
                return

            # Generate a unique ID for this entry based on entity_id and timestamp
            doc_id = f"{entity_id}_{datetime.datetime.now().timestamp()}"

            # Store in ChromaDB (run in executor to avoid blocking I/O)
            await self.hass.async_add_executor_job(
                self._add_to_collection,
                self._collection,
                [doc_id],
                [embedding],
                [{"entity_id": entity_id}],
                [state_text],
            )
            _LOGGER.debug("Stored embedding for %s", entity_id)

        except Exception as err:
            _LOGGER.error("Error storing embedding for %s: %s", entity_id, err)
            raise

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using Ollama."""
        try:
            response = await self.ollama_client.embeddings(
                model=self.model,
                prompt=text,
            )
            return response.get("embedding")
        except (ResponseError, ConnectionError) as err:
            _LOGGER.error("Error generating embedding: %s", err)
            return None

    async def async_retrieve_relevant_context(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant context based on user query."""
        if not self._collection:
            _LOGGER.error("ChromaDB collection not initialized")
            return []

        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []

            # Query ChromaDB for similar documents (run in executor to avoid blocking I/O)
            results = await self.hass.async_add_executor_job(
                self._query_collection,
                self._collection,
                [query_embedding],
                MAX_CONTEXT_RESULTS,
            )

            processed_results = []

            # Process results
            if (
                results.get("documents")
                and results.get("distances")
                and results.get("metadatas")
            ):
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    processed_results.append(
                        {
                            "document": doc,
                            "entity_id": metadata.get("entity_id"),
                            "area_id": metadata.get("area_id"),
                            "floor_id": metadata.get("floor_id"),
                            "relevance_score": 1.0
                            - float(distance),  # Convert distance to relevance score
                        }
                    )

                _LOGGER.debug(
                    "Retrieved %s relevant context items for query",
                    len(processed_results),
                )
            else:
                _LOGGER.debug("No relevant context found for query")

        except (chromadb.errors.ChromaError, ResponseError, ConnectionError) as err:
            _LOGGER.error("Error retrieving context: %s", err)
            return []
        else:
            return processed_results

    async def async_unload(self) -> None:
        """Unload the embedding manager and clean up resources."""
        for remove_callback in self._embedding_update_remove_callbacks:
            remove_callback()
        self._embedding_update_remove_callbacks.clear()
