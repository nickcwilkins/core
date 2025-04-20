"""Memory manager for Queen's Guard integration using Weaviate."""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import logging
from typing import TypedDict
import uuid

import ollama
import weaviate
import weaviate.classes as wvc

from homeassistant.core import HomeAssistant

# pylint: disable=hass-relative-import
from queensguard.const import WEAVIATE_MEMORY_COLLECTION

_LOGGER = logging.getLogger(__name__)


class Memory(TypedDict):
    """TypedDict for memory objects in Weaviate."""

    id: str
    memory_text: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    relevance: float


class MemoryManager:
    """Manages storing and retrieving memories using Weaviate and Ollama."""

    def __init__(
        self,
        hass: HomeAssistant,
        weaviate_client: weaviate.WeaviateAsyncClient,
        ollama_client: ollama.AsyncClient,
        embedding_model: str,
    ) -> None:
        """Initialize the MemoryManager."""
        self.hass = hass
        self.weaviate_client = weaviate_client
        self.ollama_client = ollama_client
        self.embedding_model = embedding_model
        self._memory_collection: weaviate.collections.CollectionAsync | None = None
        _LOGGER.info(
            "Initialized MemoryManager with Weaviate client and Ollama client (Model: %s)",
            embedding_model,
        )

    async def async_setup(self) -> None:
        """Set up the Weaviate collection for memories."""
        _LOGGER.debug(
            "Setting up Weaviate memory collection: %s", WEAVIATE_MEMORY_COLLECTION
        )

        collection_exists = await self.weaviate_client.collections.exists(
            WEAVIATE_MEMORY_COLLECTION
        )

        if not collection_exists:
            _LOGGER.info(
                "Memory collection '%s' does not exist, creating it",
                WEAVIATE_MEMORY_COLLECTION,
            )

            self._memory_collection = await self.weaviate_client.collections.create(
                name=WEAVIATE_MEMORY_COLLECTION,
                properties=[
                    wvc.config.Property(
                        name="memory_text",
                        data_type=wvc.config.DataType.TEXT,
                    )
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
            )
            _LOGGER.info(
                "Successfully created memory collection '%s'",
                WEAVIATE_MEMORY_COLLECTION,
            )
        else:
            _LOGGER.debug(
                "Memory collection '%s' already exists", WEAVIATE_MEMORY_COLLECTION
            )
            # Get the collection object for later use
            self._memory_collection = self.weaviate_client.collections.get(
                WEAVIATE_MEMORY_COLLECTION
            )

    async def _generate_embedding(self, text: str) -> Sequence[float]:
        """Generate embedding for text using Ollama."""
        response = await self.ollama_client.embeddings(
            model=self.embedding_model,
            prompt=text,
        )
        return response.embedding

    async def async_add_memory(self, memory_text: str) -> None:
        """Add a new memory to the Weaviate collection.

        Args:
            memory_text: The text content of the memory to add.

        Returns:
            The UUID of the added memory, or None if embedding failed.

        """
        assert self._memory_collection is not None
        _LOGGER.debug("Adding new memory: %s", memory_text[:100])

        embedding = await self._generate_embedding(memory_text)
        memory_uuid = uuid.uuid4()

        await self._memory_collection.data.insert(
            properties={"memory_text": memory_text},
            uuid=memory_uuid,
            vector=embedding,
        )
        _LOGGER.info("Successfully added memory with UUID: %s", memory_uuid)

    async def async_remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the Weaviate collection by its UUID.

        Args:
            memory_id: The UUID of the memory to remove.

        """
        assert self._memory_collection is not None
        _LOGGER.debug("Attempting to remove memory with UUID: %s", memory_id)

        result = await self._memory_collection.data.delete_by_id(uuid=memory_id)
        if result:
            _LOGGER.info("Successfully removed memory with UUID: %s", memory_id)
        else:
            _LOGGER.warning("Failed to remove memory with UUID: %s", memory_id)

    async def async_update_memory(self, memory_id: str, new_memory_text: str) -> None:
        """Update an existing memory in the Weaviate collection.

        Args:
            memory_id: The UUID of the memory to update.
            new_memory_text: The new text content for the memory.

        """
        assert self._memory_collection is not None
        _LOGGER.debug("Attempting to update memory with UUID: %s", memory_id)

        new_embedding = await self._generate_embedding(new_memory_text)

        await self._memory_collection.data.update(
            uuid=memory_id,
            properties={"memory_text": new_memory_text},
            vector=new_embedding,
        )
        _LOGGER.info("Successfully updated memory with UUID: %s", memory_id)

    async def async_retrieve_relevant_memories(
        self, query: str, limit: int = 5
    ) -> list[Memory]:
        """Retrieve relevant memories based on a query string.

        Args:
            query: The query string to find relevant memories for.
            limit: The maximum number of memories to retrieve.

        Returns:
            A list of relevant Memory objects, or an empty list if none found or embedding failed.

        """
        assert self._memory_collection is not None
        _LOGGER.debug("Retrieving relevant memories for query: %s", query[:100])

        query_embedding = await self._generate_embedding(query)

        response = await self._memory_collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(
                distance=True,
                creation_time=True,
                last_update_time=True,
            ),
            return_properties=["memory_text"],
        )

        memories: list[Memory] = []
        for obj in response.objects:
            try:
                memory_id = str(obj.uuid)
                memory_text = obj.properties.get("memory_text", "")
                creation_time = obj.metadata.creation_time
                last_update_time = obj.metadata.last_update_time
                relevance = (
                    1.0 - obj.metadata.distance
                    if obj.metadata and obj.metadata.distance is not None
                    else 0.0
                )

                if not memory_text:
                    _LOGGER.warning("Memory text is empty for UUID: %s", memory_id)
                    continue

                memories.append(
                    {
                        "id": memory_id,
                        "memory_text": memory_text,
                        "created_at": creation_time,
                        "updated_at": last_update_time,
                        "relevance": relevance,
                    }
                )
            except (ValueError, AttributeError, KeyError) as err:
                _LOGGER.warning("Failed to parse memory object: %s", err)
        _LOGGER.debug(
            "Retrieved %d memories, distances: %s",
            len(memories),
            [obj.metadata.distance for obj in response.objects],
        )
        return memories
