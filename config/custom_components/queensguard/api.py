"""API for Queen's Guard integration."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import API, APIInstance, LLMContext, Tool

from .memories import MemoryManager

# Import the new tool classes
from .tools import AddMemoryTool, RemoveMemoryTool, UpdateMemoryTool

_LOGGER = logging.getLogger(__name__)

QUEENS_GUARD_API_ID = "queensguard"


class QueensGuardAPI(API):
    """API exposing Queen's Guard RAG capabilities to LLMs."""

    def __init__(self, hass: HomeAssistant, memory_manager: MemoryManager) -> None:
        """Initialize the Queen's Guard API."""
        super().__init__(
            hass=hass,
            id=QUEENS_GUARD_API_ID,
            name="Queen's Guard Memory",  # Renamed for clarity
        )
        self.memory_manager = memory_manager
        _LOGGER.info("Initialized Queen's Guard API with Memory Manager")

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""

        relevant_memories: list[str] = []
        if llm_context.user_prompt:
            relevant_memories = (
                await self.memory_manager.async_retrieve_relevant_memories(
                    llm_context.user_prompt
                )
            )

        return APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(relevant_memories),
            llm_context=llm_context,
            tools=self._async_get_tools(),
            custom_serializer=None,
        )

    def _async_get_api_prompt(self, relevant_memories: list[str]) -> str:
        """Return the prompt for the API, including relevant memories."""
        prompt_parts = [
            "You have access to tools for managing memories about the user.",
            "Use these tools whenever you learn something new or need to update existing information.",
        ]

        if relevant_memories:
            prompt_parts.append("\nHere are some potentially relevant memories:")
            for memory in relevant_memories:
                # Limit memory length in prompt if necessary
                prompt_parts.extend(
                    f"- {memory[:200]}{'...' if len(memory) > 200 else ''}"
                )

        return "\n".join(prompt_parts)

    def _async_get_tools(self) -> list[Tool]:
        """Return a list of LLM tools for memory management."""
        # Instantiate tools with the memory manager instance
        return [
            AddMemoryTool(self.memory_manager),
            RemoveMemoryTool(self.memory_manager),
            UpdateMemoryTool(self.memory_manager),
        ]

    async def async_unload(self) -> None:
        """Unload the API and clean up resources."""
        _LOGGER.debug("Unloading Queen's Guard API")
        # Memory manager is handled by __init__.py unload
