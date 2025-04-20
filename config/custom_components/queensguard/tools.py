"""LLM Tools for the Queen's Guard integration."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import LLMContext, Tool, ToolInput
from homeassistant.util.json import JsonObjectType

from .memories import MemoryManager


class AddMemoryTool(Tool):
    """Tool to add a new memory."""

    name = "add_memory"
    description = "Adds a new memory about the user or their preferences. Use this when you learn something new."
    parameters = vol.Schema(
        {
            vol.Required("memory_text"): str,
        }
    )

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Initialize the tool."""
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        await self.memory_manager.async_add_memory(tool_input.tool_args["memory_text"])

        return {}


class RemoveMemoryTool(Tool):
    """Tool to remove an existing memory."""

    name = "remove_memory"
    description = "Removes an existing memory using its unique ID (UUID). Use this if a memory is explicitly stated as wrong or irrelevant."
    parameters = vol.Schema(
        {
            vol.Required("memory_id"): str,
        }
    )

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Initialize the tool."""
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        await self.memory_manager.async_remove_memory(tool_input.tool_args["memory_id"])

        return {}


class UpdateMemoryTool(Tool):
    """Tool to update an existing memory."""

    name = "update_memory"
    description = "Updates an existing memory using its unique ID (UUID) with new text. Use this if a memory needs correction or refinement."
    parameters = vol.Schema(
        {
            vol.Required("memory_id"): str,
            vol.Required("new_memory_text"): str,
        }
    )

    def __init__(self, memory_manager: MemoryManager) -> None:
        """Initialize the tool."""
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        await self.memory_manager.async_update_memory(
            tool_input.tool_args["memory_id"],
            tool_input.tool_args["new_memory_text"],
        )

        return {}
