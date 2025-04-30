"""LLM Tools for the Places API (Queen's Guard)."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import LLMContext, Tool, ToolInput
from homeassistant.util.json import JsonObjectType

from .service import PlaceService


class PlaceSearchTool(Tool):
    """Tool to search for nearby places based on a text query."""

    name = "place_search"
    description = "Search for nearby places based on a text query. Optionally provide a location (lat, lng) and radius in meters."
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
            vol.Optional("location"): vol.All(list, vol.ExactSequence([float, float])),
            vol.Optional("radius"): int,
        }
    )

    def __init__(self, service: PlaceService) -> None:
        """Initialize the tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
        self.service = service

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        return await self.service.async_place_search(
            tool_input.tool_args["query"],
            tool_input.tool_args.get("location"),
            tool_input.tool_args.get("radius"),
        )


class GetDirectionsTool(Tool):
    """Tool to get directions from an origin to a destination."""

    name = "get_directions"
    description = "Get directions from an origin to a destination. Optionally specify a mode (driving, walking, bicycling, transit)."
    parameters = vol.Schema(
        {
            vol.Required("origin"): str,
            vol.Required("destination"): str,
            vol.Optional("mode", default="driving"): str,
        }
    )

    def __init__(self, service: PlaceService) -> None:
        """Initialize the tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
        self.service = service

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        return await self.service.async_get_directions(
            tool_input.tool_args["origin"],
            tool_input.tool_args["destination"],
            tool_input.tool_args.get("mode", "driving"),
        )


class GeocodeTool(Tool):
    """Tool to convert an address to geographic coordinates."""

    name = "geocode"
    description = "Convert an address to geographic coordinates."
    parameters = vol.Schema(
        {
            vol.Required("address"): str,
        }
    )

    def __init__(self, service: PlaceService) -> None:
        """Initialize the tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
        self.service = service

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        return await self.service.async_geocode(tool_input.tool_args["address"])


class ReverseGeocodeTool(Tool):
    """Tool to convert geographic coordinates into an address."""

    name = "reverse_geocode"
    description = "Convert geographic coordinates (lat, lng) into an address."
    parameters = vol.Schema(
        {
            vol.Required("lat"): float,
            vol.Required("lng"): float,
        }
    )

    def __init__(self, service: PlaceService) -> None:
        """Initialize the tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
        self.service = service

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> JsonObjectType:
        """Call the tool."""
        return await self.service.async_reverse_geocode(
            tool_input.tool_args["lat"],
            tool_input.tool_args["lng"],
        )
