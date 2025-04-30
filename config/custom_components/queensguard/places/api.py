"""API for the Places LLM integration."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import API, APIInstance, LLMContext, Tool

from .service import PlaceService
from .tools import GeocodeTool, GetDirectionsTool, PlaceSearchTool, ReverseGeocodeTool

_LOGGER = logging.getLogger(__name__)

PLACES_API_ID = "places"


class PlaceAPI(API):
    """API exposing Places tools to LLMs."""

    def __init__(self, hass: HomeAssistant, service: PlaceService) -> None:
        """Initialize the Places API."""
        super().__init__(
            hass=hass,
            id=PLACES_API_ID,
            name="Places",
        )
        self._service = service
        _LOGGER.debug("Initialized Places API")

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Return the instance of the API."""
        return APIInstance(
            api=self,
            api_prompt=self._async_get_api_prompt(),
            llm_context=llm_context,
            tools=self._async_get_tools(),
            custom_serializer=None,
        )

    def _async_get_api_prompt(self) -> str:
        """Return the prompt for the API."""
        return (
            "You have access to tools for searching for places, getting directions, and geocoding. "
            "Use these tools to answer questions about locations, addresses, and navigation."
        )

    def _async_get_tools(self) -> list[Tool]:
        """Return a list of LLM tools for places and navigation."""
        return [
            PlaceSearchTool(self._service),
            GetDirectionsTool(self._service),
            GeocodeTool(self._service),
            ReverseGeocodeTool(self._service),
        ]

    async def async_unload(self) -> None:
        """Unload the API and clean up resources."""
        _LOGGER.debug("Unloading Places API")
