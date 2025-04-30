"""Service functions for the Places API (Queen's Guard)."""

from __future__ import annotations

import asyncio
from typing import Any

import googlemaps


class PlaceService:
    """Service class for Google Maps Places API."""

    def __init__(self, api_key: str) -> None:
        """Initialize the service with the API key."""
        self.client = googlemaps.Client(key=api_key)

    async def async_place_search(
        self,
        query: str,
        location: list[float] | None = None,
        radius: int | None = None,
    ) -> dict[str, Any]:
        """Search for places using a text query and optional location/radius."""

        return await asyncio.to_thread(
            self.client.places, query=query, location=location, radius=radius
        )

    async def async_get_directions(
        self, origin: str, destination: str, mode: str = "driving"
    ) -> dict[str, Any]:
        """Get directions from origin to destination."""

        return await asyncio.to_thread(
            self.client, origin=origin, destination=destination, mode=mode
        )

    async def async_geocode(self, address: str) -> dict[str, Any]:
        """Geocode an address to coordinates."""

        return await asyncio.to_thread(self.client.geocode, address)

    async def async_reverse_geocode(self, lat: float, lng: float) -> dict[str, Any]:
        """Reverse geocode coordinates to an address."""

        return await asyncio.to_thread(self.client.reverse_geocode, (lat, lng))
