"""Util functions for Queensguard."""

from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.core import EventStateChangedData


def _handle_entity_state_changed_filter(self, data: EventStateChangedData) -> bool:
    """Filter for entity state changed events."""
    entity_id = data["entity_id"]
    return async_should_expose(self.hass, CONVERSATION_DOMAIN, entity_id)
