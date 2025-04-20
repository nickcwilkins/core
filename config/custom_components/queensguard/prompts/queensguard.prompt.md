This is a Home Assistant custom integration called "Queen's Guard" that houses LLM APIs for performing various tasks.

Each API is located in a subfolder with three possible files:

1. api.py: This file contains the API implementation.
    See the AssistAPI in homeassistant/helpers/llm.py for an example of what an LLM api should look like.
2. service.py: This file contains a "manager" type class or simple functions for connecting to outside services.
3. tools.py: This file contains tool classes that are used by the API.

APIs:
1. MemoryAPI: An api that provides "memory" RAG support to an assistant.
   Status: Complete
2. MapsAPI: A google maps backed API for finding information about places nearby.
   Status: Not Yet Implemented

Always reference the relevant documentation or files by fetching the web page or searching the codebase before editing code.

References:
For core home assistant functionality, check `homeassistant/core.py`
Selectors for config flows are in `homeassistant/helpers/selector.py`
For information about exposed entities, see `homeassistant/components/homeassistant/exposed_entities.py`

Area Registry: `homeassistant/helpers/area_registry.py`
Device Registry: `homeassistant/helpers/device_registry.py`
Entity Registry: `homeassistant/helpers/entity_registry.py`
Floor Registry: `homeassistant/helpers/floor_registry.py`