Goal: Build a Home Assistant Integration that exposes an llm.API for Voice Assistants with RAG support for retrieving relevant home context. This will allow exposing more entities to the LLM as every entity won't be provided as context for every chat request.

Implementation Steps:
1. Create config_flow.py for configuring the integration. See [Config Flow Handler](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/), for setting up a config flow. This is the config flow:
    - Step One: Setup weaviate and ollama connections, test that the connections are valid.
    - Step Two: After verifying the connections, list the available models and allow the user to select a downloaded embedding model to use.
        - The listed models should come from ollama, similar to how the ollama integration gets downloaded_models in `homeassistant/components/ollama/config_flow.py`
        - If the model does not exist, allow download the model with ollama and wait for the download to complete, similar to the ollama integration config_flow.
2. in async_setup_entry in `__init__.py`, store the ollama and weaviate client instances in the config entry, similar to the ollama integration in `homeassistant/components/ollama/__init__.py`
3. Create an llm API Instance similar to HomeAssistants AssistAPI.
    - Expose the same tools as the Assist API
    - When generating the prompt, use the ContextResult from the embeddings manager to filter the results
4. Create an embeddings manager in embeddings.py which sets up listeners so that each time the state of an entity exposed to voice assistants changes, we update the embeddings for that entity, area, or floor.
    - When the integration config entry is loaded (in async_setup_entry), check that the embeddings are up to date for all currently exposed entities.
        - If an entity is no longer exposed, remove it from chroma.
        - If there is a new entity that doesn't exist in the weaviate, add it.
    - When new entities, floors, or areas are added, add the embeddings for them.
    - Create a collection specifically for attributes, where the document id is the attribute name and the text is also the attribute name.

General Guidance:
This integration only supports a single config entry.

Store all constants in const.py

Use the weaviate python package for weaviate. Use the ollama python package for ollama. Do not use an http client.

Add logging for debugging issues. Each step should have some associated logging.

References:
For core home assistant functionality, check `homeassistant/core.py`
Selectors for config flows are in `homeassistant/helpers/selector.py`
For home assistant llm API helpers, see: `homeassistant/helpers/llm.py`
For information about exposed entities, see `homeassistant/components/homeassistant/exposed_entities.py`

Area Registry: homeassistant/helpers/area_registry.py
Device Registry: homeassistant/helpers/device_registry.py
Entity Registry: homeassistant/helpers/entity_registry.py
Floor Registry: homeassistant/helpers/floor_registry.py
