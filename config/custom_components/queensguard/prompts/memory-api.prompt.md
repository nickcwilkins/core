The following is a step-by-step list of implementation steps for building the MemoryAPI.

1. Create config_flow.py for configuring the integration. See [Config Flow Handler](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/), for setting up a config flow. This is the config flow:
    - Step One: Setup weaviate and ollama connections, test that the connections are valid.
    - Step Two: After verifying the connections, list the available models and allow the user to select a downloaded embedding model to use.
        - The listed models should come from ollama, similar to how the ollama integration gets downloaded_models in `homeassistant/components/ollama/config_flow.py`
        - If the model does not exist, allow download the model with ollama and wait for the download to complete, similar to the ollama integration config_flow.
    [Config Entry Documentation](https://developers.home-assistant.io/docs/config_entries_index)
    [Data Entry Flow Documentation](https://developers.home-assistant.io/docs/data_entry_flow_index)
2. in async_setup_entry in `__init__.py`, store the ollama and weaviate client instances in the config entry, similar to the ollama integration in `homeassistant/components/ollama/__init__.py`
3. Create an llm Memory API using the Home Assistant AssistAPI in `homeassistant/helpers/llm.py` as a reference.
    - Create tools for adding memories, removing memories, and updating memories.
    - Do not provide a tool for getting memories. This will happen automatically in step 5.
    - The prompt should include a sentence about using the memory tools anytime the assistant learns something relevant about the user.
4. Create a memory manager for the api service in memory/service.py which will handle interacting with weaviate for managing memories.
    Weaviate Documentation:
    [Managing Collections](https://weaviate.io/developers/weaviate/manage-data/collections)
    [Creating Objects](https://weaviate.io/developers/weaviate/manage-data/create)
    [Searching/Retrieval](https://weaviate.io/developers/weaviate/concepts/search)
5. When generating the prompt, automatically include memories that are deemed relevant. The assistant should never have to retrieve it's own memories, it can only add new ones or update them.

General Guidance:
Store all constants in const.py

Use the weaviate python package for weaviate. Use the ollama python package for ollama. Do not use an http client.

We are using weaviate v4. Don't use the v3 API. Here is a link to the migration guide: https://weaviate.io/developers/weaviate/client-libraries/python/v3_v4_migration
