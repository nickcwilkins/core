Goal: Build a Home Assistant Integration that exposes an llm.API for Voice Assistants with RAG support.

Requirements:
Connection to external ChromaDB for storing embeddings.
Connection to external Ollama instance for generating embeddings.
Update embeddings each time the state of a entity exposed to Voice Assistants is updated
Fetch relevant embeddings and automatically add that context to the LLM prompt when receiving a user chat message.

Add logging for debugging issues.

For now, expose the same tools as the AssistAPI, but replace the API prompt with our embeddings.

Use the chromadb python package for chromadb. See [Client](https://docs.trychroma.com/reference/python/client) and [Collection](https://docs.trychroma.com/reference/python/collection)

Use the ollama python package for ollama.