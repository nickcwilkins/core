This is an LLM API called "Places" that provides tools for connecting to a mapping service.

This API is backed by google maps, but should not unnecessarily reference "google maps" (for example in text strings). Furthermore, try to implement code in a way that google maps can be swapped out with minimal changes.

Use the googlemaps python package.

Implementation steps:
1. Update config_flow.py to accept a google maps api key.
2. Create places/tools.py. Implement the following tools:
    - PlaceSearch: Search for nearby places based on a text query.
    - GetDirections: Get directions from an origin to a destination.
    - Geocode: Convert an address to geographic coordinates.
    - ReverseGeocode: Convert geographic coordinates into an address.
3. Create service.py for calling google maps services.