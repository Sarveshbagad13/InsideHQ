# NovaSphere API Documentation
The public API is versioned under `/api/v1`.

Authentication uses OAuth 2.0 access tokens.

Successful requests return JSON. Clients should inspect HTTP status codes before processing response bodies.

Common responses:
- 200: successful request
- 201: resource created
- 400: invalid request
- 401: authentication required or invalid
- 403: insufficient permission
- 404: resource not found
- 429: rate limit exceeded
- 500: internal server error

The default API rate limit is 100 requests per minute per user. Enterprise customers may receive higher limits under their contract.
