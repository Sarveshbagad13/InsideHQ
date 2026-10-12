# NovaSphere System Architecture
NovaSphere products use a service-oriented architecture.

The public API layer handles authentication, request validation, rate limiting, and routing.

Application services contain business logic. PostgreSQL is used for transactional relational data, while Redis is used for caching and short-lived state.

Object storage is used for uploaded documents and generated reports.

Services communicate asynchronously through the internal event bus for selected workflows.

Production deployments run in isolated cloud environments. Secrets are stored in a managed secrets system and must never be committed to source control.
