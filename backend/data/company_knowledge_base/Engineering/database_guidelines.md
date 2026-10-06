# Database Guidelines
PostgreSQL is the primary transactional database.

Database schema changes must be reviewed before production deployment.

Indexes should be added based on measured query requirements rather than speculation.

Destructive migrations require a documented rollback or recovery strategy.

Production data must not be copied into developer environments unless approved and appropriately sanitized.

Database credentials are managed through the secrets system.
