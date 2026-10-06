# Testing Strategy
NovaSphere uses unit, integration, API, and end-to-end tests.

Unit tests should cover business logic and important edge cases.

Integration tests verify interactions between services, databases, and external dependencies.

End-to-end tests cover critical customer workflows.

A pull request should not be merged when required automated checks are failing unless an explicit exception is approved.

Production defects should normally result in a regression test when practical.
