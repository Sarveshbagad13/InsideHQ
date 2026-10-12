# Deployment Guide
Code is deployed through the CI/CD pipeline.

Pull requests must pass automated tests, linting, and security checks before merging.

Production deployments require approval from the responsible engineering owner.

Database migrations must be backward compatible whenever possible.

A deployment should include a rollback plan when the change has meaningful production risk.

After deployment, engineers monitor application health, error rates, latency, and key business metrics.

Emergency deployments may bypass the normal release window but must be documented afterward.
