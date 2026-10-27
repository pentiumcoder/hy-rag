# Northwind Engineering Handbook

This handbook describes how the Northwind Cloud engineering organization writes, reviews and ships code. It applies to every repository in the `northwind` GitHub organization. When this handbook and a team README disagree, this handbook wins.

## Code review

Every change to a production service goes through a pull request. Direct pushes to `main` are blocked by branch protection on all repositories.

- Standard services need **one approval** from a member of the owning team.
- Tier-1 services need **two approvals**, at least one from the owning team. The tier-1 services are `nw-gateway`, `keystone` and `ledger`, because an outage in any of them blocks every customer.
- Reviewers should respond within one business day. If a review has been waiting longer than that, ping the owning team's channel rather than merging without approval.
- Authors must not approve their own pull requests, even with admin rights.

Branch names start with a type prefix: `feat/`, `fix/`, `chore/` or `docs/`, followed by a short description, for example `fix/ingest-retry-jitter`.

## Testing expectations

Unit tests run on every pull request and must pass before merge. Services with an HTTP API also run contract tests against the published OpenAPI schema. Coverage must not drop by more than 1 percentage point in a single pull request; the CI bot comments with the delta.

## Deploying to production

Production deploys go through the weekly release train described in the Release Process document. Teams can also ship outside the train using `nwctl deploy`, provided the change has passed staging for at least 2 hours.

Production deploys are frozen on Fridays after 14:00 UTC and all weekend. The only exception is a fix for an active SEV1 or SEV2 incident, which needs explicit approval from the incident commander. A company-wide release freeze also runs for the last two weeks of December.

## Configuration changes

Production configuration is changed with `nwctl config set`, never by editing files on hosts. Since April 2026 every production config change requires a change ticket (`CHG-` number) and two approvers. The Configuration Reference lists every key, its default and its allowed range.

## Ownership

Each service has an owning team listed in `OWNERS.yaml` at the repository root. The owning team is responsible for its service's alerts, runbooks and on-call coverage.
