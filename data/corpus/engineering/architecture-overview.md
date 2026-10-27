# Northwind Cloud Architecture Overview

Northwind Cloud is a hosted event analytics platform. Customers send product events through the API, Northwind stores them, and customers query them with SQL or through dashboards. This page describes the main services and how data flows between them.

## Request path

1. Every external request enters through `nw-gateway`, the API gateway. It terminates TLS, authenticates the API key or bearer token with `keystone`, and enforces per-key rate limits.
2. Event batches sent to `POST /v2/events` are validated by the gateway and written to the Kafka topic `events.raw`.
3. `ingestd`, the ingestion service, consumes `events.raw`, enriches events, and inserts them into the ClickHouse cluster `ch-events` in batches.
4. Queries go to `queryd`, which compiles customer SQL, applies tenant isolation and runs it against ClickHouse. Results are cached in Redis.

## Services

| Service | Responsibility | Tier | Owning team |
|---|---|---|---|
| `nw-gateway` | API gateway, auth check, rate limiting | Tier-1 | Edge |
| `keystone` | API keys, tokens, SSO | Tier-1 | Identity |
| `ingestd` | Kafka consumer, enrichment, ClickHouse inserts | Tier-2 | Data Platform |
| `queryd` | SQL compilation and execution | Tier-2 | Query |
| `ledger` | Usage metering and invoicing | Tier-1 | Billing |
| `notifier` | Webhooks and email notifications | Tier-3 | Integrations |

## Datastores

- **Postgres (`pg-main`)**: accounts, API keys, billing records. Primary in us-east with a synchronous replica and an asynchronous replica in eu-west.
- **ClickHouse (`ch-events`)**: customer event data, sharded by tenant.
- **Kafka**: the `events.raw` topic buffers incoming events for 72 hours, so ingestion can fall behind without losing data.
- **Redis**: query result cache and rate-limit counters.

## Regions

Northwind runs in three regions: us-east, eu-west and ap-south. Customers choose a home region at signup and their event data never leaves it. Accounts with EU data residency use the `api.eu-west.northwind.example` endpoint.

## Observability

All services export Prometheus metrics and structured JSON logs. Dashboards live in Grafana; the most used are "Ingest Overview", "Gateway Errors" and "Query Latency". Alerts are routed to the on-call engineer through Beacon, our paging tool.
