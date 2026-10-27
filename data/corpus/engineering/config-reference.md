# Configuration Reference

This reference lists the production configuration keys that operators change most often. Read a value with `nwctl config get <key> --env prod`. Changing a production value requires a change ticket and two approvers (see the Engineering Handbook). Values outside the allowed range are rejected by config validation.

## Ingestion (`ingestd`)

### ingest.max_batch_size
Maximum number of events accepted in one `POST /v2/events` request and processed by `ingestd` in one insert.
- Default: `500`
- Allowed range: 1 to 5000
- Exceeding it returns error `NW-4012 BATCH_TOO_LARGE`.
The hard upper limit of 5000 has been enforced by validation since release v4.12, after the March 2026 ingest outage.

### ingest.flush_interval_ms
How long `ingestd` waits to fill a batch before inserting a partial one.
- Default: `250`
- Allowed range: 50 to 5000

### ingest.retry.max_attempts
How many times `ingestd` retries a failed ClickHouse insert before sending the batch to the dead-letter topic `events.dlq`.
- Default: `5`

### ingest.retry.backoff_base_ms
Base delay for exponential retry backoff, doubled on every attempt and capped at 10 seconds.
- Default: `200`

### ingest.consumer.replicas
Number of `ingestd` consumer pods per region.
- Default: `6`
- Maximum: `24` (one per Kafka partition)

## Gateway (`nw-gateway`)

### gateway.rate_limit.rps
Sustained requests per second allowed per API key.
- Default: `100`
- Plan overrides: Starter 50, Growth 100, Enterprise 1000.
- Exceeding it returns `NW-4290 RATE_LIMITED`.

### gateway.rate_limit.burst
Short burst allowance, expressed as a multiple of `gateway.rate_limit.rps`.
- Default: `2` (twice the sustained rate for up to 10 seconds)

## Query (`queryd`)

### queryd.timeout_ms
Maximum execution time of a single query.
- Default: `30000` (30 seconds)
- Exceeding it returns `NW-5002 QUERY_TIMEOUT`.

### queryd.cache.ttl_seconds
How long query results stay in the Redis cache.
- Default: `300`

### queryd.max_rows
Maximum rows returned by one query; larger results must be paginated.
- Default: `100000`

## Identity (`keystone`)

### keystone.token_ttl_minutes
Lifetime of an access token issued by `POST /v2/auth/token`.
- Default: `60`
- Allowed range: 15 to 1440 (bounds added after the June 2026 token incident)

### keystone.refresh_token_ttl_days
Lifetime of a refresh token.
- Default: `30`

## Billing (`ledger`)

### ledger.invoice_day
Day of the month on which invoices are generated.
- Default: `1`

## Webhooks (`notifier`)

### webhooks.max_retries
- Default: `8`

### webhooks.timeout_seconds
- Default: `10`
