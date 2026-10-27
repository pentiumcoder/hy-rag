# REST API v2 Reference

Base URL: `https://api.northwind.example/v2`. Customers with EU data residency must use `https://api.eu-west.northwind.example/v2`.

## Authentication

Send your API key in the `X-NW-Api-Key` header. Keys are created in the dashboard under Settings → API Keys. Live keys start with `nw_live_` and test keys with `nw_test_`. Test keys write to a sandbox and are never billed.

For short-lived credentials, exchange a key for a bearer token:

- `POST /v2/auth/token` returns an access token (valid for 60 minutes by default) and a refresh token (valid for 30 days).
- `POST /v2/auth/refresh` exchanges a refresh token for a new access token.

Expired access tokens return `NW-4031 TOKEN_EXPIRED`.

## Endpoints

### POST /v2/events
Send a batch of events. The body is a JSON object with an `events` array. A batch may contain at most `ingest.max_batch_size` events, which is 500 unless your account has a higher limit (never above 5000). Larger batches are rejected with `NW-4012 BATCH_TOO_LARGE`. Returns `202 Accepted` with the number of events accepted.

### GET /v2/query
Run a SQL query over your events. Parameters: `sql` (required), `page_size` and `cursor`. Queries time out after 30 seconds with `NW-5002 QUERY_TIMEOUT`.

### POST /v2/query/async
Start a long-running query. Returns a `query_id`; poll `GET /v2/query/async/{query_id}` for the result. Asynchronous queries can run for up to 15 minutes.

### GET /v2/usage
Return event volume and query counts for the current billing period.

## Pagination

List and query endpoints are paginated with cursors. Pass the `next_cursor` value from one response as `cursor` in the next request. `page_size` defaults to 100 and can be at most 1000.

## Rate limits

Every response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers. Limits are applied per API key and depend on your plan. Exceeding them returns `NW-4290 RATE_LIMITED` with a `Retry-After` header.

## Versioning

v2 is the current version. v1 was retired on 2025-12-31 and now returns HTTP 410 for every request.
