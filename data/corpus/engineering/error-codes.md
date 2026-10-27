# API Error Code Catalogue

Every error response from the Northwind API has a JSON body with `code`, `message` and `request_id`. Error codes have the form `NW-XXXX`; the first digit mirrors the HTTP status class. Include the `request_id` in any support ticket.

## Client errors (4xx)

### NW-4001 INVALID_PAYLOAD
HTTP 400. The request body is not valid JSON or does not match the event schema. The `message` names the first invalid field. Do not retry without fixing the payload.

### NW-4012 BATCH_TOO_LARGE
HTTP 413. The batch sent to `POST /v2/events` contains more events than `ingest.max_batch_size` allows (500 by default, never more than 5000). Split the batch and resend. The official SDKs split batches automatically, so this error usually means a custom client.

### NW-4031 TOKEN_EXPIRED
HTTP 401. The bearer token has expired. Obtain a new one with `POST /v2/auth/refresh` using the refresh token. Python SDK versions 2.3 and later refresh automatically.

### NW-4033 INSUFFICIENT_SCOPE
HTTP 403. The API key is valid but lacks the scope needed for this endpoint, for example a `read`-only key calling `POST /v2/events`.

### NW-4290 RATE_LIMITED
HTTP 429. The API key exceeded its rate limit (`gateway.rate_limit.rps`, which depends on the plan). The response includes a `Retry-After` header in seconds; clients should wait at least that long. The SDKs retry automatically.

## Server errors (5xx)

### NW-5002 QUERY_TIMEOUT
HTTP 504. The query ran longer than `queryd.timeout_ms` (30 seconds by default). Narrow the time range, add filters or use the asynchronous query endpoint.

### NW-5031 INGEST_BACKPRESSURE
HTTP 503. Ingestion is falling behind and the gateway is shedding load to protect the pipeline. Events already accepted are safe. Clients should retry with exponential backoff. Operators: follow the Ingest Backpressure runbook.

### NW-5040 UPSTREAM_UNAVAILABLE
HTTP 502. An internal dependency of the gateway did not respond. Usually transient; retry with backoff. If it persists for more than 5 minutes, check the status page.
