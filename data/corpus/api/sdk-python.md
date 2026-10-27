# Python SDK Guide

The official Python SDK wraps the REST API v2 and handles batching, retries and token refresh.

## Install

```
pip install northwind-sdk
```

The SDK supports Python 3.9 and later. The current release is 2.4.

## Create a client

```python
from northwind import Client

client = Client(
    api_key="YOUR_API_KEY",
    region="eu-west",      # us-east (default), eu-west or ap-south
    max_retries=3,
    timeout=30,
)
```

`region` selects the API endpoint. Use `eu-west` for accounts with EU data residency.

## Sending events

```python
client.events.send_batch(events)
```

`send_batch` accepts any number of events and splits them into requests of 500 events, the default `ingest.max_batch_size`. If your account has a higher limit, pass `batch_size=` (up to 5000).

## Retries

The client retries automatically on:
- `NW-4290 RATE_LIMITED`, waiting for the number of seconds in the `Retry-After` header;
- `NW-5031 INGEST_BACKPRESSURE` and `NW-5040 UPSTREAM_UNAVAILABLE`, with exponential backoff starting at 1 second.

It never retries `NW-4001 INVALID_PAYLOAD` or `NW-4012 BATCH_TOO_LARGE`, because resending the same request cannot succeed. `max_retries` controls the number of attempts (default 3).

## Token refresh

When the client is created with `use_tokens=True`, it exchanges the API key for a bearer token and refreshes it automatically when the API returns `NW-4031 TOKEN_EXPIRED`. Automatic refresh was added in version 2.3. Older versions raise `TokenExpiredError` and long-running jobs must refresh manually.

## Querying

```python
rows = client.query("SELECT event, count() FROM events GROUP BY event")
```

For queries that may take longer than 30 seconds, use `client.query_async(...)`, which polls the asynchronous query endpoint.
