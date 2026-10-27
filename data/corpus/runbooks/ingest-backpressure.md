# Runbook: Ingest Backpressure

**Service:** `ingestd` · **Owning team:** Data Platform · **Alert:** `IngestConsumerLagHigh`

## When this runbook applies

The `IngestConsumerLagHigh` alert fires when the metric `ingest_consumer_lag` stays above 50,000 messages for 10 minutes in any region. Customers may start receiving `NW-5031 INGEST_BACKPRESSURE` errors once lag passes 500,000 messages, because the gateway begins shedding load at that point.

Events already written to Kafka are not lost: `events.raw` retains data for 72 hours. Being slow is acceptable; dropping data is not.

## Diagnose

1. Open the "Ingest Overview" Grafana dashboard and select the affected region.
2. Check whether lag is growing on all partitions or only a few. A few hot partitions usually means a single very large tenant; all partitions means a capacity or downstream problem.
3. Check ClickHouse insert latency on the same dashboard. If p99 insert latency is above 2 seconds, the bottleneck is ClickHouse, not `ingestd`.
4. Look for `ingestd` pod restarts: `nwctl logs --service ingestd --since 30m | grep OOMKilled`.

## Mitigate

- **Not enough consumers:** scale out with `nwctl scale ingestd --replicas <n>`. The default is 6 replicas and the maximum is 24 (one per partition). Double the count first, then reassess after 10 minutes.
- **ClickHouse is slow:** check for a merge backlog on `ch-events` and page the Data Platform secondary if merges are stuck. Scaling `ingestd` will not help and adds pressure.
- **Pods are being OOMKilled:** check whether `ingest.max_batch_size` was recently changed. Never raise it above 5000; that change caused the March 2026 outage.

## Escalate

If lag is still growing 30 minutes after mitigation, or customers are receiving `NW-5031`, declare a SEV2 and post on the status page. If more than 25% of customers are affected, it is a SEV1.

## After the incident

Confirm lag has returned below 10,000 messages, scale `ingestd` back to its default replica count, and record the timeline in the incident channel for the postmortem.
