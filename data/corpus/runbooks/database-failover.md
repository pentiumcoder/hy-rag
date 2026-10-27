# Runbook: Postgres Failover (pg-main)

**Cluster:** `pg-main` · **Owning team:** Platform Infrastructure · **Alert:** `PgPrimaryUnhealthy`

## Overview

`pg-main` stores accounts, API keys and billing records. The primary runs in us-east with a synchronous replica in a second availability zone and an asynchronous replica in eu-west. `pg_auto_failover` monitors the primary and promotes the synchronous replica automatically after the primary has been unhealthy for 30 seconds.

Recovery objectives for `pg-main`:
- **RTO** (time to restore service): 15 minutes
- **RPO** (maximum data loss): under 1 minute

## Automatic failover

In most cases no action is needed. When `PgPrimaryUnhealthy` fires:

1. Confirm in the `#pg-alerts` channel that `pg_auto_failover` promoted the replica.
2. Check that `keystone` and `ledger` reconnected. Both use the cluster's virtual IP, but long-lived connections may need a restart: `nwctl restart --service keystone`.
3. Confirm the old primary was fenced and is not accepting writes.

## Manual failover

Use manual failover only if automatic failover did not happen within 2 minutes, or for planned maintenance.

```
nwctl db failover --cluster pg-main --confirm
```

The command asks for a change ticket number. During an incident, use the incident ID instead.

Never promote the asynchronous eu-west replica unless the whole us-east region is unavailable. It can be several seconds behind, and promoting it breaks the RPO.

## While the database is failing over

- `ledger` pauses invoice generation jobs automatically and resumes them once the new primary is healthy. Do not re-run invoice jobs by hand.
- Login and API key validation in `keystone` may fail for up to 60 seconds. This is expected.

## After failover

Rebuild a new synchronous replica from the latest backup within 24 hours so the cluster is protected again. Until then, the cluster is running without a synchronous standby and the incident stays open.
