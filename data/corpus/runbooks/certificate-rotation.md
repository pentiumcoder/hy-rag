# Runbook: TLS Certificate Rotation

**Owning team:** Edge · **Alert:** `CertExpiringSoon`

## How certificates are managed

All certificates are issued and renewed by cert-manager running in each Kubernetes cluster.

- Public endpoints such as `api.northwind.example` use certificates from Let's Encrypt.
- Service-to-service traffic uses mutual TLS with certificates from the internal CA, `nw-internal-ca`.

cert-manager renews every certificate automatically 30 days before it expires. The `CertExpiringSoon` alert fires when any certificate is within 14 days of expiry, which means automatic renewal has been failing for over two weeks.

## When CertExpiringSoon fires

1. Find the failing certificate: `kubectl get certificates -A | grep -v True`.
2. Describe it and read the latest events: `kubectl describe certificate <name> -n <namespace>`.
3. Common causes:
   - Let's Encrypt rate limits after many failed attempts. Wait for the limit to reset (up to 1 hour) before retrying.
   - DNS challenge failing because a record was changed by hand.
   - The internal CA issuer is unhealthy.
4. Force a renewal once the cause is fixed: `kubectl cert-manager renew <name> -n <namespace>`.

## Internal CA rotation

The root certificate of `nw-internal-ca` is rotated every two years. The next rotation is scheduled for February 2027. Root rotation is a planned change led by the Edge team with a 4-week overlap period, during which both the old and new roots are trusted.

## Emergency

If a public certificate has already expired, customers see TLS errors on every request. That is a SEV1: page the Edge on-call immediately and declare the incident before starting the steps above.
