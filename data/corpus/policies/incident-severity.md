# Incident Severity Levels

Severity is set by the first responder and can be raised or lowered by the incident commander as more is learned. When unsure between two levels, choose the higher one.

## SEV1: Critical

Any of the following:
- more than 25% of customers cannot use a core feature (ingest, query or login);
- any confirmed loss of customer data;
- any confirmed security breach or exposure of customer data.

Response: an incident commander is assigned, a dedicated incident channel is opened, and the public status page is updated within 15 minutes of declaration and then at least every 30 minutes.

## SEV2: Major

Any of the following:
- a core feature is degraded for more than 5% of customers;
- a single Enterprise customer is completely unable to use the product;
- a tier-1 service is running without redundancy (for example, a database without a standby).

Response: the owning team's on-call leads the incident; the status page is updated if customers are visibly affected.

## SEV3: Minor

A non-core feature is broken or a core feature is degraded for a small number of customers, with a workaround available. Handled during business hours.

## SEV4: Cosmetic

Visual glitches, typos or minor inconsistencies with no functional impact. Tracked as normal tickets.

## Postmortems

A written postmortem is required for every SEV1 and SEV2 incident. The draft must be shared within 5 business days of resolution and reviewed at the weekly reliability meeting. Postmortems are blameless: they describe what the system allowed to happen, not who made a mistake. Each action item needs an owner and a due date.
