# On-Call Policy

Every team that owns a production service runs an on-call rotation. This policy sets the minimum standard; teams may be stricter.

## Rotation

- Rotations are weekly, with a **primary** and a **secondary** engineer.
- Handoff happens every Monday at 10:00 UTC. The outgoing primary writes a handoff note in the team channel listing open incidents and noisy alerts.
- New engineers join the rotation after six weeks, once they have shadowed one full on-call week.
- Pages are delivered through Beacon. Each on-call engineer must have the Beacon mobile app installed with notifications allowed.

## Acknowledgement targets

| Severity | Acknowledge within |
|---|---|
| SEV1 | 5 minutes |
| SEV2 | 15 minutes |
| SEV3 | next business day |

## Escalation

If the primary does not acknowledge a page within 5 minutes, Beacon automatically pages the secondary. If neither has acknowledged after a further 10 minutes, Beacon pages the team's engineering manager. For SEV1 incidents an incident commander (IC) is assigned from the IC rotation in addition to the team's on-call.

## Swaps and time off

Swaps must be recorded in Beacon before they take effect; a swap agreed only in chat does not count. Engineers who plan time off during their on-call week must arrange a swap first.

## Compensation

On-call engineers receive a stipend of $250 per on-call week. An engineer who is paged more than three times between 22:00 and 07:00 local time during their week also gets one day off in lieu, to be taken within the following month.

## Health of the rotation

Teams review their alerts monthly. Any alert that paged more than five times in a month without requiring action must be tuned or removed.
