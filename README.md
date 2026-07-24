# usain-bot

Usain Bot — a local-first, agentic running coach. You invoke it manually
before a run; it reads your real Garmin activity history, compares what
you actually ran against what the plan said, reasons about the delta,
and tells you the optimal distance for today plus a rolling view of the
rest of the plan.

Injury prevention outranks plan adherence, always. See the design spec
in the original build prompt for the full rationale; this README covers
setup and day-to-day usage.

## How it's put together

```
config.yaml                 athlete/goals/guardrail config (edit freely)
src/usain_bot/
  models.py                 shared dataclasses/enums, no I/O
  config.py                 loads config.yaml + .env
  classification.py         raw Garmin activities -> classified runs -> load anchors
  guardrails.py             every §5 formula as a pure, unit-tested function
  planner.py                macro plan generation, versioning, conversational overrides
  garmin_adapter/           all Garmin I/O isolated here (live + mock)
  storage/                  StorageBackend ABC + LocalBackend (SQLite+FS) + GCPBackend (stub)
  agent.py                  orchestrates the §4.3 decision procedure end to end
  cli.py                    `usain-bot` entrypoint
tests/                      pytest suite, network-free via MockGarminAdapter
references/                 example coaching reference articles you can import
```

No agent/planner/guardrail code imports a concrete storage backend or
the `garminconnect` library directly — everything goes through
`StorageBackend` and `GarminAdapter`. That's what makes the whole system
testable offline and lets storage move from local SQLite to GCP later
without touching reasoning code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your Garmin credentials:

```bash
cp .env.example .env
```

```
GARMIN_EMAIL=you@example.com
GARMIN_PASSWORD=your-password
GARMINTOKENS=~/.usain-bot/garmin_tokens
```

Credentials are read from the environment only (via
`GarminCredentials.from_env()`) — never hardcode them, never commit
`.env`. On first login, `garminconnect` (via its `garth` auth layer)
caches a session token at `GARMINTOKENS`, so most subsequent invocations
reuse the cached session instead of re-authenticating — this matters
because Garmin rate-limits and occasionally changes its endpoints.

Review `config.yaml` before your first run: `athlete.available_run_days_per_week`,
`goals[].date`, and `athlete.baseline_long_run_mi` are all editable
without touching code. The baseline long-run figure is only a starting
hypothesis — the first invocation verifies it against actual Garmin
history and says so if they disagree.

## First run

```bash
usain-bot init
```

This pulls the last 12 weeks of Garmin history, classifies it, reports
actual weekly volume / long-run capacity / adherence / any gaps found,
generates the full macro arc (base building → 13.1mi capability →
marathon block → taper → marathon → recovery → 50K block → 50K), and
asks you to confirm before persisting it as plan version 1. Nothing is
written to storage until you confirm (`--yes` skips the prompt for
scripted use).

The 50K block and race are always emitted as explicit **TBD
placeholders** — per the injury-first design, they're never scheduled
with a real date until the marathon is complete and recovery data
confirms readiness.

## Day to day

```bash
usain-bot run
```

Prints today's recommendation (run type, distance, effort guidance, the
binding constraint that produced it, and what would unlock more next
time) plus a rolling forward view (next 7 days, remainder of the plan
week by week, distance to the next milestone). A machine-readable JSON
copy is written alongside it under `<data_dir>/output/`.

Useful flags:

```bash
usain-bot run --dry-run              # show the recommendation, write nothing to storage
usain-bot run --flag hip             # force the conservative branch (hip/back/fatigue)
usain-bot run --date 2026-08-01      # override "today", mainly for testing
```

## Conversational overrides

```bash
usain-bot override "make next week easier"
usain-bot override "shift the marathon block back two weeks"
usain-bot override "I want to move my long run to Sunday"
```

Recognized overrides mutate the plan, write a new plan version with the
diff and rationale, and print any guardrail flags the change raises —
overrides are honored even when risky, but the risk is never silently
absorbed. Unrecognized phrasing is rejected with a clear message rather
than guessed at; edit `config.yaml` directly for anything not covered.
If you just run less than recommended, that's not an override — it's
data, and the next invocation absorbs it into the anchors automatically.

## Reference articles

```bash
usain-bot reference add references/acwr-and-injury-risk.md
usain-bot reference list
usain-bot reference search "return to running after a break"
```

References are chunked and indexed separately from conversation memory.
When the agent updates the plan in response to a gap or a regeneration
trigger, it consults the store and cites whichever reference informed
the change (visible in `usain-bot plan --history`). Two example articles
are included in `references/` to seed the store.

## Health flags

`--flag hip`, `--flag back`, or `--flag fatigue` force the conservative
branch regardless of what the guardrail math would otherwise allow: cap
at the last completed distance, no quality work, cross-training
suggested instead. Flags are persisted so the agent can see the pattern
over time (`storage.get_recent_health_flags()`).

## Storage backend

Default is `local`: a single SQLite file plus a references directory
under `storage.local.data_dir` (`./data` by default; override with
`USAIN_BOT_DATA_DIR`). Four logically separate stores live there:
`activities` (append-only, deduped by Garmin activity ID), `plan_versions`
(immutable lineage — every change is a new version, never mutated in
place), `conversations`, and `references`.

To move to GCP later, implement `storage/gcp.py`'s `GCPBackend` (the
interface and intended BigQuery/GCS mapping are already sketched in its
docstrings) and set:

```
USAIN_BOT_STORAGE_BACKEND=gcp
USAIN_BOT_GCP_PROJECT=...
USAIN_BOT_BQ_DATASET=...
USAIN_BOT_GCS_BUCKET=...
```

No changes are needed anywhere else — `agent.py`, `planner.py`, and
`guardrails.py` only ever depend on the abstract `StorageBackend`.

## Testing

```bash
pytest
```

88+ tests, all network-free. `guardrails.py` functions are pure and
unit-tested directly, including edge cases (zero chronic load, a >6-week
gap, a long run exceeding 35% of weekly volume, ACWR undefined at cold
start). `MockGarminAdapter` reads `tests/fixtures/mock_activities.json`
so `classification.py`, `planner.py`, and the full `agent.py` decision
procedure are exercised end to end without a Garmin account.

## Design notes worth knowing

- **Guardrails vs. the macro plan.** `guardrails.py` implements every
  formula in §5 literally and is what actually binds "today's"
  recommendation in `agent.py`. The multi-month macro projection in
  `planner.py` is a coarse (volume + long-run per week) forecast,
  regenerated fresh from current anchors on every invocation — it's
  never stale, but it also isn't a session-by-session commitment.
  Fine-grained same-week rules (never bump the long run and add a new
  quality session in the same week, never raise volume and intensity
  together) are enforced live for "today," not baked into the forward
  projection.
- **Back-off math is unforgiving on paper.** Three weeks of +10% growth
  compounds to only ~1.33x, which a 70-75% back-off week (§5.5) nearly
  or fully erases — taken as a literal recursive formula over many
  months that nets to roughly flat, not the progressive overload a
  marathon build needs. The planner ratchets each build cycle's floor
  so it never regresses below where the previous cycle started, and
  flags in the marathon week's notes when the projected peak long run
  falls short of the 20-22mi target under current constraints (running
  days available, the 35%-of-volume cap) — rather than silently
  producing a plan that can't get there. Real chronic load re-derived
  from actual training on each future invocation will likely close part
  of that gap; the note says explicitly what would close the rest.
