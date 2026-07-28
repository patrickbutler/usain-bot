# usain-bot

Usain Bot — a local-first, agentic running coach. You invoke it manually
before a run (CLI) or leave the local web UI running (`usain-bot serve`);
it reads your real Garmin activity history, compares what you actually
ran against what the plan said, reasons about the delta, and tells you
the optimal distance for today plus a rolling view of the rest of the
plan.

Injury prevention outranks plan adherence, always. See the design spec
in the original build prompt for the full rationale; this README covers
setup and day-to-day usage.

## How it's put together

```
start.sh                    one-command startup (venv, deps, first run, launch UI)
config.yaml                 athlete/goals/guardrail/sync config (edit freely)
src/usain_bot/
  models.py                 shared dataclasses/enums, no I/O
  config.py                 loads config.yaml + .env
  ingest.py                 data-quality gate: validation, repair, dedupe (auto, every write)
  sessions.py               split-run merging (<=3h gap = one run)
  classification.py         merged sessions -> classified runs -> load anchors
  guardrails.py             every §5 formula as a pure, unit-tested function
  planner.py                goal-driven plan generation from actuals + overrides
  validation.py             deterministic referee for the milestone/build rules
  garmin_adapter/           all Garmin I/O isolated here (live + mock)
  storage/                  StorageBackend ABC + LocalBackend (SQLite+FS) + GCPBackend (stub)
  agent.py                  decision procedure, sync/backfill/dedupe
  service.py                CoachService: today-cache + read helpers shared by web + chat
  projection.py             shared "next 7 days" rolling view (CLI + web)
  cli.py                    `usain-bot` entrypoint
  chat/
    providers/               LLMProvider interface + OpenAIProvider (default) + GeminiProvider + AnthropicProvider
    tools.py                 tool definitions the LLM calls — dispatches into agent/planner
    session.py                provider-agnostic tool-calling loop
  web/
    app.py                   FastAPI REST endpoints
    static/                  vanilla HTML/CSS/JS frontend (Chat / Upcoming / History tabs)
tests/                      pytest suite, network-free via MockGarminAdapter + a scripted fake LLM
references/                 example coaching reference articles you can import
```

No agent/planner/guardrail code imports a concrete storage backend, the
`garminconnect` library, or a concrete LLM SDK directly — everything
goes through `StorageBackend`, `GarminAdapter`, and `LLMProvider`.
That's what makes the whole system testable offline and lets storage
move to GCP, or the chat model move to a different vendor, without
touching reasoning code.

## Quick start (one command)

```bash
./start.sh              # real Garmin data (prompts you to fill in .env on first run)
./start.sh --demo       # bundled sample data — no Garmin account or API key needed
```

`start.sh` creates the virtualenv, installs dependencies, generates your
plan on first run, starts the server, and opens the UI in your browser.
Safe to re-run. Everything below is the manual equivalent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"       # CLI only
pip install -e ".[dev,chat]"  # + the web UI's Chat tab (adds the openai, google-genai, and anthropic SDKs)
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

## Web UI

```bash
usain-bot serve
```

Opens a local web server (default `http://127.0.0.1:8420`) with three tabs, built entirely on the same `agent`/`planner`/`storage` functions as the CLI — it's a view onto the same system, not a second implementation:

- **Chat** — your coach, who speaks with a Jamaican accent and plenty of slang (style only — every number, date, and safety warning still comes from the tools, stated plainly). Ask about today's run, the plan, or your history; apply overrides ("make next week easier," "push my half marathon back three weeks"); mention symptoms ("my hip's been sore") or how a run felt ("legs were dead out there") and it records both, which feeds the recommendation math directly. Replies render as Markdown, so tables, lists and week-by-week breakdowns are readable rather than a wall of text. The composer is a textarea that grows with your message: **Enter inserts a newline, ⌘/Ctrl+Enter (or the Send button) sends** — so a long, multi-paragraph description of a niggle doesn't fire off half-written.
- **Upcoming** — today's recommendation, the binding constraint, a rolling 7-day view, the full plan week by week, one-click hip/back/fatigue flags, a refresh button, and a **Plan history** section: every plan version ever saved, newest first, with what triggered it and why (`usain-bot run` reprojection, a gap protocol, a conversational override, an approved plan revision), expandable to see exactly which weeks changed and how. Any plan change made in Chat re-renders this tab immediately.
- **History** — a weekly volume chart, a **How your runs felt** section pairing every run with its 1–5 subjective score (score unrated runs inline; already-scored runs show the score and are never asked about again), and a table of past runs (classified long/easy/quality/recovery/cross-training) pulled straight from Garmin, with a manual sync button.

### Jamaican mode (hidden settings page)

`http://127.0.0.1:8420/settings` — deliberately not linked from the
navigation. One toggle turns the Jamaican voice off in favour of neutral,
professional English for demos. **Coaching is identical either way**: same
guardrails, same tools, same numbers, only the wording changes
(`chat/session.py` swaps `JAMAICAN_VOICE` for `PROFESSIONAL_VOICE` inside
an otherwise identical system prompt). The page also shows which LLM
provider and model are currently configured.

Chat defaults to **OpenAI (ChatGPT)** and needs `OPENAI_API_KEY` (see `.env.example`) — every other tab works without it. **The LLM never computes a mileage, date, or guardrail value itself.** It only ever selects a tool (`chat/tools.py`) and the deterministic functions in `agent.py`/`planner.py`/`guardrails.py` do the actual math, same as every other entry point. If a tool can't answer something, the system prompt tells it to say so rather than estimate.

**Swapping the LLM provider:** `chat/providers/base.py` defines a small provider-agnostic interface (`LLMProvider`, plus normalized `ChatMessage`/`ToolSpec`/`ProviderResponse` types) that `chat/session.py`'s orchestration loop and `chat/tools.py`'s tool definitions depend on — never on a concrete vendor SDK. Three providers ship today, each with a meaningfully different wire format underneath the same interface (OpenAI keys tool results on an opaque call id via `role: "tool"` messages; Anthropic nests them as content blocks in a `role: "user"` message; Gemini keys them on the function *name* via `role: "model"`/`"user"` content parts) — proof the abstraction holds in practice, not just on paper:

| Provider | `chat.provider` value | Model default | Required env var |
|---|---|---|---|
| OpenAI (default) | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| Gemini | `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| Anthropic | `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |

None of the three are required unless you select them — `usain-bot serve` runs fine with zero LLM API keys set; only the Chat tab needs one, for whichever provider is configured.

Switch by setting in `config.yaml` (`provider` and `model` are paired — always set both together):

```yaml
chat:
  provider: gemini
  model: gemini-2.0-flash
```

or via environment variables (`USAIN_BOT_CHAT_PROVIDER=gemini`, `USAIN_BOT_CHAT_MODEL=gemini-2.0-flash`) without touching the file — see the alternate blocks in `.env.example`. To add a fourth vendor: write `chat/providers/<vendor>.py` implementing `LLMProvider`, add one branch to `chat/providers/factory.py` (and one line to its `REQUIRED_ENV_VAR` map so `usain-bot serve`'s startup warning knows which key to check for). Nothing in `chat/session.py`, `chat/tools.py`, or `web/app.py` changes — and `tests/test_chat_providers.py` / `tests/test_chat_providers_gemini.py` exercise every shipped provider's request/response translation directly (mocked SDK calls, no network — none of this has been tested against a live API key in this environment).

## Garmin data pipeline

**Scope:** only running activities are stored — `activityType.typeKey` in
`running`, `trail_running`, `treadmill_running` (`treadmill` also
accepted). Cycling, strength, and everything else are filtered out at the
adapter boundary.

```bash
usain-bot backfill      # one-time full-history import
usain-bot sync          # incremental pull (also runs automatically on `run`/`serve`)
```

Those are the only two data commands, deliberately. **Validation and
deduplication are properties of ingestion, not chores** — every write
path runs them automatically (`ingest.py`), so there is no
`usain-bot dedupe` or `usain-bot validate-data` to remember. If you ever
found yourself needing one, something upstream would be broken.

- **Backfill** walks history backwards in bounded chunks (90 days by
  default) with a pause between each, because a single wide request is
  what returns HTTP 429. The adapter additionally retries on 429 with a
  long backoff schedule (30s → 60s → 120s → 240s) versus a short one for
  ordinary transient failures. If it still gets cut off it says so and is
  safe to re-run — activities upsert by ID, so resuming never duplicates.
- **Every sync re-pulls a trailing overlap window** (`sync.overlap_days`,
  30 by default) rather than only "everything since last sync". That's
  what catches activities you *edited* on Garmin after they were first
  imported: `save_activities` upserts, so a corrected distance, duration,
  name, or HR updates the stored row instead of being ignored as a
  duplicate. Edits older than the window aren't detected — widen
  `sync.overlap_days` if you routinely fix up old activities.
- **Bad data is stopped at the door.** Unusable activities (negative or
  zero distance, non-positive duration, GPS teleports implying 900 mph, a
  300-mile "run", future dates) are rejected and logged rather than
  silently admitted — one of those would wreck every load anchor and the
  whole plan built on them. Individually implausible *fields* are instead
  repaired and the run kept: a 400 bpm HR reading or a max_hr below
  avg_hr gets nulled, pace is recomputed from distance and duration.
  Losing a real run over one glitched sensor would corrupt mileage worse
  than a missing HR value. Thresholds are loose on purpose — a 4:30/mi
  track rep and a 25 min/mi hike-jog both pass.
- **Duplicates are removed automatically** after every write: the same
  physical run stored under two Garmin IDs (near-identical start time and
  distance) collapses to the richest record.
- **Split runs are merged at read time**, and are a *different* thing
  from duplicates. If one run is recorded as several activities (watch
  stopped/restarted), any gap of **≤ 3 hours** between one ending and the
  next beginning makes them one session; more than 3 hours means
  genuinely separate runs. Stored rows stay raw and immutable — merging
  happens in `sessions.py` on the way into classification, so the rule is
  tunable (`sync.merge_gap_hours`) and re-applies to all history with no
  migration. The tight start-time tolerance on duplicate detection is
  what keeps deliberate split recordings from being deleted as dupes.

Counts for all of the above are reported inline in the sync/backfill
output and in `POST /api/sync` and `POST /api/backfill` responses.

## Training rules and plan validation

The plan is generated **from your actual data** — the long-run
progression starts at your demonstrated long-run anchor and weekly
volume at your real chronic load. It never schedules a long run below
what you've already proven you can run.

Milestone rules, enforced deterministically:

| Milestone | Date | Prerequisite | Taper |
|---|---|---|---|
| Half marathon | flexible | a 12 mi run before the taper | 1–2 weeks |
| Marathon | **fixed** (`goals.marathon.date`) | a 20 mi run before the taper | 2–3 weeks |
| 50K | flexible | ≥30 mi/week for 3 consecutive weeks | 2–3 weeks |

`validation.py` is a pure, independently-tested referee: the planner aims
to satisfy the rules, the validator says whether it actually did. It also
checks gradual build limits (long run within min(1 mi, 10%), volume
within +10%/week, ignoring expected back-off discontinuities) and
back-off cadence.

Like data validation, this runs **automatically** rather than on demand —
results appear on the Upcoming tab every time you load it, inline in
`GET /api/plan`, and via `GET /api/plan/validation`. You can also just
ask the coach ("does this plan actually get me ready?") and it calls the
same validator.

**Flexible milestones can be pushed** ("move my half marathon back three
weeks") via chat or `POST /api/milestone/push`. The delay persists as a
preference — surviving the plan being regenerated from anchors every
invocation — and the planner fills the extra weeks with normal
build/back-off weeks so your base is maintained rather than going flat.

### Rest after a long run

`guardrails.requires_rest_after_long_run` is a hard rule, not a
suggestion: **the day after a long run is always off.** It enters the
same min-across-ceilings computation as every other guardrail with a
ceiling of 0.0 mi, so on that day the recommendation is rest and the
binding constraint says so. The 7-day projection draws the rest day in
the right place too, even when it lands on a normal run day.

### Pacing: how fast the long run climbs

Two modes, because "when is the goal?" changes the right answer:

| Mode | When | Behaviour |
|---|---|---|
| `milestone_smoothed` (**default**) | a dated goal exists | spreads the ramp across the weeks actually available — if there are 12 miles to add over 20 build weeks it adds ~0.6/wk, not the full 1 mi/wk guardrail maximum followed by a month of idling at peak. Delays crossing 15 mi and lowers cumulative load. |
| `ramp_asap` | no date set (e.g. the 50K) | nothing to smooth toward, so build as fast as the guardrails safely allow and reach capability sooner. |

Two related ceilings run in both modes:

- **Long runs past 15 mi (`HIGH_MILEAGE_LONG_RUN_MI`) are spent
  strategically, not accumulated.** Smoothed pacing holds below that line
  until the calendar actually requires crossing it.
- **At most `MAX_WEEKS_AT_PEAK_LONG_RUN` (2) weeks at the peak long run.**
  Seven consecutive 22-milers is a durability liability, not a training
  stimulus. Once the allowance is spent the block *maintains* at 85% of
  the peak rather than re-climbing to it — the peak is still reached, just
  not repeated.

Both are athlete-adjustable through a plan revision ("cap me at 18 miles",
"only two weeks at the peak"), as is running frequency, which caps how
much volume a week can absorb (one long run plus easy days at 60% of it)
and removes the quality session below 4 days/week.

### Changing the plan in conversation: draft, review, publish

A plan change is a training decision, so it gets reviewed before it
becomes the thing you train off. Asking for one in chat runs a **genuine
recomputation** — `planner.revise_plan` re-runs the generator under the
new constraints, it does not restaple a rationale to the same numbers —
and holds the result as an in-memory **draft**:

1. `propose_plan_revision` → recomputes, returns a before/after summary,
   a per-week diff and a validation result. **Nothing is saved.**
2. The coach shows you what changed and asks explicitly whether to make
   it official.
3. `publish_draft_plan` (only after you say yes) → saves a new plan
   version with the reasoning and your approval note attached for
   historical reference, and **persists the approved constraints as
   preferences**.

Step 3's last clause is the load-bearing one. The plan is regenerated
from anchors on every invocation, so a change that isn't persisted as a
constraint silently reverts on the next page load — which is exactly how
"I asked chat to smooth the plan and the Upcoming tab didn't move"
happens. `tests/test_pacing_and_drafts.py` asserts the full cycle:
propose (version unchanged) → publish (version + 1) → force a full
regeneration → the approved shape is still there.

`discard_draft_plan` throws a draft away. The same three steps are
available over HTTP as `POST /api/plan/revision/propose`,
`POST /api/plan/revision/publish` and `DELETE /api/plan/revision`.

## Adapting to you

- **Run frequency is derived from actuals**, not from config: distinct
  days with a run in the trailing 28 days, ÷4. If you run more or fewer
  days than `athlete.available_run_days_per_week` says, the plan adapts
  and the plan rationale notes the mismatch.
- **How runs felt is remembered and used.** The coach asks about recent
  unrated runs; you can also rate them in the Upcoming tab or in
  History → *How your runs felt* (1–5). Or just say it — "that one was
  rough" — and the coach maps it to a score and logs it. Rating is
  idempotent per run: a scored run shows its score and is never asked
  about again, and re-scoring corrects the old value rather than
  double-counting. A poor recent average becomes a hard ceiling candidate
  (`recent_run_feeling`) in the same min-across-guardrails computation as
  the load math — this is deterministic Python, not left to the LLM.

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

**Flags are undoable.** In the web UI the flag buttons toggle — clicking
an active flag clears it, and the active flag is shown explicitly on the
recommendation card so a mis-tap is obvious and reversible. You can also
say so in chat ("that was a mistake, I'm fine") or call
`DELETE /api/health-flag`. Clearing recomputes today without the
conservative cap; the flag stays in history, it just stops constraining
today.

## Plan version history

Every plan change — a daily reprojection from `usain-bot run`, a gap-protocol
adjustment, a regeneration after a long layoff, or a conversational
override — writes a brand-new, immutable `PlanVersion` row; nothing is
ever mutated in place. Every version records:

- **trigger** — what caused it (`first_run`, `scheduled_reprojection`, `gap_detected`, `gap_regeneration`, `user_override`)
- **rationale** — the human-readable "why," including any cited reference article
- **diff_from_prior** — a text diff against the immediately preceding version
- a **structured per-week diff** (`planner.diff_plan_weeks`), computed on the fly from the stored weeks — which weeks were added/removed/changed and exactly which fields moved (volume, long run, block, back-off status)

```bash
usain-bot plan --history          # CLI: full lineage with diffs
```

```
GET /api/plan/history             # REST: same data, JSON, newest first
```

The web UI's Upcoming tab renders this as an expandable list; the chat's
`get_plan_history` tool lets you ask "why did my plan change" or "what
happened after I asked to ease up" directly. See the design notes below
for the current limit of this system: overrides show up as their own
version but don't survive the *next* natural reprojection.

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

274+ tests, all network-free — including chat, which is tested at two
levels without ever calling a real LLM API: the orchestration loop
against a scripted fake `LLMProvider` (`tests/test_chat_session.py`),
and each shipped provider's actual request/response translation against
a mocked SDK client (`tests/test_chat_providers.py`,
`tests/test_chat_providers_gemini.py`). `guardrails.py`
functions are pure and unit-tested directly, including edge cases (zero
chronic load, a >6-week gap, a long run exceeding 35% of weekly volume,
ACWR undefined at cold start). `MockGarminAdapter` reads
`tests/fixtures/mock_activities.json` so `classification.py`,
`planner.py`, the full `agent.py` decision procedure, the REST API
(`tests/test_web_api.py`, via FastAPI's `TestClient`), and the chat tool
dispatch (`tests/test_chat_tools.py`) are all exercised end to end
without a Garmin account or an API key.

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
- **Overrides are best-effort against regeneration, not permanent
  edits.** `generate_macro_plan` always regenerates the *entire*
  forward projection fresh from current anchors — that's the design
  (see the planner.py module docstring). A conversational override
  (`ease_upcoming_week`, `shift_marathon_date`, ...) writes a new plan
  version and the web UI's cache is patched in place so it's visible
  immediately without triggering another regeneration in the same
  breath (`CoachService.apply_plan_update`) — but the *next* natural
  regeneration (tomorrow's `usain-bot run`, or the day rolling over in
  the web UI) recomputes from anchors again and won't remember
  yesterday's "make next week easier." There's no persistent
  override-tracking layer yet; if you need a change to stick across
  days, edit `config.yaml` for anything durable (goal dates, baseline),
  or re-apply the override each session.
- **SQLite + a thread pool.** The web server runs each request's
  handler in a worker thread (FastAPI's default for sync `def` routes),
  so `LocalBackend` opens its connection with `check_same_thread=False`
  and serializes every call through a single lock — correct at the
  scale of one local file for one user, not something to scale up
  as-is if this ever became multi-user.
