# tfvn dataset viewer

Local-only web UI over the bilingual KB, the SFT datasets, and the pipeline
run history. FastAPI backend in `src/tfvn/webapp/`, vanilla-JS SPA in
`webapp/static/`.

## Install

```bash
.venv/bin/pip install -r requirements-web.txt
```

(Into the existing `.venv`. Dependencies: `fastapi`, `uvicorn`.)

## Run

```bash
python3 scripts/serve_webapp.py
```

Serves at http://127.0.0.1:8000. If the venv is not activated:

```bash
.venv/bin/python scripts/serve_webapp.py
```

`--port` is overridable:

```bash
.venv/bin/python scripts/serve_webapp.py --port 8097
```

OpenAPI docs are at `/docs`.

## Views

| Route | View |
|---|---|
| `#/dashboard` | Dashboard: catalog overview, hash integrity, recent runs |
| `#/cards` | Cards: KB explorer with upright/reversed compare |
| `#/dataset` | Dataset: SFT row explorer with filters and paging |
| `#/raw` | Raw: browse any dataset's raw JSONL rows, build filtered exports |
| `#/stats` | Stats: distributions, per-card matrix, IFD histogram, splits |
| `#/reports` | Reports: curated renderings of the six report JSONs |
| `#/runs` | Runs: whitelisted pipeline re-runs with live logs |

## Whitelisted runs

The Runs view can execute exactly the entries below, and nothing else. argv
is fixed server-side in `src/tfvn/webapp/runs_whitelist.py`; there is no shell
and no user-injected command string. Tier semantics: **safe** starts on one
click; **slow** requires a confirm checkbox; **billed** requires the checkbox
plus a typed cost acknowledgement (`I understand this costs money`).

| Tier | id | argv |
|---|---|---|
| safe | `build_wave1` | `scripts/build_wave1.py` |
| safe | `w21` | `scripts/build_wave2_api.py --only w21 --dry-run` |
| safe | `w23` | `scripts/build_wave2_api.py --only w23 --dry-run` |
| safe | `w35` | `scripts/build_wave3.py w35` |
| safe | `w36` | `scripts/build_wave3.py w36` |
| slow | `w33` | `scripts/build_wave3.py w33` |
| slow | `w34-skip-l4` | `scripts/build_wave3.py w34 --skip-l4 --ifd-score-map datasets/raw/ifd_scores.jsonl` |
| slow | `base_diversity` | `scripts/base_diversity_baseline.py` |
| billed | `w22` | `scripts/build_wave2_api.py --only w22` |
| billed | `w32` | `scripts/build_wave3.py w32 --limit <required>` |
| billed | `w34-full` | `scripts/build_wave3.py w34 --ifd-score-map datasets/raw/ifd_scores.jsonl` |

Notes:

- `w21` and `w23` carry `--dry-run` on purpose: without it
  `build_wave2_api.py` creates an `LLMClient()` and hits the live connection
  check. `w22` is the billed reversed-synthesis entry and deliberately has no
  `--dry-run`.
- `w32 --limit` is REQUIRED (1..500). The script default of 40 is a smoke
  size, so the webapp refuses a bare `w32`.
- The alias `kb_rebuild` resolves to `build_wave1`.

## Security

- Loopback only. `serve_webapp.py` defaults to `127.0.0.1` and refuses any
  non-loopback `--host` (exit 2). Do not proxy it outward.
- No authentication. Anyone who can reach the port can start billed runs and
  read the datasets. Keep it on your own machine.
- Whitelist-only execution. argv comes from `runs_whitelist.py`, never from
  user input; options are validated against per-entry specs (`--limit` int
  1..500, etc.).
- `.env` keys are never logged. Run output is scrubbed for `sk-…`, `Bearer …`,
  and the exact values of every `*_KEY` / `*_TOKEN` environment variable
  before it is stored or shown.
- Never start the app with `--reload` while a run is active: reload watches
  the tree and restarts on kb/datasets/logs writes, orphaning run
  subprocesses. The entry point runs `workers=1` with no reload.

## Hash integrity

The hash panel (`/api/hashcheck`) recomputes the canonical-serialisation
digests exactly as the build scripts do: `kb/cards.jsonl` against
`kb/CARDS_HASH.txt` (156 rows), and `filtered_core` + `filtered_bulk` against
`datasets/DATASET_HASH.txt` (13,571 rows). A red banner means genuine drift:
something changed a tracked artifact out from under the hashes.

The app itself never writes `kb/` or `datasets/`. Only a user-initiated
whitelisted re-run rewrites those files, and every such run recomputes its own
hash afterward, so the banner clears once the run finishes.

## Run history

Completed runs are appended to `logs/webapp_runs.jsonl`, with per-run logs
under `logs/webapp_runs/` (`<run_id>.log`). Both paths are gitignored; they
are runtime state, not tracked artifacts. A persisted `running` row is
reconciled as orphaned on boot (a re-spawned server cannot manage a
subprocess it did not start), and the Runs view asks you to acknowledge it
before anything else can run.
