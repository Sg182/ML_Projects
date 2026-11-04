# Progress Log — ERCOT Load Forecasting

Running log of *how* this project got where it is: what was tried, what each
step decided, and what is still open.

`README.md` answers "what is this and how do I run it". This file answers
"why does it look like this, and what was I doing last?"

**Last updated:** 2026-11-04 · **Status:** working end-to-end; one open decision (see [Open questions](#open-questions))

---

## Where things stand

| | |
|---|---|
| Notebooks | 01–06 complete |
| `src/` modules | 10 files — data, features, train, predict, api, day-ahead pipeline, MLP comparison, ERCOT auth/live |
| Tests | **24 passing** (`python -m pytest -q`, ~1.6s, no credentials needed) |
| CI | GitHub Actions on every push — `.github/workflows/tests.yml` |
| Docker | `Dockerfile` present, service containerized |
| Live API | `POST /forecast/day-ahead` works against the real ERCOT API |
| Models | 2 artifacts in `models/` — only the day-ahead one is served |

---

## How it got here

Eight phases. Each one existed because the previous one failed in a specific way.
Dates are the first-commit dates from git history, not from memory.

### 1. Get the data honest (notebook 01) — Sep 12

ERCOT's Native Load archive, 2022–2025, one Excel file per year. Most of the
work was timestamps, not modelling:

- hours run to `24:00`, which pandas will not parse
- spring DST drops an hour; fall DST repeats 02:00 with a `DST` marker
- one 2022 cell Excel had silently stored as a date instead of text

**Why it matters:** every lag feature downstream is only correct if the time
index is correct. Getting this wrong would have been invisible and fatal.

### 2. Baselines (notebook 02) — Sep 13

Naive, linear, and tree models with chronological splits (never random — that
leaks the future). `TimeSeriesSplit` + `GridSearchCV` for tuning.

**Result:** decent on average, *bad* during abrupt winter demand spikes.
Error analysis pointed at one culprit: the model had no idea about weather.

### 3. Add weather (notebook 03) — Sep 15

Hourly observations for 4 Texas metros from Open-Meteo, timezone-aligned and
merged onto the load series.

### 4. Does weather actually explain the failure? (notebook 04) — Sep 15

An **oracle experiment**: feed the model the weather that *actually happened*.
This is deliberately cheating — it is a diagnostic, not a candidate model.

**Result:** large error reductions on the two days holding the worst baseline
errors, much smaller ones over the full test year. Confirmed the diagnosis. The open
question became whether *forecast* weather (which you really have at prediction
time) would retain that gain.

Figures and their caveats live in the README's Findings section — those two days were
picked *because* they were the worst, so the headline number flatters the result.

### 5. Replace oracle with forecasts (notebook 05) — Sep 19

Archived 24-hour-ahead GFS forecast temperatures, leakage-free.

**Result:** forecast weather performed about as well as realized weather — close
enough over the full test year that the "share of the oracle gain retained" ratio goes
above 100% and stops carrying meaning. So the gain was real and obtainable, not an
artifact of cheating. This became the production-honest model. Figures in the README.

### 6. The protocol leak (notebook 06) — Sep 20

Connecting to ERCOT's live API exposed a problem that offline backtesting had
hidden: a model issued on a **fixed daily schedule** cannot use a 24-hour load
lag or day-1 forecasts — *at issuance time those values do not exist yet*.

The backtest had been quietly handing the model data that would not be
available in production.

**Fix:** a stricter **10 AM issuance protocol** — 48-hour lags and
previous-day2 GFS temperatures.

| Model | Test MAE |
|---|---|
| Rolling model (optimistic, leaks protocol) | ~2211 MW |
| **10 AM day-ahead protocol** | **~3073 MW** |
| Protocol-valid naive baseline | ~3630 MW |

The day-ahead number looks worse than the rolling model. It is not worse — it
is the first number that is *true*. Against the baseline it can actually be
compared to, it wins clearly.

> This is the single most interview-worthy moment in the project: the metric got
> worse and that was the correct outcome.

### 7. Engineering — Sep 16 → Oct 31

- `src/` modules reproduce the selected notebook models **bit-exactly**
  (this was the acceptance test for the refactor, not "close enough")
- Live ERCOT auth + data fetch (`ercot_auth.py`, `ercot_live.py`)
- FastAPI service, lazy model loading so the app starts without an artifact
- Containerized (Oct 11)
- GitHub Actions CI on a clean machine (Oct 31) — catches anything that only
  worked because of local laptop setup

### 8. Was it ever the algorithm? (`src/train_mlp_dayahead.py`) — Nov 4

Every model so far has been a tree or a linear fit. The obvious unanswered question:
would a neural network have done better all along?

A multi-layer perceptron, same nine features, same split, same protocol, tuned on
development data only. Two scalers are needed that the tree models never required —
one on the inputs (hour is 0–23, load is ~30,000–80,000 MW) and one on the target
(a default output layer cannot reach tens of thousands without a long warm-up).

| Model | Test MAE | Test RMSE | R² |
|---|---|---|---|
| Gradient boosting | 3,073 MW | 3,692 MW | 0.867 |
| **MLP** | **3,092 MW** | **3,680 MW** | **0.868** |

**Result:** a tie — 0.6% behind on MAE, slightly ahead on RMSE. Which is the answer to
the question. Notebook 02 saw six model classes cluster within ~8%; the MLP joins the
same cluster. Features and protocol set the ceiling here, not the function class, and
no amount of model shopping was going to move it.

---

## Open questions

### Which model gets served? *(needs a decision)*

`models/` holds two artifacts, and `src/api.py` currently hardcodes one:

```python
MODEL_PATH = "models/ercot_gb_dayahead.joblib"
```

| Artifact | Protocol | MAE | Served? |
|---|---|---|---|
| `ercot_gb_dayahead.joblib` | 10 AM, honest | ~3073 | yes |
| `ercot_gb_model.joblib` | rolling, optimistic | ~2211 | no |

The day-ahead model is the defensible default. The rolling model is only
meaningful if something can genuinely supply fresh 24h lags at request time —
worth exposing as a separate `/predict` route if so, **not** as a swap.

---

## Known limitations

**Lag vintage mismatch.** The live pipeline pulls load lags from ERCOT's
real-time "actual system load" API, but the models were trained on the
settlement "native load" archive. The real-time feed reads about **2% higher**
(+925 MW mean in a Nov 2025 sample).

Timestamp and DST semantics were verified identical (`scripts/verify_ercot_dst.py`) —
so this is not an alignment bug, it is a different data vintage. Likely fix is
retraining on API-sourced lags. Documented in `src/train_dayahead.py`.

---

## Next up

- [ ] Decide the served-model question above
- [ ] Retrain on API-sourced lags to close the vintage gap
- [ ] Monitoring — the last unchecked item from the README roadmap
- [ ] Optional: `/predict` route for the rolling model
- [ ] Later ideas: apparent-temperature features, token caching for ERCOT auth

### Housekeeping

- `src/practice.py` and `test_module.py` are untracked Python-import scratch
  files, unrelated to ERCOT — delete or move out of the project
