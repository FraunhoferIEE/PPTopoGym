---
name: pptopo-perf-loop
description: Run the full autonomous profile-optimize-verify cycle on PPTopoGym until measured gains plateau. Use when asked to iteratively or autonomously improve the environment's performance, to keep optimizing until it stops paying off, or to drive repeated rounds of profiling and optimization.
---

# The PPTopoGym performance loop

Drives `pptopo-profile` and `pptopo-optimize` in a cycle, autonomously, until further
work stops paying. This skill owns the **stopping condition** and the **ledger** — the
other two own measurement and change.

## Invariants

- **Never land an unverified change.** The gate is mypy → ruff → pytest → `ab_compare`.
- **One optimization per commit.** Bisectable, revertable, attributable.
- **The ledger is append-only.** Negative results are as valuable as wins; they stop the
  next cycle from re-investigating a dead end.
- **Do not touch UCTE data.** Out of scope for this project — never read or import it.
- Scope is everything used to build and run a PPTopoGym environment: config
  construction, action-space generation, substations, profiles, the env itself,
  observations, N-1, and the benchmark agents.

## Establish the baseline once

Before cycle 1:

```bash
git status --porcelain          # record dirty state
poetry run pytest -q 2>&1 | tail -5
poetry run python scripts/bench_pptopo.py --suite build --suite step --suite obs \
    --json profiling/baseline.json --label "cycle-0 baseline"
```

A pre-existing test failure must be recorded as pre-existing *now*, or a later cycle
will misattribute it to its own change. The suite is slow — run it in the background
and continue reading code while it runs.

## Each cycle

```
1. Profile        -> invoke pptopo-profile, ranked hotspot list   -> verify: findings file written
2. Pick ONE       -> highest absolute ms, plausible fix           -> verify: hypothesis stated in ledger
3. Optimize       -> invoke pptopo-optimize                       -> verify: gate green + ab_compare win
4. Commit or revert                                               -> verify: tree clean, ledger updated
5. Re-profile     -> the fix moves the bottleneck                 -> verify: new ranked list
```

Step 5 is not optional. Optimizing the same stale profile twice is how effort gets spent
on what is no longer the bottleneck.

### Picking the target

Rank by **absolute milliseconds saved**, not percentage. Then discount by risk:

| signal | action |
|---|---|
| pure setup code, result cacheable | take it — highest value/risk ratio |
| env-owned per-step work | take it if the profile shows real ms |
| cost inside pandapower marshalling | only "call it less often" fixes work |
| anything needing `recycle` or ls2g batch APIs | skip — settled dead ends |

## The ledger

Maintain `profiling/PERF_LEDGER.md`. One row per attempt, wins and failures alike:

```markdown
| cycle | target (file:func) | hypothesis | result | measured | landed | commit |
|-------|--------------------|------------|--------|----------|--------|--------|
| 1 | utils_scaling.find_scaling_recursive | rescales full profile frames per recursion | 2355 ms -> 480 ms | -1875 ms | yes | abc1234 |
| 2 | simulation_env.create_observation | dict rebuild per step | within noise | 0 | no (reverted) | - |
```

## Stopping condition

Stop, and report, when **any** of these is true:

- **Plateau:** two consecutive cycles land no change clearing the ±3% noise threshold.
- **Ceiling reached:** the top remaining hotspot is pandapower-internal with no
  call-count reduction available. Say so explicitly — that is a real, reportable result.
- **Risk floor:** the only remaining wins would break the correctness bar. Do not take
  them; propose them as opt-in flags instead and stop.

Do not invent new scope to keep the loop alive. When the measured answer is "it is as
fast as it reasonably gets", that is the finish line.

## Final report

- Cumulative before/after per benchmark measurement, from one back-to-back run.
- Every landed change with its measured contribution and commit.
- Every rejected attempt with why — this is the map for whoever picks this up next.
- Remaining known ceilings, and what would be needed to break them.
