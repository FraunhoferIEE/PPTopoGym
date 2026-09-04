---
name: pptopo-profile
description: Profile and trace the PPTopoGym environment - env construction, reset/step, observations, N-1, and the greedy agents - and report where the time actually goes. Use when asked to profile, trace, benchmark, find hotspots, or investigate why the environment or an agent is slow.
---

# Profiling PPTopoGym

Produce an evidence-backed picture of where wall-clock time goes, at a granularity that
names a function and a line. Do not propose fixes here — that is `pptopo-optimize`.

## Before you measure

Rules that invalidate results if broken:

1. **Timings swing ~5% with machine load.** A single run compared to a number stored in a
   file is not evidence. Anything claiming a speedup must come from `scripts/ab_compare.py`,
   which runs both sides back to back.
2. **Never A/B two variants in the same process.** This is the easiest way to produce a
   completely wrong number, and it has already happened once here: comparing a cold
   `pp.runpp` against a warm `run_powerflow` in one process reported a **9.8× speedup**
   that was really **1.16×**, because whichever variant ran second inherited caches the
   first had filled (pandapower internals, imports, allocator state, the Simbench profile
   cache). Give each variant a fresh interpreter and alternate the order:

   ```bash
   for i in 1 2 3; do
     poetry run python probe.py variant_a | tail -1
     poetry run python probe.py variant_b | tail -1
   done
   ```

   `ab_compare.py` already does this — it runs each side in a subprocess. Ad-hoc probes
   are where this trap bites.
3. **Check whether a cache-based win is reachable in the real workload.** A process-local
   cache shows a huge win on the second call and *zero* on the first. If the production
   path builds one env per process, the cold number is the one that matters. State which
   you measured.
4. **Subtract one-time per-process costs before calling anything a hotspot.** The first
   power flow in any process pays ~1850 ms of **numba JIT compilation** for pandapower's
   kernels (`build_bus.fill_bus_lookup`, `pf.makeYbus_numba.gen_Ybus`,
   `pf.pfsoln_numba.calc_branch_flows`). Whichever function happens to solve first absorbs
   that entire cost and looks like the top hotspot. This is what made
   `find_scaling_recursive` look like 47% of build time when it actually takes **229 ms**
   warm. Before optimizing any stage that runs a power flow, re-measure it with the JIT
   already warm:

   ```python
   us.run_pf(prepared_net())      # burn the JIT
   t = time.perf_counter(); stage_under_test(); warm_ms = ...
   ```

   `NUMBA_CACHE_DIR` does **not** fix this — pandapower declares `cache=True` on only one
   of those kernels (verified 2026-08-25). Treat the ~1850 ms as a fixed floor.
4. **~98% of a step is pandapower marshalling around a ~170 µs solve.** Before reporting a
   step-path hotspot, check whether it is env-owned code or `_pd2ppc`/`_extract_results`.
   Env-side work has a hard ceiling; say so in the report rather than promising a big win.

Check the tree state first — a dirty tree means the profile describes uncommitted work:

```bash
git status --porcelain && git log --oneline -1
```

## Step 1: coarse phase breakdown

```bash
poetry run python scripts/bench_pptopo.py --suite build --suite step --suite obs
```

This gives median-of-N milliseconds per named stage and is the number to quote. Add
`--suite greedy` when the greedy agents are in scope (it is slow — it sweeps the whole
action space serially).

The phase table answers "which stage", not "which line". Expect construction to dominate
a fresh process: `config_case30()` is several seconds, of which `find_scaling_recursive`
and `get_first_sb_profiles` are the bulk.

## Step 2: per-line attribution

```bash
poetry run python scripts/profile_env.py
```

Writes to `profiling/`:

| file | what it answers |
|---|---|
| `env_report.txt` | phase table — % of wall time per phase, p50/p99, RSS growth |
| `env_lines.txt` / `.html` | per-source-line time inside `pandapower_env` only |
| `env_trace.html` | timeline of phases |

`LineProfiler(project_folder=...)` scopes attribution to this package, so pandas and
pandapower internals stay collapsed into the call that entered them. That is what you
want: it separates "our code is slow" from "we called pandapower a lot".

Gotcha: `lineprofiler`'s accounting layer is opt-in. `acc.start(...)` without
`enabled=True` silently records nothing and still exits 0. If a report reads
`Runtime 0ns Processes 0`, nothing was measured — do not interpret it as "fast".

## Step 3: profile the greedy agents

The greedy path has a different cost structure from `step` and must be profiled
separately. It evaluates every legal action via `greedy_worker.evaluate_action`.

```bash
poetry run python -X importtime -c "pass" >/dev/null 2>&1   # sanity: interpreter OK
poetry run python -m cProfile -o /tmp/greedy.prof scripts/run_greedy.py
poetry run python -c "
import pstats; s=pstats.Stats('/tmp/greedy.prof'); s.sort_stats('cumulative').print_stats(30)"
```

When profiling greedy, record these separately — they have different fixes:

- **per-action power flow** — the irreducible core, scales with action count
- **net (de)serialization** — `_ensure_net_from_blob` is cached per process and per blob;
  a cache miss per action means the `to_json` roundtrip is being paid repeatedly
- **topology/profile re-application** — `_apply_topology` / `_inject_profile` per action
- **process pool overhead** — only meaningful with `n_workers > 1`

Note that parallel N-1 auto-degrades to serial inside child processes, so a greedy run
with `n-1 parallel` set is measuring the *serial* N-1 path.

## Step 4: report

Write findings to `profiling/findings-<date>.md`. Required content:

- Tree state (commit + dirty flag) and the machine, so the numbers are reproducible.
- The phase table, with absolute ms and % of total.
- Per hotspot: **file:line**, measured cost, share of its phase, and *why* it is slow
  (algorithmic, per-call overhead, allocation, redundant recomputation).
- An explicit split between **env-owned cost** (fixable here) and **pandapower-owned
  cost** (fixable only by calling it less often).
- Anything measured that turned out *not* to be a hotspot — that saves the next cycle
  from re-investigating it.

Rank by absolute time saved, not by percentage of a small phase. A 40% win on a 2 ms
stage loses to a 5% win on a 2000 ms stage.

## Known-answered questions

Do not re-derive these; they are settled in `CLAUDE.md`:

- lightsim2grid's batch contingency API does not work on these nets (busbar switches).
- `recycle=True` is wrong for this env (stale Ybus under topology switching).
- pandapower >3.1.2 is measurably slower here; the pin is deliberate.
- The observation topology cache keys on lookup *content*, not object identity.
