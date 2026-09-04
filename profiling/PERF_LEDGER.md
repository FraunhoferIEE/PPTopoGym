# PPTopoGym performance ledger

Append-only record of optimization attempts on the PPTopoGym build and hot paths.
Negative results are kept deliberately: they stop the next cycle re-investigating a dead
end. See `.claude/skills/pptopo-perf-loop/SKILL.md` for the process.

All numbers are case30 on `node4` unless stated. Timings on this box swing ~5% with
load, so every figure here comes from **alternating runs in isolated processes**, never
from two variants in one process (see "Measurement traps" below).

## Baseline (cycle 0)

Commit `9123238` + uncommitted work-in-progress. Full suite: **203 passed, 2 xfailed**
in 17m20s. The 2 xfails are the documented observation-space-contract ones.

`config_case30()` + `PPTopoGym(config)` build breakdown, single cold process:

| stage | ms | % of build |
|---|---|---|
| `find_scaling_recursive` | 2356 | 47% |
| `get_first_sb_profiles` | 1340 | 27% |
| `create_all_double_busbar_substations` | 594 | 12% |
| `case30()` load | 226 | 4.5% |
| `verify_all_actions` | 223 | 4.5% |
| `PPTopoGym(config)` | 221 | 4.4% |
| *(remaining stages)* | ~37 | <1% |
| **total** | **~4998** | |

## Attempts

| cycle | target | hypothesis | result | landed | tests |
|---|---|---|---|---|---|
| 1 | `utils_scaling.run_pf` | It was a plain `pp.runpp` per call — the anti-pattern `CLAUDE.md` warns about. 22 power flows per scaling search, each re-parsing options. | **1.16× on `find_scaling_recursive`** (~2345 → ~2022 ms, ~320 ms saved). Results **bit-identical** (line loadings and element powers matched to 10 dp across 6 isolated runs). | yes | `tests/toolbox/test_scaling_warm_pf_parity.py` |
| 2 | `utils_profiles.deterministic_profiles` | `simbench.get_all_simbench_profiles` costs ~1.2 s, is a pure function of the scenario index, and is not cached upstream. It loads 35136×618 values then slices to ~30 columns. | **221× on repeat calls** (1259 → 5.7 ms). No effect on the first build in a process. Slices are `.copy()`d so the shared cache cannot be corrupted (~2 ms). | yes | `tests/toolbox/test_simbench_profile_cache.py` |
| 3 | `find_scaling_recursive` (the apparent 47% hotspot) | It runs 22 power flows; a smarter search (bisection) would cut recursions. | **Abandoned — the premise was wrong.** With the numba JIT already warm the whole search takes **229 ms**, not ~2000 ms. ~1848 ms of its apparent cost is a **one-time numba JIT compilation** triggered by the first power flow in the process, which merely happens to land inside this function. Optimizing the search would relocate that cost, not remove it. | no | — |
| 3b | numba JIT warm-up (~1850 ms/process) | Setting `NUMBA_CACHE_DIR` would persist compiled kernels across processes. | **No effect** (1872 → 1778 ms, within noise). Only 2 cache files are written: pandapower marks just `_python_set_elements_oos` as `cache=True`. The expensive kernels — `build_bus.fill_bus_lookup`, `pf.makeYbus_numba.gen_Ybus`, `pf.pfsoln_numba.calc_branch_flows` — are **not** declared cacheable, so this needs an upstream pandapower change. | no | — |

### Combined effect (cycles 1+2, alternating isolated A/B vs HEAD)

| measurement | HEAD | with changes | change |
|---|---|---|---|
| first env built in a process | 4710 / 4754 ms | 4579 / 4461 ms | ~4% faster |
| second env in the same process | 3281 / 2972 ms | 1509 / 1518 ms | **~2.05× faster** |

Verified-actions count unchanged at 289 in every run, i.e. no behavioural change.

The first-build gain is small because the Simbench read cannot be cached on its first
use. The value is in **multi-env, vectorized and test workloads**, where every env after
the first is ~2× cheaper to build. Test-suite side effect: `tests/toolbox` profile tests
went 12.4 s → 4.9 s.

## Measurement traps hit in this project

- **Two variants in one process gives fabricated numbers.** Cycle 1 first measured a
  **9.83× speedup**; the real figure is **1.16×**. The warm variant had run second and
  inherited caches the cold run filled. Always one variant per interpreter, alternated.
- **A one-shot build script cannot show a cache win.** `build_breakdown.py` builds one
  env, so cycle 2 looked like it did nothing there. Report cold and warm separately.
- **Another session may be running `pytest` in this repo concurrently**, which perturbs
  timings and adds unrelated files to `git status`. Check before benchmarking, and do not
  use the stash-based `ab_compare.py` when foreign uncommitted files are present — copy
  the specific files instead.

## Corrected build budget

Cycle 3 showed the stage table above misattributes a large fixed cost. A cold
`config_case30()` + `PPTopoGym()` really decomposes as:

| component | ms | nature |
|---|---|---|
| numba JIT of pandapower kernels (first PF in the process) | ~1850 | **fixed per process**, not per build; unavoidable without upstream `cache=True` |
| `get_all_simbench_profiles` (first read in the process) | ~1250 | fixed per process since cycle 2 |
| `create_all_double_busbar_substations` | ~520 | per build |
| `find_scaling_recursive` (JIT excluded) | ~230 | per build |
| `case30()` load | ~226 | per build |
| `verify_all_actions` | ~223 | per build |
| `PPTopoGym(config)` | ~180 | per build |

So of a ~4.5 s cold build, **~3.1 s is one-time per-process cost** and only ~1.4 s is
genuinely per-build. That is why the second env in a process now builds in ~1.5 s, and it
sets the realistic floor: **no amount of env-side work makes the first build in a fresh
process much faster than ~3 s** while pandapower's numba kernels stay uncacheable.

## Grid-size scaling (cycle 4, 2026-08-25)

Every number above this section is case30. `scripts/scale_profile.py` measures the whole
build pipeline plus the hot paths across a grid ladder, **one grid per subprocess** so no
grid inherits another's numba/Simbench warm-up. Command:

```bash
poetry run python scripts/scale_profile.py --repeat 5 --n1
```

Validated against this ledger on case30 first: 289 verified actions, `net_load` 226 ms,
`create_double_busbars` ~520 ms, `verify_all_actions` 223 ms, step ~18 ms — all match.

| metric | case30 | case89pegase | ratio |
|---|---|---|---|
| substations | 10 | 38 | 3.8× |
| ppc buses (solved system) | 30 | 89 | 3.0× |
| lines | 41 | 160 | 3.9× |
| **actions** | **289** | **155 065** | **537×** |
| `verify_all_actions` | 223 ms | **193 132 ms** | 865× |
| `PPTopoGym.__init__` | 213 ms | **87 812 ms** | 412× |
| `create_double_busbars` | 533 ms | 2 552 ms | 4.8× |
| **total build** | **4.9 s** | **288 s** | **59×** |
| step (DoNothing) | 18.4 ms | 22.2 ms | 1.2× |
| N-1 serial | 400 ms | 2 329 ms | 5.8× |

### What this says

- **The action space is the binding constraint, and it is combinatorial.** A 3.8× increase
  in substations produced a **537×** increase in actions. Per-action verify cost is nearly
  flat (0.77 → 1.25 ms), so `verify_all_actions` is not slow *per action* — there are just
  155 065 of them, each costing a power flow. This is the wall, not the step path.
- **`PPTopoGym.__init__` inherits the explosion** (88 s), because `_build_action_plans`
  resolves labels to positions once per action. Linear in a number that is itself blowing up.
- **The step path scales fine** (1.2× for a 3× grid) — consistent with `CLAUDE.md`'s
  "~98% pandapower marshalling around a 170 µs solve". Optimizing it further does not move
  the grid-size ceiling at all.
- N-1 grows ~1.5× faster than line count (5.8× for 3.9× lines), as expected for
  lines × cost-per-solve.

### Hard ceiling: >96 loads cannot be built at all

case118 and case300 **fail outright**, and not for performance reasons:

```
ValueError: Length of values (96) does not match length of index (99)
  utils_profiles._add_column_names -> net[sth]["profile"] = column_names[sth]
```

`_add_column_names` assigns Simbench profile columns to elements **positionally, one
profile per element**, so a grid cannot have more loads than the Simbench library has
distinct load profiles. Measured availability:

| sb_index | unique load profiles |
|---|---|
| 0, 1, 2 | 96 |
| 3, 4, 5 | 27 |

So the package supports **at most 96 loads** (27 on half the scenario indices).
case118 has 99 loads, case300 has 193. This is a modelling limit, not a tuning knob:
fixing it means sampling/recycling profiles with replacement rather than a positional
1:1 assignment. Note `deterministic_profiles` already slices `iloc[:, :n_loads]`, so the
intent was 1:1 all along.

### Consequences for the roadmap

The realistic ceiling today is **~90 buses / ~40 substations / ~96 loads**. Before any
further constant-factor work:

1. **Cap or prune the action space.** 155 k actions is not just slow to build — it is not a
   tractable RL action space, and `n-1-topk` has no equivalent for actions. This is a
   design decision (rule-based pruning, top-k by sensitivity, hierarchical/factored
   actions), not an optimization.
2. **Lift the 96-load profile ceiling**, or no grid past case89 can be loaded.
3. Only then does `verify_all_actions` per-action cost matter.

## Cycle 5 (2026-08-26) — the simulation path, and correctness found while profiling

Cycles 1-4 measured *build* and *step*. Neither covers `simulation()` / `verify_action()`,
which is what an agent actually calls per candidate action. Counting power flows there found
the first real waste since cycle 2, and reading the same code found four defects.

| cycle | target | hypothesis | result | landed | tests |
|---|---|---|---|---|---|
| 5a | `simulation_env.end_simulation` | It restores state by calling the *public* `reset()`, which runs a power flow and builds a full observation — then immediately replays the action log and solves again. Both describe the pristine topology and are discarded. | **`simulation([a])` 4 power flows → 3.** Alternated isolated A/B, 2 rounds: `simulation([1])` 37.0 → 27.5 ms (**1.35×**), `simulation([1,2,3])` 74.3 → 66.1 ms (1.12×, the saving is per *call*), `verify_action(1)` 20.4 → 11.2 ms (**1.82×**). | yes | `tests/environments/test_env_robustness.py` |
| 5b | `toolbox.utils.total_active_overload_mva` | `net.bus.vn_kv.loc[from_bus]` is a pandas label reindex for what is a positional take on every net this package builds. | **308 → 90 µs (3.4×).** 221 of the 308 µs were that one lookup. Guarded on `index.equals(RangeIndex)` (~11 µs) so a non-positional net still goes through `.loc`. | yes | `tests/toolbox/test_utils.py` |
| 5c | `base_agents.BaseGreedyAgent.act` | The "grid is fine, do nothing" early-out sits *below* `state_from_info` (a full action-log replay + power flow) and the worker-payload packing. | Moved to the top of `act`. No benchmark: it only fires when the grid is under the overload threshold, where it now skips ~1 power flow + the payload build entirely. | yes | covered by existing agent tests |

Net effect on the paths cycles 1-4 did measure: `ab_compare --suite step --suite obs`
reports **all four measurements within the ±3% noise band**, so the correctness work below
(which added two aggregates to every step's `info`) landed for free after 5b.

### Correctness defects found in the same pass

Profiling kept walking through code that was quietly wrong. All four are fixed; none
changes the public API, the observation space, or `step`'s return shape.

1. **`reward_better_than_donothing` could never run.** Its DoNothing rollout called
   `env.step()`, which called the reward again, which started another rollout →
   `RecursionError` on the first step. It also unpacked four values from `step`'s
   five-tuple, and its inner `lru_cache` was rebuilt on every call so it never memoised.
   The only test asserted the function had been *bound*, never that it ran. Rewritten with
   a recursion guard and `save_state`/`restore_state` (not `start`/`end_simulation`, which
   would restore the topology from an action log that does not yet contain the action being
   scored, corrupting the caller's observation).
2. **Two evaluation metrics were permanently NaN.** The default
   `info_observations` names `total_energy_overload` and `max_loading_percent`; neither had
   a registry entry, so `create_observation` silently dropped them and
   `overload_energy_difference_abs_mvah` / `loading_improvement_optimization` returned NaN
   for every step ever evaluated. `PPTopoGym._get_aggregate_value` had implemented both all
   along — only the config entries pointing at it were missing. Added via a *separate*
   `build_info_observation_registry()`, because an env defaults `observation_keys` to every
   key of the main registry and growing it would move the observation space.
3. **`step()` walked off the end of the timeseries.** `self.index += 1` was unconditional,
   so a reset to a late `options["index"]` — the documented way to pick a scenario — died
   with `KeyError: <n_timesteps>`. Now truncates the episode, which is what `truncated`
   means. Unreachable from the default random scenario start, which is why it survived.
4. **`total_active_overload_mva` used the wrong line/trafo rating.** It divided by
   `max_i_ka` alone, ignoring `df` and `parallel` — the exact terms pandapower divides by
   for `loading_percent`. Both default to 1, which is why case30 never showed it; a derated
   line was reported as less overloaded than it is and a double circuit as more.
5. **`simulation()`'s bounds check was `isinstance(action, int)`**, which numpy integers
   fail — so the values agents actually return (`action_space.sample()`, `argmax`) skipped
   validation and surfaced as `KeyError` deep inside `load_action`; negative indices were
   never checked and silently applied the last action row.

### Known, deliberately not changed

- **A crashed step leaves the net desynced from `log_actions`.** `PPTopoGym.step` returns
  early on non-convergence without appending the action, so the grid carries an action the
  log does not — and `end_simulation` / `state_from_info` then restore a *different*
  topology. Harmless today because `terminated=True` ends the episode, but it is a real
  trap for anyone who resumes from such a state. (`simenv` actions 1, 2 and 4 hit this,
  which is why the new tests pin 3 and 5.)
- **`info["*_before"]` is taken *after* the action is applied**, and `*_after` at index+1.
  So `loading_improvement_optimization` compares timestep t (with action) against t+1 (with
  action), not "before vs after optimization" as its name and docstring say. Fixing the
  semantics would change every published metric value, so it is reported, not changed.
- **The adjacency observation keeps out-of-service lines as edges.** Deliberate: the
  declared space is a fixed `(n_line + n_trafo, 2)`, and `line_status` carries the
  in-service flag separately. Dropping dead edges would make the shape vary per step.

## Cycle 6 (2026-09-02) — the step path under a realistic 30/70 action mix

Cycles 1-5 measured the step with a single action kind. Re-profiled with **withlineprofiler
0.8.3** against the mix an agent actually produces (**30% DoNothing / 70% topology actions**,
seeded, stepping continuously inside episodes), which is what exposed 6a: DoNothing steps cost
the *same* as acting steps, because the env re-solved a grid nothing had touched.
Full report: `profiling/findings-2026-09-02.md`; artifacts in `profiling/mix_30_70/`.

| cycle | target | hypothesis | result | landed | tests |
|---|---|---|---|---|---|
| 6a | `gym_env_pp.run_pf` | `load_action(0)` deliberately preserves `net.converged`, but `step` solves unconditionally — so a DoNothing step re-solves the identical topology, injections and profile index. One wasted power flow on 30% of steps. | Return early when `net.converged is True` *and* `net["_ppenv_solved_for"]` matches the `(pf_type, use_ls2g, nminus1)` request. **Bit-identical** (golden record 895/895). | yes | `tests/environments/test_resolve_skip_parity.py` |
| 6b | `gym_env_pp._grid_is_disconnected` | `res_bus["vm_pu"].reindex(in_service_buses)` is a full pandas reindex for what is a positional mask on every net this package builds — and it runs once per power flow. | **151 → 10 µs (15×).** Guarded on `res_bus.index.equals(net.bus.index)`; `_grid_is_disconnected_by_label` kept as fallback *and* oracle. Same shape as 5b. | yes | `tests/environments/test_disconnect_parity.py` |
| 6c | `toolbox.ls2g_backend` write path | Six `pd.DataFrame({...})` per solve sanitize and box each column separately; and `_write_bus_lookup` / `_write_bus_results` each rebuilt the same bus→mirror mapping, the former in a per-bus Python loop. | `_result_frame` wraps one `column_stack`ed float64 matrix (~130 µs/solve); `_resolve_bus_positions` derives the mapping once; the lookup fill is vectorized. **Bit-identical** over 457 arrays. | yes | `tests/toolbox/test_ls2g_backend.py` |

### Combined effect (alternated isolated A/B vs HEAD, 3 rounds each, one variant per interpreter)

Weighted 30/70 step, case30, median of 300 steps per run:

| backend | before | after | speedup |
|---|---|---|---|
| pandapower (default) | 20.22 / 20.22 / 19.89 ms | 17.24 / 17.25 / 17.09 ms | **1.17×** (−14.5%) |
| `backend="lightsim"` | 5.32 / 5.36 / 5.37 ms | 4.52 / 4.49 / 4.50 ms | **1.19×** (−15.9%) |

Spread within each variant is under 2%, so the box was quiet; both wins are far outside the
±3% noise band. Stacked on the backend switch itself, a step is **20.1 → 4.50 ms (4.5×)**.

The correctness gate was the full `scripts/golden_record.py` (895 fingerprints across build,
episode, simulation API, N-1 and greedy on case14 + case30) for the pandapower path, and a
457-array before/after dump of every `res_*` table, reward, bus lookup and N-1 aggregate for
the lightsim path. Both came back **bit-identical** — the 1e-6 tolerance the ls2g backend is
allowed was not needed. `tests/toolbox/test_utils_graph_obs.py::test_topology_caches_key_on_lookup_content`
needed a one-line setup fix: it asserted that `run_pf` allocates a fresh `_pd2ppc_lookups`,
which is only true when a solve actually happens, so it now clears `converged` first.

### Not worth doing (measured)

- **`res_switch`** is rebuilt by pandapower on every solve (~2 ms per step pair) and nothing
  in the env reads it — but it is written inside `_extract_results`, so skipping it means
  patching pandapower, not this repo.
- **The two-power-flows-per-step structure** is env semantics (reward at index k, observation
  at k+1), not waste. Only the DoNothing case was redundant, and 6a took it.
- **Env-owned observation micro-costs** (`_fill_nan`, the per-key dispatch in
  `_get_default_observation`) are ~0.5-1.5 ms/step across ~50 keys × 3 passes. Real, but every
  item above paid more for a smaller diff.

## Cycle 7 (2026-09-03) — inside the lightsim backend's own solve

Cycle 6 left `backend="lightsim"` at ~4.5 ms/step and the README put grid2op's
`LightSimBackend` at ~1.0 ms on a comparable grid, so this cycle profiled *inside*
`LightsimBackend.solve` rather than around it. Phase split on case30 (medians, 300-400 reps,
one interpreter, current tree — the box was slower today than on 2026-09-01, so read the
**shares**, not the absolute µs):

| phase | µs | share of solve |
|---|---|---|
| `_write_results` | 809 | **54%** |
| `ac_pf` (the actual Newton solve) | 178 | 12% |
| `_sync_active_buses` | 118 | 8% |
| `_push_topology` | 114 | 8% |
| `_push_injections` | 67 | 4% |
| `current_bus_assignment` | 24 | 2% |

**The write-back, not the push, is what a lightsim solve spends its time on.** Within it:
`_write_injection_results` 322 µs, `_write_trafo_results` 138 µs, `_write_line_results` 136 µs,
`_write_bus_results` 91 µs, `_resolve_bus_positions` 41 µs, `_write_bus_lookup` 16 µs.

| cycle | target | hypothesis | result | landed | tests |
|---|---|---|---|---|---|
| 7a | `ls2g_backend._write_trafo_results` / `_write_injection_results` | A net with no transformers and no static generators publishes those tables as `pd.DataFrame()` on **every** solve. The no-argument constructor is ~132 µs — more than twice a real 41x10 frame — so case30 pays ~264 µs per solve for two tables that are empty. | Copy a module-level prototype (`_empty_result_frame`): **~132 → ~8.5 µs each**. Solve **1448 → 1128 µs (1.28×)**, weighted 30/70 step **6.43 → 5.69 ms (1.13×)**, 3 alternating rounds, one interpreter each, spread <2%. | yes | `tests/toolbox/test_ls2g_backend.py::test_absent_element_tables_are_empty_and_never_shared` |
| 7b | `ls2g_backend._push_topology` / `_sync_active_buses` | Both walk every element terminal in Python with a dict lookup per element; resolving a whole table through an assignment *lookup array* and pushing only the changed terminals should cut the ~230 µs they cost. | **Rejected — measurably slower.** `_push_topology` **114 → 173 µs**; `_sync_active_buses` improved 118 → 95 µs, but the net was **+36 µs/solve (~2% slower)** across 3 alternating rounds. See below. | no | oracle kept: `test_vectorized_push_matches_the_per_element_oracle` |

### Why 7b failed, and what it rules out

The tables are too small for numpy. On case30 the push walks **109 element terminals over 5
`(table, bus column)` pairs** (line from/to, load, gen, shunt; trafo and sgen are empty and
skipped). The vectorized form pays a fixed ~1-2 µs per numpy operation (take, compare,
`flatnonzero`, the `< 0` guard) times ~6 operations times 5 columns, plus ~11 µs to build the
lookup array — which is more than the Python loop over those 109 elements it removes. This is the general shape: **on these grids the per-element loops are not the cost;
per-*call* pandas and numpy overheads are.** 7a is the same lesson from the other side — one
badly-chosen constructor called twice per solve outweighed every element loop in the module.

The oracle written for 7b was kept even though the change was reverted:
`PerElementPushOracle` drives a second `GridModel` through the per-element loops and asserts the
two models end up in exactly the same state (every terminal bus, every in-service flag, every bus
activation, and bit-identical `get_Vm` / `get_Va`). It pins the push semantics for any future
attempt, and it verified 7a did not disturb them.

### Not worth doing (measured)

- **Vectorized injection push** (`update_loads_p` / `update_gens_p` / `update_sgens_p` and
  friends, the API grid2op's own backend uses) is **~9x faster per call** — 13.2 → 1.5 µs for
  20 loads — but its pybind signature is `numpy.ndarray[numpy.float32]`, while
  `change_p_load(i, v)` takes a C++ `double`. case30's injections carry precision past float32
  (max round-trip delta **1.06e-7 MW**), so adopting it would silently truncate every setpoint.
  It saves ~55 µs of a ~1450 µs solve, which does not buy a precision change. If it is ever
  wanted it must be tolerance-gated, not slipped in as an equivalent rewrite.

## Not yet attempted (ranked for the next cycle)

1. **Action-space pruning / profile ceiling** — see cycle 4. These gate grid size;
   everything below is a constant factor on an already-tractable grid.
2. `create_all_double_busbar_substations` — ~520 ms on case30 and the largest genuinely
   per-build stage there. Never profiled at line level. Scales benignly (4.8×).
3. `case30()` load at ~226 ms — pandapower-owned; likely cacheable per process like
   cycle 2 (it returns a fresh net each call, so any cache must deep-copy).
4. Step path (~18.7 ms/step) and greedy sweeps — per `CLAUDE.md` ~98% pandapower
   marshalling, and cycle 4 shows it scales at 1.2×, so the ceiling is low *and* it is not
   what limits grid size.
5. Upstream: propose `cache=True` on pandapower's numba kernels. Biggest single remaining
   win (~1.85 s per process) but not fixable in this repo.
6. **The rest of `ls2g_backend._write_results`** (still ~45% of a solve after 7a). The four
   remaining `_result_frame` calls are ~57 µs each, and the static per-net terms they recompute
   every solve (`max_i_ka * df * parallel`, the index objects, the trafo rating scale) are
   ~16 µs. Beyond that the structural fix is the one grid2op does not need: stop materializing
   `res_*` DataFrames the observation builder immediately reads back column by column, and hand
   the observation path the numpy arrays directly. That is a cross-module change
   (`utils_graph_obs.batch_observations` + `_get_table_value`), not a local one.
7. **Batched N-1** via `ContingencyAnalysisCPP` on the mirror model, and native islanding
   instead of the `_solve_with_pandapower` fallback (6 of 41 contingencies on case30, 56 of 210
   on case89, each a full ~9-10 ms pandapower solve — most of what the sweep still costs).
