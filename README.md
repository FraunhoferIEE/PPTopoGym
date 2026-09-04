# PPTopoGym

## Getting Started

The module is built for distribution using poetry and we recommend the same for usage and development.

### Installation

The following steps should help you install gridgenerate in a virtual environment. Note that Python and poetry must be installed totally independent of Anaconda to avoid conflicts!

Step 1. Download and install Python 3.10.11 from [python.org](https://www.python.org/downloads/release/python-31011/)

Step 2. Install Poetry using the official installer and add Poetry to your PATH. See [python-poetry.org](https://python-poetry.org/docs/#installing-with-the-official-installer) for instructions.

Step 3. Create a virtual environment: `poetry env use <path_to_python_executable>`

Step 4. Install: `poetry install`

## The code

The code basis are the environments `GymEnvPP` and `SimulationEnv`, saved in `pandapower_env/environments`.

### Overview of the code

For demonstration, we provide several notebooks. Start with these two:

- [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) walks through the environment end to end: loading a grid, creating double-busbar substations, building the action space and stepping an agent.
- [`notebooks/environment_configuration.ipynb`](notebooks/environment_configuration.ipynb) is the reference for `env_config` — every configuration key of `PPTopoGym`, what it defaults to and what changes when you set it, ending with the ready-made configurations in `pandapower_env/data/example_configs.py`.

The remaining notebooks go into single topics:

- `action_space.ipynb` shows how the action space is built and how unrealistic grid states are filtered out.
- `detect_substations.ipynb` shows how substations are detected in a pandapower net, and `mbb_actions.ipynb` how the multi-busbar substation actions are encoded.
- `pst_config.ipynb` shows the phase-shift-transformer configuration.
- `scaling.ipynb` shows how to select profiles based on the renewable energy generation, and scale them to the net such that overloads occur.
- `heuristic_agents.ipynb` shows how the provided benchmark agents are used on a power grid, and `inspect_case30.ipynb` inspects one grid in detail.
- `ray_notebook.ipynb` trains the RLlib GNN module on the environment. It needs `torch` and `ray`, and takes long enough that CI skips it.
- `evaluation.ipynb` compares agents with the metrics in `pandapower_env/metrics/` — the metric catalog, how to score an action sequence, how to register your own metric — and produces the agent-comparison figures in `notebooks/plots`. It runs the greedy agent, which takes minutes, so CI skips it.


### Details about the code

- In `action_space`, actions for switching double busbars, and switching lines, are created. Further, unrealistic actions are filtered out.
- In `agents`, the agents are created object-oriented: There exist base-agents with core functionalities and built upon benchmark-agents that can be used by users. The greedy agents simulate every action and evaluate it using a feedback function, and the greedy rollout agents simulate for each action additional rollouts, and evaluate these rollouts using a feedback function for the rollouts, and then again aggregate the rollout feedback to generate an overall evaluation for each action.
- In `environments`, the two basis environments are stored. The more sophisticated `SimulationEnv` is able to simulate actions and restore the power grid state. It implements simple observations, e.g., graph observations and maximal line loadings.
- In `metrics`, a class for evaluating a sequence of actions is provided.
- In `observation_space` functions to create observations for the agent are gathered
- In `substation` functionalities to create and plot double busbars are gathered.
- In `toolbox` additional utility functions, like scaling the power grid elements, or running a N-1 power flow, are gathered.

### Notable configuration keys

`PPTopoGym(env_config)` takes a serializable dict. Beyond the required `net`, `n_episodes`, `episode_length` and `action_space`, a few optional keys change behaviour or cost significantly — the full list is in [`notebooks/environment_configuration.ipynb`](notebooks/environment_configuration.ipynb):

| key | default | effect |
|---|---|---|
| `nminus1` | `False` | Evaluate N-1 contingencies each step. Much slower, and `run_pf` then requires `max_loading_percent` in `res_line`. |
| `n-1-topk` | `100.0` | Only the top *k* % of lines by N-0 apparent power flow are switched off as contingencies. Cuts the *contingency* set, not the *monitored* set, so reported loadings can only decrease vs. full N-1. |
| `n-1 parallel` / `n-1 workers` | `False` / all CPUs | Spread contingencies over loky workers. Bit-for-bit identical to serial. Auto-degrades to serial inside child processes, so it is a no-op in spawned workers and in the parallel greedy agent. Useful workers ≈ contingencies / 6 — cap it for small grids. |
| `static_obs_space` | `False` | Declare node observations at their static upper bound and zero-pad. Needed for `SyncVectorEnv` / `AsyncVectorEnv`: node observation length grows as substations split, so by default most observations fail `observation_space.contains(obs)` and the vector envs crash. **Opt-in**, because enabling it changes observation shapes. Rewards are identical either way. |

## Profiling

Profiling and benchmarking are kept out of the core environment: nothing under `pandapower_env/` imports a profiler or a benchmark helper, so the shipped env carries no extra overhead or dependency. The tooling lives in `scripts/`:

```bash
poetry run python scripts/profile_env.py                  # phase + per-line profile
poetry run python scripts/bench_pptopo.py                 # median-of-N benchmark suites
poetry run python scripts/bench_pptopo.py --suite build   # one suite (build/step/obs/greedy)
poetry run python scripts/ab_compare.py --suite build     # working tree vs. stashed baseline
poetry run python scripts/profile_memory.py               # per-environment memory cost
poetry run python scripts/branch_compare/run_comparison.py \
    --grids DIR --worktree PATH_TO_OTHER_CHECKOUT         # this branch vs. another branch
```

`profile_env.py` profiles the hot path from the outside using [`with-line-profiler`](https://github.com/mathematiger/withlineprofiler). Its workload is 10 topology steps (substation switching actions spread over different substations) plus 3 DoNothing steps on `config_case30`, each preceded by a reset, with a warm-up step outside the measurement.

`bench_pptopo.py` is the stable A/B baseline: each suite reports the **median** of `--repeat` runs under a machine-readable key, so two runs can be diffed.

> **Measuring correctly matters more than usual here.** Timings on this hardware swing several percent with background load, and — the trap this project actually hit — two variants benchmarked *in the same process* produce fabricated numbers, because whichever runs second inherits the caches the first filled. One optimization first measured 9.83× that way when the true figure was 1.16×. Always run one variant per interpreter and alternate the order; `ab_compare.py` does this via subprocesses. Likewise, a script that builds a single environment cannot show a per-process cache win at all — quote cold and warm numbers separately.

### Results

Measured on a 128-core node (120 cores available), `config_case30`, AC power flow, N-1 off. The phase table below was recorded at commit `a3536bc`; the build-path optimizations landed since then (Simbench cache, warm scaling power flow) mainly speed up the *second and later* env built in a process, which this single-build workload does not exercise — see [`profiling/PERF_LEDGER.md`](profiling/PERF_LEDGER.md) for the per-change numbers.

| Phase | Entries | Wall time | % of run | p50 | p99 |
|---|---|---|---|---|---|
| `build_env` | 1 | 5.05 s | 87.3 % | 5.10 s | 5.36 s |
| `step_topology` | 10 | 353.0 ms | 6.1 % | 35.9 ms | 45.7 ms |
| `reset` | 11 | 253.9 ms | 4.4 % | 20.1 ms | 58.3 ms |
| `step_donothing` | 3 | 96.8 ms | 1.7 % | 32.5 ms | 33.5 ms |
| `warmup` | 1 | 32.4 ms | 0.6 % | 32.5 ms | 33.5 ms |

Total runtime 5.84 s, of which 98.6 % is on-CPU (the run is compute-bound, not waiting).

**Speed.** Building the environment dominates any short run — it scales the grid, creates the double-busbar substations and verifies every action — which is why `config_case30()` should be built once and reused (via `env.orig_config`) rather than rebuilt per agent. Once built, a topology step costs ~36 ms (p50) and a DoNothing step ~33 ms; a reset costs ~20 ms (p50, mean 23.1 ms).

A full `env.step()` measured **45.3 ms before / 21.0 ms after** the step-path optimisations below (2.2×). Every change here leaves results bit-for-bit identical:

- **Warm power-flow options** (`toolbox/utils.run_powerflow`) — about two thirds of every `pp.runpp` was pandapower re-parsing its options (`_init_runpp_options`, ~16 `DataFrame.query` calls per step), not solving. Parsing once and reusing the low-level `_powerflow` is ~2.5–2.9× faster per power flow. The same fix was later applied to the scaling search (`utils_scaling.run_pf`, ~1.16× on `find_scaling_recursive`).
- **Content-keyed observation cache** (`toolbox/utils_graph_obs`) — pandapower allocates a fresh `_pd2ppc_lookups` on every power flow, so an identity-keyed cache rebuilt the whole node mapping once per step (~1.5 ms). Comparing the bus-lookup *content* (~3 µs) keeps it alive across timesteps and rebuilds only on a real topology change.
- **Positional action writes** (`PPTopoGym._build_action_plans`) — action switch/line/trafo labels are resolved to row positions once at init, so `load_action` is numpy fancy-index assignment instead of three `DataFrame.loc` label writes: ~460 µs → ~10 µs.
- **Union-find bus-switch traversal** (`toolbox/topology_helpers`) — replaced a per-bus fixpoint loop with a single components pass: case118 all-buses **209.6 ms → 0.81 ms (260×)**.
- **Per-process Simbench profile cache** (`utils_profiles`) — `get_all_simbench_profiles` costs ~1.2 s, is a pure function of the scenario index and is not cached upstream: 1259 ms → 5.7 ms on repeat calls. Every env after the first in a process is **~2× cheaper to build** (~3.0–3.3 s → ~1.5 s).

**Where the remaining build time goes.** Of a ~4.5 s cold build, **~3.1 s is one-time per-process cost, not per-build**: ~1850 ms is numba JIT of pandapower's kernels (triggered by whichever function runs the first power flow) and ~1250 ms is the first Simbench read. Only ~1.4 s is genuinely per-build. This sets a realistic floor — no env-side work makes the *first* build in a fresh process much faster than ~3 s while pandapower's numba kernels stay uncacheable (`NUMBA_CACHE_DIR` does not help; only one kernel is declared `cache=True`).

**The step-path ceiling.** ~98 % of a step is pandapower marshalling around a ~170 µs solve (`_pd2ppc` + `_extract_results` + the ppc→ppci reduction). `recycle` would skip that rebuild but is **wrong here**: it reuses the old Ybus and silently ignores topology switching, which is the entire point of this environment.

**Inside the lightsim backend.** Everything above is the *pandapower* step path. Profiling *within* `LightsimBackend.solve` rather than around it (2026-09-03) moves the target completely: the result write-back costs more than four times the solve it publishes.

| phase of a case30 `solve` | share |
|---|---|
| `_write_results` | 54 % |
| `ac_pf` (the Newton solve itself) | 12 % |
| `_sync_active_buses` | 8 % |
| `_push_topology` | 8 % |
| `_push_injections` | 4 % |
| `current_bus_assignment` | 2 % |

The largest single item inside it turned out to be a constructor. **`pd.DataFrame()` with no arguments takes ~132 µs** — more than twice a populated 41 × 10 frame (~57 µs). case30 has no transformers and no static generators, so every solve built two of those *empty* tables: ~264 µs, 18 % of the whole solve, for tables with no rows. `_empty_result_frame` copies a module-level prototype instead (~8.5 µs, **15×**), and still returns a fresh object per solve — which is what stops a caller holding last step's table from seeing it change underneath. Measured over 3 alternating rounds, one interpreter per variant: `solve` **1448 → 1128 µs (1.28×)**, weighted 30/70 step **6.43 → 5.69 ms (1.13×)**.

Two changes were tried here and **rejected on measurement**:

- **Vectorizing the per-element push loops is *slower*** — `_push_topology` **114 → 173 µs**. The push walks 109 element terminals spread over 5 table columns on case30, which sits below the size where numpy's ~1–2 µs per-operation overhead pays for itself: six operations per column cost more than the Python loop they replace. The lesson generalises: cost in this module is per *call*, not per element, which is the same thing the `pd.DataFrame()` finding says from the other side. `PerElementPushOracle` in `tests/toolbox/test_ls2g_backend.py` keeps the per-element loops as an oracle — it drives a second `GridModel` and asserts identical model state — so a future attempt starts with its correctness gate already written.
- **lightsim2grid's vectorized `update_loads_p` / `update_gens_p`** — the API grid2op's own backend uses, and ~9× faster per call — takes **float32**, while `change_p_load` takes a double. The injections here carry precision past float32 (round-trip delta **1.06e-7 MW**), so adopting it would silently truncate every setpoint, to save ~55 µs of a ~1450 µs solve. It would have to be tolerance-gated, not slipped in as an equivalent rewrite.

Absolute microsecond figures in this subsection come from a busier box than the 2026-09-01 measurements quoted further down (which put the same `solve` at 0.625 ms), so compare the *shares* and *ratios* across sections, not the raw µs.

**Memory.** Peak RSS 1003 MB, growing 306 MB over the run — essentially all of it under `build_env` (56.0 MB/s), while stepping is near-flat. I/O is served entirely from the page cache (114.0 MB read, 0 B from disk, 0 B written). Profile tables used to dominate per-env memory (~53 of ~54 MB); they are now **shared** between environments rather than copied, bringing the per-env cost to **~0.7 MB (73× less)** and making the vectorized/multi-env setting affordable. This is sound only because the tables are read-only after `setup_profiles` — never mutate `net.profiles` or a `df_profiles_*` in place; rebind instead.

### Output files

The run writes these to [`profiling/`](profiling/):

- [`env_report.txt`](profiling/env_report.txt) — phase/resource report (the table above)
- [`env_report.json`](profiling/env_report.json) — same report, machine-readable
- [`env_trace.html`](profiling/env_trace.html) — timeline of the recorded spans
- [`env_trace.json`](profiling/env_trace.json) — raw trace
- [`env_lines.html`](profiling/env_lines.html) — per-line profile, browsable with source
- [`env_lines.txt`](profiling/env_lines.txt) — top lines by time

Alongside them, [`profiling/PERF_LEDGER.md`](profiling/PERF_LEDGER.md) is an append-only record of every optimization attempt — including the **rejected** ones and the measurement traps that produced them, kept so the next cycle does not re-investigate a dead end.

### Speed: this version vs. the ECML 2025 release, and vs. grid2op

`scripts/branch_compare/` times two checkouts of PPTopoGym against each other on the **same physical grid**, across grid sizes. The baseline is the published ECML 2025 release ([`FraunhoferIEE/PPTopoGym`](https://github.com/FraunhoferIEE/PPTopoGym), branch `ecml2025`, commit `ff9a990`). Measured 2026-08-27 on `node4` (pandapower 3.1.2, lightsim2grid 0.10.3), median of 3 rounds × 5 repetitions, one interpreter per measurement with the two versions alternating per round. The indented rows add [grid2op](https://grid2op.readthedocs.io) 1.12.5 on the nearest-size RTE grid it ships, same box, same method (30 steps per step figure, 3 full sweeps per N-1 figure, backends alternated):

| Grid | Backend | DoNothing step | Switching step | N-1 sweep |
|---|---|---|---|---|
| **case14** (14 bus → 57 / 7 subs, 20 contingencies) | PPTopoGym, pandapower (+ lightsim2grid solver) | 64.3 → 19.5 ms (**3.29×**) | 64.2 → 20.9 ms (**3.07×**) | 457.9 → 222.1 ms (**2.06×**) |
| ↳ grid2op `l2rpn_case14_sandbox` (14 subs, 20 lines) | grid2op, pandapower backend | 15.3 ms | 16.9 ms | 300.7 ms |
| ↳ same grid2op env | grid2op, `LightSimBackend` | 0.92 ms | 1.07 ms | 11.3 ms (batched: **0.43 ms**) |
| **case30** (30 bus → 93 / 10 subs, 41 contingencies) | PPTopoGym, pandapower (+ lightsim2grid solver) | 87.3 → 18.4 ms (**4.73×**) | 88.5 → 19.5 ms (**4.54×**) | 880.5 → 387.3 ms (**2.27×**) |
| ↳ grid2op `l2rpn_neurips_2020_track1_small` (36 subs, 59 lines) | grid2op, pandapower backend | 15.7 ms | 17.3 ms | 916.1 ms |
| ↳ same grid2op env | grid2op, `LightSimBackend` | 0.98 ms | 1.22 ms | 37.8 ms (batched: **1.95 ms**) |
| **case89** (89 bus → 476 / 38 subs, 210 contingencies) | PPTopoGym, pandapower (+ lightsim2grid solver) | 644.5 → 22.9 ms (**28.1×**) | 641.7 → 24.2 ms (**26.5×**) | 4935.4 → 2315.1 ms (**2.13×**) |
| ↳ grid2op `l2rpn_wcci_2022` (118 subs, 186 lines) | grid2op, pandapower backend | 18.6 ms | 19.7 ms | 3499.1 ms |
| ↳ same grid2op env | grid2op, `LightSimBackend` | 1.24 ms | 1.63 ms | 174.2 ms (batched: **18.9 ms**) |

Left figure is the ECML 2025 release, right is this version; "→ 57 / 7 subs" is the grid after the double-busbar expansion. Both versions expanded all three grids identically (same substation, switch, bus and action counts) and solved identical injections, so the difference is code, not physics.

The grid2op rows are **not the same networks** — they are RTE-derived grids of comparable size, where *every* substation is a switchable double busbar — so read them as an order-of-magnitude reference, not a like-for-like delta. Their step is one `env.step()` on a chronic (injections advance, power flow, full observation), the same shape as a step here; the switching step alternates split/merge `set_bus` at the first substation with ≥ 4 elements, with substation and line cooldowns disabled. The N-1 sweep is one power flow per line outage driven directly on the backend (disconnect → solve → read flows → reconnect), the analogue of `run_nminus1_powerflow`; "batched" is lightsim2grid's `ContingencyAnalysis` doing the whole sweep in one call.

**Which environment runs on which backend.** Every row in the table is either PPTopoGym on pandapower (the bold rows) or grid2op on one of grid2op's two backends (the indented rows) — **no row is PPTopoGym on lightsim2grid via `toolbox/ls2g_backend.py`.** The two lightsim paths have near-identical names and are easy to confuse:

| Name | Whose | What it is |
|---|---|---|
| grid2op `LightSimBackend` | grid2op | grid2op's C++ backend, replacing its `PandaPowerBackend`. Drives the indented `LightSimBackend` rows above. |
| `LightsimBackend` (`toolbox/ls2g_backend.py`) | **this package** | PPTopoGym's own mirror-net lightsim2grid path (`backend="lightsim"`). Opt-in, **not** in the table — its figures are in the bullet below (0.625 ms vs. `run_pf`'s 9.016 ms on case30). |

Note that the bold PPTopoGym rows still use lightsim2grid *as pandapower's solver* (`use_ls2g="auto"`, see below) — that is pandapower calling lightsim2grid for the ~170 µs Newton solve while doing all the marshalling itself, which is a different thing from either backend above.

- **A switching action costs the same as a DoNothing action** in both versions — applying the topology is under a millisecond of the step, which is otherwise a power flow plus an observation. The axis that matters is DoNothing vs. N-1, not DoNothing vs. switching.
- **Turning N-1 on costs 42× a plain step on case30 and 207× on case89** (on the ECML release: 21× and 16×, only because its plain step is already slow). An N-1 *step*, measured net of reset, is ≈ 2 × a bare sweep in **both** versions — something in the step path solves the contingency set twice. Same factor on both sides, so not a regression, but worth chasing.
- **N-1 gains least** (2.1–2.3×): a sweep is dominated by pandapower's per-contingency `runpp`, which neither version changed. What this version saves there is the parsed-options reuse, applied once per sweep rather than once per contingency.
- **The step-path gain grows sharply with grid size** — 3.3× on case14, 4.7× on case30, **28× on case89** — because the work removed is per-element, not a fixed per-call overhead. Profiling one ECML case89 step puts ~90 % of it inside `_get_original_bus`, which rescans the substation table with `DataFrame.iterrows` *once per bus* to build the observation's node mapping (476 buses × 38 substations). This version derives that mapping once and keys its cache on the pandapower bus-lookup content, rebuilding only on a real topology change.
- Parallel N-1 and `n-1-topk` exist only in this version and were left at their serial / full defaults, so both versions did the same amount of work.

Reading grid2op into it:

- **On the same solver stack, this version is in grid2op's class.** grid2op's default `PandaPowerBackend` steps in 15.3 / 15.7 / 18.6 ms against 19.5 / 18.4 / 22.9 ms here — 1.2–1.3× apart, on grids that carry no busbar-switch layer. Both are paying the same bill: ~98 % of the step is pandapower rebuilding and extracting the ppc around a ~170 µs solve. The ECML release, at 64–645 ms, is the outlier.
- **The whole gap is the backend, not the environment.** Swapping in grid2op's `LightSimBackend` — same grid2op env, same code path, C++ solver instead of pandapower — gives ~1 ms/step, **15–17×** its own pandapower backend and **18–21×** this version. That is the marshalling being deleted, not better numerics; the Newton solve is ~170 µs either way.
- **Our N-1 sweep already beats grid2op's on the shared stack.** Per contingency: 11.1 / 9.4 / 11.0 ms here vs. 15.0 / 15.5 / 18.8 ms for grid2op's pandapower backend (**1.3–1.7× in our favour**) — that is the parsed-options reuse in `run_nminus1_powerflow`, which grid2op does not do. Against grid2op's `LightSimBackend` we are 12–20× behind per contingency, and against its **batched** `ContingencyAnalysis` 110–515× (case14: 222.1 ms vs. 0.43 ms for the same 20 outages).
- **That path is now open for PPTopoGym too — `toolbox/ls2g_backend.py`.** grid2op models a substation as a per-element bus-assignment vector over 2 busbars, with no switch layer on any of these grids (`detailed_topo_desc` is absent), and lightsim2grid consumes that natively. This env expresses the same topology as pandapower switches, and lightsim2grid *ignores* switches — so handed the expanded net its `GridModel` reads case30's 93 buses as 93 separate, mostly injection-free nodes and returns an all-NaN sweep in 0.3 ms. The fix is to re-model rather than to swap a backend: build a **switch-free mirror net** (case30: 93 bus / 116 switch → **40 bus / 0 switch**) in which every element terminal attaches straight to a busbar bus, then push the live switch state each solve as a per-element **bus assignment** (`change_bus_powerline_or/ex`, `change_bus_trafo_hv/lv`, `change_bus_load/gen/sgen/shunt`; a closed bus coupler collapses both busbars onto busbar 0). Unused busbars must be `deactivate_bus`'d — an empty busbar is an injection-free node and the solve silently returns an empty vector.

  Measured on case30 (2026-09-01): over **80 converging actions**, split substations included, the worst `loading_percent` difference against the pandapower path is **1.9e-11 percentage points** with zero mismatches, and `LightsimBackend.solve` takes **0.625 ms** against `run_pf`'s **9.016 ms** — **14.4×**. It is opt-in and *not* bit-identical (a different solver), so it is gated on tolerance parity plus decision parity rather than on the golden record. Note this is the **power flow**, roughly half a step; the observation build is still pandapower, so it is not a 14× end-to-end step figure. Where that half-step itself goes — and the further **1.28×** since taken off it — is under **Inside the lightsim backend** above; the short version is that **54 % of this solve is the `res_*` write-back and only 12 % is the Newton solve**, so what is left between here and grid2op's ~1 ms step is still marshalling, just this package's own rather than pandapower's.
- **lightsim2grid is already the default here, and it is genuinely active** — there is nothing to turn on. `use_ls2g="auto"` is the default on every power-flow entry point (`run_powerflow`, `run_pf`, both N-1 backends), and after a reset `net._options["lightsim2grid"]` is `True` on case30 and case89. There is no `env_config` key for it: `run_pf(use_ls2g=...)` is a method argument and every internal call site takes the default. Note that `"auto"` falls back **silently** in two cases — DC power flow, and more than one in-service slack without `distributed_slack`.
- **Forcing the pandapower solver instead costs only 1.3–1.4× a step, which is the whole point.** case30: 8.25 → 11.31 ms per power flow (**1.37×**) and 18.4 → 24.6 ms per DoNothing step (**1.33×**); case89: 10.21 → 14.87 ms (**1.46×**) and 22.8 → 32.3 ms (**1.42×**) — medians of 3 alternating rounds, one interpreter per measurement. The solve is ~170 µs of an 18 ms step, so changing solvers can only ever move a sliver of it. grid2op's `LightSimBackend` is 18–21× faster because it removes the marshalling *around* the solve, not because it solves faster.
- **Evaluating contingencies through grid2op's public API costs more, not less.** A sweep of `obs.simulate()` — the call a grid2op agent actually makes — costs 552 / 3078 / 22721 ms on the pandapower backend and 33.7 / 118.1 / 570.4 ms on LightSim, i.e. **1.8–6.5×** the backend loop in the table, because each call re-primes a forecast backend. The public-API analogue here, `simulation()`, is one step plus a restore.
- **Build cost lands where the numba JIT does.** `grid2op.make()` takes 3.1–3.6 s on the pandapower backend and 0.57–1.1 s on LightSim; the ~1.85 s difference is the same one-time numba compilation of pandapower's kernels documented above, which a C++ backend never triggers. Our ~5 s cold build additionally scales the grid, creates the substations and verifies the actions.

**The two versions disagree about which busbar configurations survive.** On case30, actions 2, 3 and 4 converge on the ECML release (rewards 121.278 / 121.221 / 121.249) and return `worst_reward` here. That is a behavioural divergence, not a timing artefact — which is why the harness probes both versions and times the lowest action that converges on *both* with a matching reward (action 5 on case14 and case30, action 3 on case89; the rewards agree to within 1e-6, so both really are in the same electrical state). Unresolved.

Three constraints shape the harness, and they apply to any future version comparison:

- **The substation expansion cannot be shared.** The two versions write different `multi_bb_substation` schemas, so a net expanded by one cannot be loaded by the other. `build_grids.py` therefore ships the state *before* the double-busbar expansion — the scaled net as plain pandapower tables — and each version expands it with its own code. The harness then checks the two sides agree; on all three grids they matched exactly: 7 / 10 / 38 substations, 79 / 116 / 736 switches, 57 / 93 / 476 buses and 175 / 347 / 500 actions.
- **The timeseries has to survive two different ingestion routes.** The ECML release reads only `net.profiles` (Simbench per-unit shapes it scales itself) and raises without it, while this version also accepts pre-scaled tables via `env_config["profiles"]`. The base net therefore ships with its Simbench tables, and both sides are handed the same finished absolute tables afterwards — the ECML release's own scaling reproduced them exactly (max deviation 0.0 on all three grids), so nothing is lost by pinning them.
- **One variant per interpreter, alternating order** — the same trap as `ab_compare.py`.

`case14` needs one adjustment: pandapower ships it with 42 kA line ratings, so the base case sits at ~1.5 % loading and no load scaling congests it (`find_scaling_recursive` hits the recursion limit; `find_scaling_iterative` exhausts its 50 iterations). Each line is derated to its own base-case current at 60 % loading first. `case89`'s action space is capped at 500 entries — step cost depends on *which* action is applied, not how many exist, and the full grid generates over 150 000.

Two adaptations of the checked-in harness were needed for the ECML baseline and are **not** checked in: `build_grids.py` keeps `net.profiles` on the shipped net instead of deleting it, and the benchmark writes the shipped absolute tables into both envs after construction (the `env_config["profiles"]` route the checked-in version relies on does not exist in the ECML release). Convergence is also read off the step result rather than `net.converged`, which the ECML release never sets.

The grid2op figures were taken with a throwaway harness that is **not checked in** — grid2op is not a dependency of this package. Reproducing them needs `grid2op==1.12.5`, `pandapower==3.1.2` and `lightsim2grid==0.10.3` in a separate environment, the three datasets named in the table, and `Parameters(NO_OVERFLOW_DISCONNECTION=True, NB_TIMESTEP_COOLDOWN_SUB=0, NB_TIMESTEP_COOLDOWN_LINE=0)` so that no step is silently rejected or turned into a DoNothing by a cooldown.

## Reproducing the results
The figures and results in the paper have been made with the notebooks `getting_started.ipynb` and `evaluation.ipynb`. The agent evaluation is stored in `notebooks/plots`.

## Author Contributions

Dominik Köhler: Conceptualization (lead), Methodology (agents, evaluation, customization), Coding (implementation, maintenance, documentation, review, planning), Writing (original draft, review & editing)

Mohamed Hassouna: Conceptualization (co-lead), Methodology (Agent design, Observation), Coding (review, planning), Writing (review & editing)

Dmitry Degtyar: Methodology (GNN-agent), Coding (GNN agent, parallelization, maintenance, review), Writing (Sec. 4.5)

Kurt Brendlinger: Conceptualization (co-lead), Methodology (Sec. 4.1, support), Coding (implementation, maintenance, documentation), Writing (Sec. 4.1, review & editing)

Jonas Krauß: Conceptualization (Sec. 4.4), Coding (Scenarios, Observation), Writing (Sec. 4.4)

Christoph Scholz: Conceptualization (support), Writing (review), Supervision

## Acknowledgements

This work was supported by (i) Graph Neural Networks for Grid Control (GNN4GC) founded by the German Federal Ministry for Economic Affairs and Climate Action (BMWK) (03EI6117A), (ii) "Norddeutsches Reallabor," funded by BMWK (03EWR007N2), and (iii) the research group Reinforcement Learning for Cognitive Energy Systems (RL4CES) founded by the German Federal Ministry of Education and Research (01|S22063B).

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

## Cite this paper

Köhler, D., Hassouna, M., Degtyar, D., Krauß, J., Brendlinger, K., Scholz, C. (2026). PPTopoGym: Towards an RL Environment for Topology Actions on Power Grids. In: Koprinska, I., Mendes-Moreira, J., Branco, P. (eds) Machine Learning and Principles and Practice of Knowledge Discovery in Databases. ECML PKDD 2025. Communications in Computer and Information Science, vol 2841. Springer, Cham. https://doi.org/10.1007/978-3-032-19102-1_6

```bibtex
@inproceedings{kohler2026pptopogym,
  author    = {K{\"o}hler, Dominik and Hassouna, Mohamed and Degtyar, Dmitry and Krau{\ss}, Jonas and Brendlinger, Kurt and Scholz, Christoph},
  title     = {{PPTopoGym}: Towards an {RL} Environment for Topology Actions on Power Grids},
  booktitle = {Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  editor    = {Koprinska, Irena and Mendes-Moreira, Jo{\~a}o and Branco, Paula},
  series    = {Communications in Computer and Information Science},
  volume    = {2841},
  publisher = {Springer, Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-19102-1_6}
}
```

