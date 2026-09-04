# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pandapower_env` (PPTopoGym) is a Gymnasium-compatible reinforcement-learning environment for **power-grid topology control**. Agents act on a [pandapower](https://pandapower.org) network — switching double-busbar substations, connecting/disconnecting lines, and changing phase-shift-transformer (PST) tap positions — to relieve line overloads. The package ships the environment, a set of benchmark agents, an RLlib GNN module, evaluation metrics, and example grid configs.

The package is published with Poetry; Python `^3.10` (developed against 3.10.11).

## Markdown style

- In `README.md`, write one line per paragraph (no manual line-wrapping within a paragraph or
  list item) — makes diffs reviewable line-by-line. Tables, code fences and headings are
  unaffected.

## Git rules

- **NEVER push to the `develop` branch on GitLab (`origin/develop`)** — not directly, not via a
  force push, not by pushing a local branch to it. Ever. Work on a feature branch and let a human
  open the merge request. Reading `develop` (checkout, `git worktree add`, `git show`) is fine.

## Cluster rules

- **NEVER run anything on the `standard` SLURM partition — only on `kes`.** Every `sbatch`/`srun`
  must carry `--partition=kes` (the scripts under `scripts/` already do). Never submit a job that
  would fall back to the cluster's default partition, and never change an existing
  `#SBATCH --partition=kes` line to anything else.

## Commands

```bash
poetry install                       # set up the environment (do everything via `poetry run`)
poetry run pytest                    # run all tests
poetry run pytest tests/agents/test_greedy_worker.py            # single file
poetry run pytest tests/agents/test_greedy_worker.py::test_name # single test
poetry run pytest --cov=pandapower_env --cov-fail-under=80      # with coverage (CI gate: 80%)
poetry run mypy                      # type check (config in pyproject.toml)
poetry run ruff check                # lint (ruff "select = ALL", line-length 120)
./test.sh                            # mypy + ruff + pytest, the full local gate
```

`poetry run pytest --nbmake notebooks --ignore=notebooks/ray_notebook.ipynb --ignore=notebooks/evaluation.ipynb` executes the demo notebooks as tests (CI does this). The notebooks under `notebooks/` are the primary usage docs and reproduce the paper's figures.

CI (`.gitlab-ci.yml`) runs lint (allowed to fail) → tests → build. `mypy` is configured with `platform = "linux"` even though development happens on Windows/macOS.

## Architecture

### Environment layering (`pandapower_env/environments/`)
- **`BaseEnvPP`** (`gym_env_pp.py`) — abstract `gym.Env`. Owns the pandapower net, the Simbench-style timeseries profiles, the power-flow runner, and the gym `step`/`reset` loop. Subclasses must implement `load_action`, `create_observation`, `calculate_reward`.
- **`PPTopoGym`** (`simulation_env.py`) — the concrete environment everything else uses. Adds the discrete action space (built from a `df_actions` DataFrame), the configurable observation space, custom/named reward functions, and crucially the **simulation API**: `start_simulation()` / `simulation(actions)` / `end_simulation()` let an agent apply a sequence of actions, read the outcome, and then restore the exact prior grid state (topology + profile index) — analogous to Grid2Op. State can also be serialized via `state_to_info` / `state_from_info` (must be restored on a *different* instance).
- **`multi_pp_env.py`** — helpers to wrap a `PPTopoGym` factory into vectorized (`AsyncVectorEnv`/`SyncVectorEnv`) envs.

Key env semantics: one timestep ≈ one episode by default; the env advances the profile index after computing reward (action stays in place for the *next* observation). Power flow failure → `worst_reward`, `terminated=True`, and an "empty" observation (`_handle_loadflow_failure`).

### Config-driven construction
`PPTopoGym(env_config: dict)` takes a **serializable** config (so it works under RLlib). Required keys: `net` (a `pandapowerNet` or path), `n_episodes`, `episode_length`, `action_space`. Optional: `reward` (a callable, or a string naming a function in `data/rewards.py`), `observation` (list of custom `{name, function, spaces}`), `observation_keys`, `nminus1`, `n-1 parallel`, `n-1 workers`, `n-1-topk` (percentage of lines evaluated as N-1 contingencies, default `100.0`), `pf_type` (`"ac"`/`"dc"`), `resolution`, `worst_reward`, `clip_max_loading`.
Build ready-made configs from **`data/example_configs.py`** (e.g. `config_case30()`) — these assemble a net, scale profiles until overloads occur, create double-busbar substations, and generate + verify the action list.

### Action space (`pandapower_env/action_space/`)
`create_actions_df(net, action_space)` turns an action spec into the `df_actions` DataFrame; row 0 is always **DoNothing**. Action columns drive `PPTopoGym.load_action`: `open_switches`/`closed_switches` (substation busbar config), `lines`/`disconnect_lines`, `trafos`/`tap_pos` (PST). `substation_action_rules.py` filters out unrealistic grid states via rules (`passes_two_bus_symmetry_rule`, `passes_islanded_elements_rule`, `passes_n_elements_rule`, `passes_fully_connected_grid_rule`). `verify_action` / `verify_all_actions` run a throwaway simulation to keep only legal actions.

### Substations (`pandapower_env/substation/`)
Models double-busbar (and 3-busbar-with-PST) substations on top of pandapower. Substation states are encoded as bitsets/hex strings; `multi_bb_substation` is a table added to the net. This is where the topology actions physically map to switch operations.

### Observations (`pandapower_env/observation_space/`)
`obs_space_utils.build_observation_registry()` defines the catalog of named observations (`ObservationConfig`, `ObsType` = PROFILE / AGGREGATE / CUSTOM / TABLE). `PPTopoGym` resolves these into a `spaces.Dict` and computes them via `_get_default_observation`. `pp_to_observation.py` + `toolbox/utils_graph_obs.py` (`PPObservation`) build graph/adjacency and aggregate observations efficiently (with a profile-table cache).

### Agents (`pandapower_env/agents/`)
- `base_agents.py` — `BaseAgent` (abstract `act`), `BaseGreedyAgent`.
- `benchmark_agents.py` — `DoNothingAgent`, `RandomAgent`, and greedy / greedy-rollout agents. Greedy agents simulate every legal action and score it with a feedback function; rollout variants additionally roll out future steps. `greedy_worker.py`'s `evaluate_action` is the unit parallelized with `joblib`.

### RLlib GNN (`pandapower_env/rlib_agents/`)
`gnn_agents.py` defines `GINETorchRLModule`, a `torch_geometric` GINE-based `TorchRLModule` for Ray RLlib (consumes the graph/adjacency observations). `callbacks.py` holds training callbacks.

### Metrics & toolbox
- `metrics/` — `evaluation_metrics.py` evaluates a sequence of actions on a grid (used for the paper's agent comparison).
- `toolbox/` — utilities: `utils.py` (`run_powerflow`, `run_nminus1_powerflow`, overload calculations — lightsim2grid is the default fast backend), `utils_scaling.py` (scale grid until N lines overload), `utils_profiles.py` (Simbench profile handling), plus graph/plotting helpers.

### Profiles
The net must carry `net.profiles`, a dict of Simbench-style DataFrames (`load`, `renewables`, `powerplants`, `gen_vm`, `sgen_q`) — one timeseries column per matching net element. `BaseEnvPP.setup_profiles()` validates lengths and pre-multiplies profiles by the base element values into the immutable `df_profiles_*` tables that drive each timestep.

## Common gotchas

- **`reset` semantics are not the gym defaults.** To reset to a specific timeseries timestep you pass `options={"index": N}`, **not** the `seed` arg — despite the base docstring claiming "seed: timeseries index". `seed` only feeds `random.seed()`. With no `options`, `reset` picks a *random* scenario start, so tests/repro need an explicit `index`.
- **`reset` deep-copies the whole net every call** (`net_copy_from`). It's correct but expensive; avoid resetting in hot loops.
- **`state_from_info` must run on a *different* env instance** than the one that produced the info dict — calling it on the same instance raises `ValueError` (guarded by `_source_instance_id`).
- **Simulations advance the profile index.** `simulation()`/`simulation_nminus1()` call `step()` internally, which increments `self.index`; `end_simulation()` restores topology *and* index. If you call `step` manually inside `start/end_simulation`, you own the restore.
- **Action 0 is always DoNothing** and is special-cased in `load_action` (it preserves the prior `converged` flag and returns early). Don't assume index 0 is a real topology change.
- **Action verification rules only support 2-busbar substations.** `verify_action` / the `passes_*` rules are not valid for 3+ busbar substations.
- **Power flow failure is non-fatal but lossy.** On non-convergence the env returns `worst_reward`, `terminated=True`, `info["crashed"]=True`, and an all-zeros `_empty_obs()` — not an exception. Check `info["powerflow_converged"]` rather than trusting the observation.
- **lightsim2grid is the default backend (`use_ls2g="auto"`)** and is silently skipped for DC power flow or when the net can't use it (only a warning is logged) — results then come from the slower native pandapower solver.
- **`nminus1=True` changes observation/PF shape:** `run_pf` then requires `max_loading_percent` in `res_line` and raises if absent. N-1 is much slower; most example configs default it off.
- **`config_case30()` and friends mutate and return a fresh net each call** (scaling, substation creation, action generation). They're not cheap — build the config once and reuse via `env.orig_config` (a deep copy) when an agent needs its own env.
- **Power-flow cost is pandapower's option-parsing, not the solve.** ~⅔ of each `pp.runpp` is `_init_runpp_options` / `_check_lightsim2grid_compatibility` (re-parsing options, ~16 `DataFrame.query` calls per step), not the numerics. Both `run_nminus1_powerflow` (per contingency) and `run_powerflow` (per call) therefore initialise options once and then reuse the low-level `pandapower.powerflow._powerflow`, which is ~2.5-2.9× faster and re-derives the ppc from the live net tables every call, so switches / `in_service` / tap changes are always picked up. Don't reintroduce a plain `pp.runpp`-per-call loop. `run_powerflow` tracks what its options were parsed for in `net["_ppenv_warm_options"]` = `(pf_type, use_ls2g, init_vm_pu)`; the marker is stored **on the net** so it shares the exact lifetime of `net._options` (both survive deepcopy/pickle, both are dropped by the `to_json` roundtrip that ships nets into greedy / N-1 child processes). `init_vm_pu` is part of the key because pandapower derives it from the in-service gen/ext_grid `vm_pu`, and freezing it would change the Newton-Raphson starting point — including it keeps results bit-for-bit identical for ~120 µs. Don't let `run_nminus1_powerflow`'s warm-up set the marker: `run_contingency` mutates `in_service` during the sweep.
- **lightsim2grid ignores pandapower switches -- but a switch-free *mirror net* unblocks it (`toolbox/ls2g_backend.py`).** Handed the expanded net, `GridModel.ac_pf` returns an empty array and `ContingencyAnalysisCPP` an all-NaN matrix in 0.3 ms: a fast *wrong* answer (63 of case30's 93 buses are auxiliary busbar buses reachable only through switches, so ls2g sees injection-free isolated nodes). The fix is to re-express the topology the way grid2op does: drop every auxiliary bus and substation switch (case30: 93 bus/116 switch -> **40 bus/0 switch**), wire elements straight onto busbar buses, then per solve push the live switch state as a **per-element bus assignment** (`change_bus_powerline_or/ex`, `change_bus_trafo_hv/lv`, `change_bus_load/gen/sgen/shunt`; a closed `b01_switch` coupler collapses both busbars onto busbar 0). **Unused busbars must be `deactivate_bus`'d** -- an empty busbar is an injection-free node and the solve silently returns an empty vector. Measured on case30: 80 converging actions, worst `loading_percent` delta **1.9e-11**, `solve` **0.625 ms** vs `run_pf` **9.016 ms** (**14.4x**). Opt-in and NOT bit-identical, so it is gated on tolerance + decision parity, not the golden record. Note `get_line*_res_full()` returns current in **kA already** -- dividing by 1000 gives ~117 pp of `loading_percent` error while `p_from_mw` still matches to 7e-13.
- **N-1 runs on the same `backend="lightsim"` switch, and it needs the pandapower fallback for islanding contingencies.** `LightsimBackend.solve_nminus1` takes each line/trafo out of service on the live net, re-solves through `solve` (so a contingency sees the same mirror-net translation, split substations included) and reduces the results into the columns `pandapower.contingency.run_contingency` writes — `res_line`/`res_trafo` `max/min_loading_percent` + `cause_element`/`cause_index`/`causes_overloading`, `res_bus` `max/min_vm_pu` — over the N-1 cases only, ending on one more N-0 solve. `run_pf` dispatches on the backend **before** `"n-1 parallel"`, deliberately: the lightsim sweep is already single-process. Measured (2026-09-01, 3 rounds × 5 reps, isolated interpreters): bare sweep case30 **451 → 88 ms (5.1×)**, case89 **2792 → 913 ms (3.1×)**; parity to ~1e-11 on every aggregate with an identical NaN pattern. Three traps: (1) lightsim2grid has **one slack**, so an outage that islands part of the grid returns an *empty* voltage vector while pandapower promotes a generator in the island and answers — dropping those cases silently under-reports the risk (case30: `max_loading_percent` off by 0.3 pp, `min_loading_percent` by 69 pp), so `_solve_with_pandapower` re-solves exactly them — **6 of 41 contingencies on case30, 56 of 210 on case89**, and since each costs a full `run_powerflow` (~9-10 ms) they are most of what is left in the sweep, which is why case89 gains less than case30. Handling islands natively is the next win here, not a faster solve; (2) `trafo3w` contingencies cannot be modelled, so the sweep **raises** rather than quietly skipping them; (3) trafo `in_service` must be pushed to the model (`deactivate_trafo`) or every transformer contingency is a no-op that looks like a healthy grid — case30 has no trafos, so only case14/case89 catch it.
- **An out-of-service line's `loading_percent` is `0.0` in pandapower, not NaN — and the ls2g backend must match.** `greedy_worker` reads `res_line["loading_percent"].to_numpy(dtype=float, na_value=1000.0)`, so a NaN there scores every line-disconnection action as a catastrophe. The same applies to `res_trafo`. This only shows up on actions that open a line (or on an N-1 sweep, where every contingency creates one), which is why the N-0 parity tests missed it: `np.nanmax(|a - b|)` ignores a value that is NaN on one side only.
- **In the ls2g backend, the cost is per *call*, not per element — `pd.DataFrame()` is the worst offender.** The no-argument constructor takes **~132 µs**, more than twice a real 41×10 frame (~57 µs), so a net with no transformers and no static generators used to spend ~264 µs per solve publishing two *empty* tables — 18% of the whole solve. `_empty_result_frame` copies a module prototype instead (~8.5 µs, **15×**), and it must keep returning a **fresh object per solve**: the frames are handed to callers who may hold last step's table, and `.copy()` is what preserves that while the shared prototype would not. Conversely, don't "modernise" the per-element push loops (`_push_topology`, `_sync_active_buses`, `_push_injections`) into numpy: it was tried and is **slower** (`_push_topology` 114 → 173 µs), because the 109 element terminals spread over 5 table columns on case30 are far below the size where numpy's ~1-2 µs per-operation overhead pays for itself. `PerElementPushOracle` in `tests/toolbox/test_ls2g_backend.py` keeps the per-element loops as the oracle for anyone who tries again. Related: lightsim2grid's vectorized `update_loads_p` / `update_gens_p` API (what grid2op's own backend uses) is ~9× faster per call but takes **float32**, while `change_p_load` takes a double and this project's injections carry precision past float32 (round-trip delta 1.06e-7 MW) — adopting it silently truncates every setpoint and must be tolerance-gated, not treated as an equivalent rewrite.
- **Newer pandapower is *slower* here, and N-1 is no exception — the project is deliberately pinned to `>=3.1.2,<3.2`.** Measured on case30 (2026-08-25), same machine, same code: serial N-1 **390 ms (3.1.2) → 419 ms (3.2.0) → 483 ms (3.4.0)**; `run_pf` **8.4 ms → 9.0 ms → 10.6 ms**; random-step **18.7 → 19.9 → 23.2 ms/step**. The regression is monotonic and lives in `_pd2ppc` (2.86 → 4.24 ms). `contingency.py` itself is **documentation-only** between 3.1.2 and 3.4 — there is no new fast contingency path to adopt, and the only new `runpp` option is `enforce_p_lims` (a modelling feature, not a speed knob). So the pin is a ~20-25% across-the-board win, not conservatism: **don't "modernise" it without re-running the benchmark.** (3.4 was tried and passes the suite — 192 passed / 1 pre-existing flaky perf test — so upgrading is safe if a future 3.x feature is ever needed; it just costs speed. 3.4 additionally pulls in `pandera`/`typeguard`/`typing-inspect`, which 3.1.2 does not need.)
- **The observation topology cache keys on lookup *content*, not object identity.** pandapower allocates a fresh `net._pd2ppc_lookups` (and a new `["bus"]` array) on **every** power flow, while the content only changes when the topology really does. `utils_graph_obs._topology_fingerprint_changed` therefore compares `_pd2ppc_lookups["bus"]` against a copy held in the cache (~3 µs) instead of comparing identity, which used to throw the whole node mapping away and rebuild it (~1.5 ms) once per step for nothing. Don't "restore" the `is` check — and note the `*_lookups_ref` cache keys now hold a **copy of the bus array**, not the lookups object (`lookups_ref` / `lookup_obj_id` remain, for the sub-caches and observability, and do not decide validity).
- **Double-busbar substations add many out-of-service auxiliary buses.** e.g. case30 has 93 `net.bus` rows but only ~30 active — relevant whenever code assumes `len(net.bus)` equals the solved system size.
- **The bus-switch-tree traversal is one union-find pass, not a per-bus search.** `toolbox/topology_helpers.bus_switch_components` unions the endpoints of the bus-bus switches once; `find_bus_switch_tree` / `find_bus_switch_trees_from_list` are thin lookups into it. It replaced a fixpoint loop that re-sliced `net.switch` and ran `DataFrame.isin` over a growing set *once per queried bus* — case118 all-buses **209.6 ms → 0.81 ms (260×)**. The contract is unchanged and load-bearing for substation creation: the requested bus stays **first** in its tree (it becomes bus0), `include_single_buses` still yields `[ibus]` for untouched buses, and `fail_on_overlap=True` still raises `ValueError`. Don't reintroduce a per-bus loop. `tests/toolbox/test_topology_helpers.py` keeps the old implementation as an oracle and compares element-for-element over randomized switch states on case30 + case118.
- **`harmonize_gen_voltage_setpoints` is ~free because no production grid needs it.** `_pcc_generator_groups` first asks `_pcc_is_possible`: if no two generators share a bus, and none land in the same component when *every* bus-bus switch is treated as closed, then no switch state can ever group them and the answer is permanently `[]`. **case30 and case118 both return zero PCC groups**, so the whole call costs ~1 µs instead of 158 ms/step on case118 (**705×**). Its cache is keyed on the **bus-bus switch subset only** — element-switch writes cannot regroup generators, so they must not invalidate it — and the mask is re-derived if `len(net.switch)` ever changes. Because both production grids exercise only the early-out, the *grouping* path is guarded by a synthetic `shared_pcc_net` fixture in `tests/toolbox/test_pandapower_tools.py`; a bug there would silently change physics on any future grid whose generators share a busbar, not raise. Note this function is currently **called from `run_pf` only on `merge_muzero`**, not on `develop_muzero`.
- **Profile tables are shared between environments, not copied.** They dominate an env's memory (~53 of ~54 MB per case30 env before this change: once in the live net, once in `net_copy_from`, once in `_orig_config`, plus the derived `df_profiles_*`). `gym_env_pp.deepcopy_net_sharing_profiles` deep-copies a net but aliases `net.profiles`, and `_SHARED_PROFILE_TABLES` (a `WeakValueDictionary` keyed on the profile object ids + the base element values) lets envs with identical inputs share the derived `df_profiles_*` tables too. Per-env cost is now **~0.7 MB (73× less)**, which is what makes the vectorized / multi-env setting affordable. This is sound only because both are **read-only after `setup_profiles`** — never mutate `net.profiles` or a `df_profiles_*` in place; rebind (`net.profiles = {...}`) instead. The cache is per-process, so spawned workers start empty.
- **An env takes its timeseries from `env_config["profiles"]` when that key is present, otherwise from `net.profiles` — and the config values are assigned, never multiplied.** `net.profiles` is the Simbench route: per-unit shapes that `setup_profiles` scales by the net's base `p_mw`/`q_mvar`/`vm_pu`. `env_config["profiles"]` is the `{element: {variable: DataFrame}}` route (`load`/`gen`/`sgen` × `p_mw`/`q_mvar`/`vm_pu` only, one column per element, absolute values already scaled) that `setup_profiles_from_config` copies straight into the same six `df_profiles_*` tables. Reusing `setup_profiles`' `@ np.diag(...)` there would **double-scale** and silently hand the agent a different grid; `tests/environments/test_config_profiles.py` compares the two routes table-for-table to catch exactly that. The config path exists because the frozen-episode MuZero datasets repeat one profile row for a whole episode and delete `net.profiles` to keep the per-actor pickle small — a config that was ignored would run *time-varying* episodes and score them against constants, with no exception raised. An `(element, variable)` pair outside those six raises: varying arbitrary net columns is a UCTE data-model feature that was deliberately **not** migrated from `merge_muzero`. Sharing (`_SHARED_PROFILE_TABLES`, `_copy_config_sharing_profiles`) covers both routes under disjoint key namespaces.
- **`greedy_worker.evaluate_action` takes the pre-action state two ways, and `supported_action_types` is narrowed per net.** `base_topology` + `profile_slice` are the packed numpy arrays the greedy agents build (injections travel separately, so one static blob serves every timestep); `grid_snapshot` is the `{element: DataFrame}` slice a caller takes off the live net with `PPTopoGym.supported_action_types`. The latter restores **topology only** — the vendored `snapshot_grid_state` tests each profile *variable* name with `v in net`, which is never true for a pandapower net, so `load`/`gen` come back with an empty column list; a caller using it must re-dump the blob per episode or it scores every later episode against the first one's injections. `supported_action_types` is built by `_supported_action_types(net)`, which drops columns the net lacks — the double-busbar tables this branch builds have **no `state` column** (that is a UCTE-tooling column), and naming it would make every snapshot raise `KeyError`. Nothing is lost: the substation configuration is fully described by `switch.closed`.
- **`current_step` and `episode_step_counter` are not the same counter.** `step()` raises `current_step` *before* applying the action, so a reward function sees `k+1` on the k-th step; `episode_step_counter` only advances *after* the reward has been taken, so it is `k` there and reads as "steps completed this episode". Reward functions gate their episode-boundary carry-over on `episode_step_counter == 0` — aliasing it to `current_step` would make that test never fire on the first step. Both reset to 0 in `reset()`; neither is restored by `restore_state` / `end_simulation` (same as `merge_muzero`, which this matches deliberately).
- **The Simbench profile library is cached per process, and `deterministic_profiles` must keep copying its slices.** `simbench.get_all_simbench_profiles(sb_index)` re-reads several CSVs (~35136 × 618 values) on **every** call, costs ~1.2 s, and is a pure function of the scenario index — it is not cached upstream. `utils_profiles._SIMBENCH_PROFILE_CACHE` memoises it per process (1259 ms → 5.7 ms, 221×), which makes every env after the first in a process ~2× cheaper to build (case30 second build 3.0-3.3 s → ~1.5 s). The `.copy()` on each `iloc` slice in `deterministic_profiles` is **load-bearing, not defensive style**: the frames come from the shared cache, so returning views would let one net's in-place edit silently corrupt every net built afterwards in that process (`tests/toolbox/test_simbench_profile_cache.py` pins this). Copying the ~30-column slice costs ~2 ms. The cache is per-process, so spawned workers start empty and the *first* build in a process is unaffected.
- **`utils_scaling.run_pf` must stay on the warm `run_powerflow`, not `pp.runpp`.** `find_scaling_recursive` runs one power flow per recursion (22 on case30, ~98% of its runtime), and each plain `pp.runpp` re-parses pandapower's options from scratch. Routing it through `toolbox.utils.run_powerflow` reuses the parsed options for ~1.16× on the scaling search (~2345 → ~2022 ms) with **bit-identical** results — the scaling trajectory and final net match to 10 decimal places (`tests/toolbox/test_scaling_warm_pf_parity.py`). This is the same "don't reintroduce a plain `pp.runpp`-per-call loop" rule as above; the scaling path had simply been missed.
- **Never A/B two performance variants in the same process.** Whichever runs second inherits the caches the first filled (pandapower internals, imports, the Simbench cache), which fabricates speedups: the `run_pf` swap above first measured **9.83×** this way when the true figure is **1.16×**. Run one variant per interpreter and alternate the order (`scripts/ab_compare.py` does this via subprocesses; ad-hoc probes are where the trap bites). Related: a build script that constructs a single env cannot show a per-process cache win at all — quote cold and warm numbers separately.
- **`load_action` applies precomputed positional writes.** `PPTopoGym._build_action_plans` resolves every action's switch/line/trafo *labels* to row positions once at init, so `load_action` is numpy fancy-index assignment (~460 µs → ~10 µs on case30) instead of three `DataFrame.loc` label writes. The column arrays must be **re-fetched on every call** — `reset` / `restore_topology` reallocate them, so a cached view would write into a detached array. Nets whose tables are not indexed `0..n-1` fall back to `_load_action_by_label`, which is kept as the correctness reference (`tests/environments/test_load_action_parity.py` compares both paths over the whole action space).
- **The observation space is a lie whenever a substation splits — unless `static_obs_space` is on.** Node-aggregated observations have length `n_nodes`, but `n_nodes` *grows as substations split* (case30: 30 at reset, up to 35 over action sequences), while `define_observation_space` froze the shape at the reset topology. ~53/60 observations then fail `observation_space.contains(obs)`, and `SyncVectorEnv`/`AsyncVectorEnv` crash with `could not broadcast input array from shape (31,) into shape (30,)`. Setting `env_config["static_obs_space"] = True` declares node observations at their static upper bound (`_compute_max_n_nodes` = reset nodes + one per extra busbar; case30: 40) and zero-pads them up to it → 0/60 violations and the vector envs run. It is **opt-in and off by default**, because turning it on changes observation shapes and so would break consumers' network input dimensions (MuZero). Rewards are identical in both modes. The two `xfail`s in `tests/environments/test_observation_space_contract.py` record the default's brokenness and flip to XPASS if the default ever changes.
- **`pandapower` 3.4.0 is ~28% slower than 3.1.2 on this workload.** `run_pf` on case30 goes 8.4 ms → 10.6 ms, entirely inside `_pd2ppc` (2.86 ms → 4.24 ms); the lightsim2grid solve is unchanged at ~170 µs in both. The env-owned cost is not the issue: **~98% of a step is pandapower marshalling around a 170 µs solve** (`_pd2ppc` + `_extract_results` + ppc→ppci reduction). `recycle` would skip that rebuild but is **wrong here** — it silently reuses the old Ybus, so topology switching (the whole point of this env) either gives stale results or fails to converge.
- **Parallel N-1 (`"n-1 parallel"` config) auto-degrades to serial inside child processes.** `toolbox/nminus1_parallel.run_nminus1_powerflow_parallel` splits the contingencies across loky workers (bit-for-bit identical to serial `run_nminus1_powerflow`, which is kept untouched). It falls back to serial when `multiprocessing.parent_process() is not None` or `n_workers <= 1`, so it never nests pools — meaning the flag is a **no-op (serial) inside spawn MuZero workers and the parallel greedy agent**, and only speeds up the main process. Speedup scales with contingency count, not CPUs: useful workers ≈ contingencies / ~6 (case30 ~3.4×, case89 ~8.5×); over-provisioning slightly *hurts*. `"n-1 workers"` defaults to all CPUs — cap it for small grids.
- **`"n-1-topk"` (default `100.0`) caps which *lines* are switched off as N-1 contingencies.** Below 100, only the top `ceil(topk% * n_lines)` lines by N-0 apparent power flow (`S = √(P²+Q²)` MVA, max of from/to ends) are evaluated — trafo/trafo3w contingencies are untouched. The filter cuts the *contingency* set, not the *monitored* set: `res_line.max_loading_percent` is still produced for every line, just over fewer outages, so it can only *decrease* vs. full N-1. Selection lives in `toolbox/utils.select_topk_line_contingencies` (run after the warm base power flow) and is applied identically by the serial and parallel backends, so the parallel result stays bit-for-bit equal to serial.
- **`end_simulation` must not go back through `reset()`.** It restores state with `PPTopoGym._reset_state` (the state half of `reset`) and then replays the action log. Calling the public `reset()` there — as it used to — runs a power flow *and* builds a full observation at the pristine topology, both of which are thrown away one line later when the log is replayed and the grid re-solved. That was 1 of the 4 power flows every `simulation([a])` did: `simulation([1])` 37.0 → 27.5 ms, `verify_action(1)` 20.4 → 11.2 ms. `_reset_state` deliberately still zeroes `episode_step_counter`, matching what `reset` did, so the documented "not restored by `end_simulation`" behaviour is unchanged.
- **`run_pf` returns early when the results on the net already answer the request, and `net.converged` is what makes that safe.** The flag is the "results are current" marker: *every* path that changes what a power flow would produce clears it to `None` — `load_action` (via `BaseEnvPP.load_action`), `load_profile_timestep_into_net`, `_restore_baseline_net`, and `_reset_state`. The single exception is **DoNothing, which deliberately preserves the flag**, so a DoNothing step used to re-solve a grid whose topology, injections and profile index were all unchanged — one wasted full power flow on every such step (30% of them under a typical agent). The guard also compares `net["_ppenv_solved_for"]` = `(pf_type, use_ls2g, nminus1)`, because `pf_type` genuinely differs between callers (`reset` passes `self.pf_type`, `step`/`create_observation` take the default `"ac"`) and an N-1 sweep writes columns a plain solve does not; without that key a DC env would silently be served an AC result. The marker is stored **on the net**, the same reasoning as `_ppenv_warm_options`: it then shares the exact lifetime of the `converged` flag it qualifies. Measured on the 30/70 action mix: case30 step 22.6 → 19.9 ms (pandapower backend), and the golden record is bit-identical, because re-solving an unchanged net is deterministic (both solvers flat-start). **If you add a path that mutates the net, it must clear `net.converged`** — that, not the marker, is the invariant this rests on. Note `LightsimBackend.solve` sets `converged = True` without the marker, so a direct backend call is conservatively re-solved by the next `run_pf`.
- **`_grid_is_disconnected` reads `res_bus` positionally, and the ls2g backend builds its result tables from one 2-D matrix.** Both are per-solve overheads that pandas charges in full even when nothing needs the general path. The islanding check's `res_bus["vm_pu"].reindex(in_service_buses)` costs ~151 µs against ~10 µs for a numpy mask, because pandapower writes `res_bus` with `net.bus`'s own index and the reindex is a no-op every time; `_grid_is_disconnected_by_label` is kept as the fallback for a non-aligned net *and* as the oracle `tests/environments/test_disconnect_parity.py` compares against — don't delete it, and don't "simplify" the fast path back to it. In `ls2g_backend`, `_result_frame` wraps a `column_stack`ed float64 matrix instead of a per-column dict (`pd.DataFrame({...})` sanitizes and boxes each column separately: ~100 → ~58 µs for `res_line`, ~130 µs per solve across the six tables); it still returns a **fresh frame per solve**, matching pandapower, so nothing downstream can alias a previous step's results. `_resolve_bus_positions` derives the bus→mirror mapping once for `_write_bus_lookup` and `_write_bus_results`, which were computing it twice. The vectorized lookup must keep numbering isolated buses **consecutively in bus order** — `tests/toolbox/test_ls2g_backend.py` pins it against the per-bus loop it replaced. All of this is bit-identical, verified over 457 arrays (result tables, rewards, bus lookups, N-1 aggregates).
- **`total_energy_overload` and `max_loading_percent` are info-only observations and must stay out of `build_observation_registry`.** An env defaults `observation_keys` to *every* key of that registry, so putting them there grows the observation space and changes the input dimension of any trained network (MuZero). They live in `build_info_observation_registry()`, which `PPTopoGym` merges only into `_computable_obs_configs` — the dict used to *compute* an explicitly requested key. `define_observation_space` stays on `active_obs_configs`. They had no entry at all before, so `create_observation` silently dropped them from `info` and `overload_energy_difference_abs_mvah` / `loading_improvement_optimization` returned NaN for every step ever evaluated.
- **`reward_better_than_donothing` needs its recursion guard, and must use `save_state`/`restore_state` — not `start/end_simulation`.** Its DoNothing rollout drives `env.step()`, which calls the reward again; without the `env.cache["_donothing_rollout_active"]` flag that recursion never ends (it used to `RecursionError` on the first step, and also unpacked 4 values from `step`'s 5-tuple). The state round trip cannot use `end_simulation` because the reward runs from *inside* `step()`, before the scored action has been appended to `log_actions` — the action-log replay would restore the grid to the state *before* that action and the caller's observation would describe the wrong topology. `save_state` captures the live topology instead. `episode_step_counter` is saved and restored by hand because `restore_state` deliberately does not touch it.
- **Overload ratings include `df` and `parallel`.** `toolbox/utils.total_active_overload_mva` divides by `max_i_ka * df * parallel` (lines) and `sn_mva * parallel * df` (trafos) — the same denominators pandapower uses for `loading_percent`. It used to omit both, under-reporting a derated line and over-reporting a double circuit; case30 has `df=1, parallel=1` everywhere, which is why it went unnoticed. The bus voltage lookup goes through `_bus_vn_kv`, which takes positionally when `net.bus` is indexed `0..n-1` (~11 µs guard vs a ~221 µs `.loc` reindex) — this runs twice per step.
- **A crashed step leaves the net desynced from `log_actions`.** `PPTopoGym.step` returns early on non-convergence *without* appending the action, so the grid carries an action the log does not, and `end_simulation` / `state_from_info` then restore a different topology. Harmless while `terminated=True` ends the episode, but do not build resume-from-crash logic on the action log. On the `simenv` test fixture, actions 1, 2 and 4 are the non-converging ones.


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## 5. Test everything

- If you need to fix a bug, then write a test, reproduce the bug, and make the test (and the code) run.
- If you add new features or lines of code, which are not covered by a current test, then extend a test or write a new one.

## 6. Coding Standards in the team

- Simplicity driven: If a code is complicated, but could be rewritten simpler, do that.
- Function-driven: If you write a class to abstract things, which could have gone into functions, use functions. Extend with decorators, if needed.
- One function, one purpose: if things in a function do not follow the general purpose of the function, try to make them external functions.
- Readable naming: Make the code readable from variable and function names, without needing to read to doc-strings. A junior software engineer should be able to read and understand the code quite fast.
- Docstrings: Include docstrings explaining what a function does, what its input is, and what the output is; and possible exceptions that are thrown. Also give an intuitive understanding, and write, how the function communicates with the rest of the code.
- Ruff+mypy: Ensure, ruff+mypy is running after your changes. First check+fix mypy, then ruff, then loop until no mypy or ruff errors occur anymore.

## 7. Readability — Structure Over Comments

**Code should read top-down like a narrative. Util files are for shared plumbing.**

- Public functions at the top, private helpers below. Reader sees intent before implementation.
- Group related logic into clearly named functions — prefer readable call chains over inline blocks.
- Extract pure utility logic (math helpers, string formatting, generic transforms) into `*_utils.py` files alongside the module that uses them.
- Don't create a god-object `utils.py` — scope utils to their domain (e.g. `mcts_utils.py`, `network_utils.py`).
- Naming: variables and functions should make comments unnecessary. `filtered_actions` > `fa`. `compute_td_target` > `calc`.
- Blank lines separate logical blocks within a function — treat them like paragraph breaks.

The test: A new team member should understand the module's flow by reading function names alone.

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 8. Lazy development

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size, lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung, a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs (the platform is never the spec ideal, a clock drifts, a sensor reads off), anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.
