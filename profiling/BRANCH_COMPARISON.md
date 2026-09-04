# `develop` vs `develop_muzero` — step and N-1 speed

Measured 2026-08-26 on `node4`, pandapower 3.1.2, lightsim2grid backend, load average ~7.

Reproduce with `scripts/branch_compare/` (see "How this was measured" below). Raw medians are in
`profiling/branch_comparison_develop_vs_muzero.json`.

## Result

`develop_muzero` is **3.3–4.4× faster per step** and **2.1–2.4× faster on a full N-1 sweep**, and
the step-path advantage *grows* with grid size while the N-1 advantage is flat.

| grid | measurement | `develop` | `develop_muzero` | speed-up |
|---|---|---:|---:|---:|
| **case14** | DoNothing step | 69.1 ms | 19.9 ms | **3.47×** |
| 14 bus → 57 / 7 subs | switching step | 70.0 ms | 21.3 ms | **3.30×** |
| | N-1 sweep (20 contingencies) | 473.2 ms | 226.8 ms | **2.09×** |
| | N-1 DoNothing step | 974.7 ms | 454.5 ms | 2.14× |
| | reset | 43.2 ms | 11.2 ms | 3.87× |
| **case30** | DoNothing step | 75.2 ms | 18.4 ms | **4.10×** |
| 30 bus → 93 / 10 subs | switching step | 75.2 ms | 19.2 ms | **3.92×** |
| | N-1 sweep (41 contingencies) | 908.0 ms | 385.3 ms | **2.36×** |
| | N-1 DoNothing step | 1861.2 ms | 771.6 ms | 2.41× |
| | reset | 48.2 ms | 10.1 ms | 4.76× |
| **case89** | DoNothing step | 99.5 ms | 22.9 ms | **4.35×** |
| 89 bus → 476 / 38 subs | switching step | 100.0 ms | 24.4 ms | **4.11×** |
| | N-1 sweep (210 contingencies) | 5062.6 ms | 2346.1 ms | **2.16×** |
| | N-1 DoNothing step | 10182.0 ms | 4766.7 ms | 2.14× |
| | reset | 67.4 ms | 12.7 ms | 5.32× |

Reading the numbers:

- **A switching action costs the same as DoNothing on both branches** (case30: 75.2 vs 75.2 ms on
  `develop`, 19.2 vs 18.4 ms here). Applying the topology is ~1 ms of the step even after
  `load_action` was reduced to positional numpy writes; the step is a power flow plus an
  observation, and the switch writes disappear into it. The interesting axis is DoNothing vs N-1,
  not DoNothing vs switching.
- **An N-1 step costs two sweeps, not one.** `nminus1.donothing` is measured net of reset, and it
  still comes out at ≈ 2 × `nminus1.powerflow` on *both* branches (case30: 771.6 vs 385.3 here,
  1861.2 vs 908.0 on `develop`) — something in the step path solves the contingency set twice.
  Worth chasing; it is the same factor of 2 on both, so it is not a branch difference.
- **N-1 dominates everything else.** On `develop_muzero` an N-1 DoNothing step is **42×** a plain
  one on case30 and **208×** on case89 (on `develop`: 25× and 102×).
- **N-1 gains less than the step path** (2.1–2.4× vs 3.3–4.4×) because a contingency sweep is
  dominated by pandapower's per-contingency `runpp`, which neither branch has changed. What
  `develop_muzero` saves there is the parsed-options reuse in `run_nminus1_powerflow`, applied once
  per sweep rather than once per contingency.
- **The step-path gap widens with grid size** (3.5× → 4.1× → 4.4×), consistent with the wins being
  in per-element work — observation building, the topology fingerprint check, `load_action` —
  rather than in a fixed per-call overhead.

Not measured: parallel N-1 (`"n-1 parallel"`, `develop_muzero` only) and top-k contingency trimming
(`"n-1-topk"`, `develop_muzero` only) are both left at their serial/full defaults, so the N-1 rows
compare the same amount of work. Both would widen the gap further in the configurations that use
them.

## Caveat: the branches disagree about which busbar configurations are survivable

On case30, actions **2, 3 and 4** converge on `develop` (rewards 121.278 / 121.221 / 121.249) and
**fail to converge on `develop_muzero`** (`worst_reward`, `-1000`). Action 1 (`state '000000'`,
everything on busbar 0) is a no-op on `develop_muzero` — it writes no switches — while `develop`
opens 7 switches to reach the same state.

That is a behavioural divergence, not a timing artefact, and it is why the harness probes both
branches and times the lowest action that converges on *both* **with a matching reward** (action 5
on case14/case30, action 3 on case89 — the rewards agree to within 1e-6, so both branches really
are in the same electrical state). Timing an un-checked action index would have compared a real
busbar split against a crash-and-return-early path.

Worth a separate look — it is not obvious from here which branch is right.

## How this was measured

`scripts/branch_compare/`, three scripts:

1. **`build_grids.py`** — builds each grid once and pickles it. Deliberately ships the state
   *before* the double-busbar expansion: the scaled net, plain pandapower tables, plus the absolute
   timeseries. The two branches assemble configs differently (`find_scaling_iterative` +
   `apply_scaling_to_profiles` vs `find_scaling_recursive`; `verify_all_actions` only on
   `develop_muzero`), so calling each branch's own `config_caseXX()` would compare two different
   grids.
2. **`bench_branch.py`** — expands that base grid and times it, importing `pandapower_env` from
   whichever checkout is on `sys.path`.
3. **`run_comparison.py`** — probes both branches for a common action, then runs 3 rounds ×
   5 repetitions, **alternating** which branch goes first, each measurement in its **own
   interpreter**.

Three things this had to get right:

- **The substation expansion cannot be shared.** `develop`'s `double_busbar_substation` reads a
  `b0_1_switch` column that this branch does not write (`KeyError`), so a net built here cannot be
  loaded there. Each branch therefore expands the same base grid with its own code — which is
  itself part of what differs. The harness checks the two sides agree afterwards, and they did on
  all three grids: identical `n_substation`, `n_switch`, `n_bus_expanded`, `n_actions_used`.
- **The timeseries goes in via `env_config["profiles"]`**, the one ingestion route both branches
  implement identically (`develop` requires the key outright). The absolute tables are extracted
  once from a `develop_muzero` env built down the `net.profiles` Simbench route, so neither branch
  has to agree on how the per-unit → absolute scaling is computed.
- **One variant per interpreter, alternating.** Per `CLAUDE.md`: whichever side runs second inherits
  the caches the first filled (Simbench profiles, shared profile tables, parsed pandapower options),
  which fabricates speed-ups.

`case14` needed one adjustment: pandapower's `case14` ships 42 kA line ratings, putting the solved
base case at ~1.5% loading, so no amount of load scaling congests it — `find_scaling_recursive`
recurses to the limit and `find_scaling_iterative` walks all 50 iterations without an overload. Each
line is derated to its own base-case current at 60% loading first, which gives the 14-bus grid the
same "a few lines near their limit" character case30 and case89 have.

`case89`'s action space is capped at 500 entries. Step cost does not depend on how many actions
exist, only on which one is applied, but the full grid generates >150k actions and building that
DataFrame twice per run would dominate the benchmark without saying anything about the step path.
