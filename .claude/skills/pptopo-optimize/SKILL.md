---
name: pptopo-optimize
description: Apply and verify a performance optimization to PPTopoGym - env construction, step/observation path, N-1, or the greedy agents - under a strict correctness gate. Use after profiling has identified a hotspot, when asked to make the environment or agents faster, or to optimize states and actions that hurt performance.
---

# Optimizing PPTopoGym

One hotspot at a time. Each change must be independently measurable and independently
revertable — a batch of five changes that is collectively 8% faster tells you nothing
about which of the five to keep.

## The correctness bar

**Numerically equivalent within tolerance.** Optimizations may reorder or vectorize
arithmetic so results shift in the last floating-point bits. They may **not** change:

- which action is legal (`verify_action` / the `passes_*` rules)
- the profile-index trajectory (`step` / `simulation` / `end_simulation` semantics)
- observation *shapes* or dict keys
- reward values beyond float tolerance
- convergence outcomes — a case that converged must still converge

Assert this with an explicit parity test, not by eyeballing. Precedent to copy:
`tests/environments/test_load_action_parity.py` compares the fast positional path
against the label-based reference over the whole action space.

```python
np.testing.assert_allclose(fast_result, reference_result, rtol=1e-9, atol=1e-12)
```

If a change *cannot* hold this bar, it does not land as a default. Gate it behind an
opt-in config flag defaulting to off — the precedent is `static_obs_space`.

## Workflow

### 1. Write the parity test first

Before touching the implementation, add a test that pins current behaviour. It must fail
if the optimization changes results. Keep the slow path available as the reference where
practical, as `_load_action_by_label` does.

### 2. Make the change

Follow the house rules in `CLAUDE.md`:

- Functions over classes; decorators to extend.
- One function, one purpose; public functions above private helpers.
- Names carry the meaning — `filtered_actions`, not `fa`.
- Docstrings state what it does, params, returns, raises, and how it talks to the rest.
- Surgical: every changed line traces to the optimization. Do not tidy adjacent code.
- Add a `CLAUDE.md` gotcha entry when the change creates a non-obvious invariant a
  future reader could naively "fix" back (e.g. "don't restore the `is` check").

### 3. Verify — in this order

```bash
poetry run mypy                                   # fix mypy first
poetry run ruff check                             # then ruff
poetry run pytest -q                              # then the suite
```

Loop until mypy and ruff are both clean. Then measure, back to back:

```bash
poetry run python scripts/ab_compare.py --suite build --suite step --repeat 7
```

`ab_compare.py` stashes the working tree, benchmarks HEAD, restores, benchmarks the
change, and prints per-measurement deltas. **Never** compare against a number stored in
a file — load on this box moves timings several percent.

**Never time both variants in one process.** Whichever runs second inherits the first's
warmed caches and looks artificially fast. This produced a reported 9.8× "speedup" here
that was really 1.16× once each variant got a fresh interpreter. If you write an ad-hoc
probe, run one variant per process and alternate the order across repetitions.

**Say whether a cached win is cold or warm.** A process-local cache saves nothing on the
first call. If the real workload builds one env per process, quote the cold number as the
headline and the warm number as the multi-env benefit — do not merge them.

### 4. Accept or revert

Land only if **all** hold:

- mypy clean, ruff clean, full suite passes (no new failures vs. the baseline)
- the parity test passes
- `ab_compare` shows a win beyond the ±3% noise threshold on the targeted measurement
- no other measurement regressed beyond noise

Otherwise revert it. A change that is within noise is not a speedup; keeping it adds
risk and code for nothing. Record the negative result so it is not retried.

## Where the wins actually are

Measured on case30. Re-measure before trusting any of this.

**Construction (~5 s, the largest single target).**
`find_scaling_recursive` ≈ 47% and `get_first_sb_profiles` ≈ 27% of build time.
`find_scaling_recursive` runs one power flow per recursion and rescales full-length
profile frames each time; `get_first_sb_profiles` materializes the whole Simbench
profile set. Both are pure setup — a correct result cached or computed on fewer rows is
still correct. Watch out: the scaling trajectory must be preserved exactly, since it
determines the final net.

**Step path (~20 ms/step).** ~98% is pandapower marshalling around a ~170 µs solve.
The lever is **calling pandapower less**, not micro-optimizing around it. Env-owned
work (`load_action`, observation building) is already down to tens of µs.

**Greedy agents.** Cost = actions × per-action power flow. Levers, in order: prune the
action set before simulating; keep the per-process net cache hitting (one blob per
`act()`); avoid re-serializing the net per action.

**N-1.** Contingency count drives it. `n-1-topk` already cuts the contingency set;
parallel N-1 helps only in the main process.

## Traps

- `reset` deep-copies the net every call; profile tables are **shared, not copied** —
  never mutate `net.profiles` or a `df_profiles_*` in place, rebind instead.
- `recycle=True` silently reuses a stale Ybus. Wrong for a topology-switching env.
- Don't reintroduce a plain `pp.runpp`-per-call loop; warm options are the point.
- `run_nminus1_powerflow`'s warm-up must not set the warm-options marker — the
  contingency sweep mutates `in_service`.
- Action 0 is DoNothing and is special-cased in `load_action`.
- Nets not indexed `0..n-1` must keep falling back to the label path.
