# Feedback on `with-line-profiler` 0.8.2

Collected while replacing a hand-rolled in-repo line profiler with `with-line-profiler`
in PPTopoGym (`scripts/profile_env.py`). Environment: Python 3.12, Linux, single process,
no GPU. Ordered by how much time each cost me.

## 1. `accounting.start()` silently records nothing unless opted in

This is the big one. The obvious first program:

```python
from lineprofiler import accounting as acc
acc.start(run_dir="prof/run", role="x")
with acc.phase("work"):
    do_work()
acc.stop()
```

runs cleanly, prints nothing, exits 0 — and **writes no files at all**. `run_dir` is not
even created. Rendering then reports an empty run:

```
$ lineprofiler report prof/run --format text
Runtime 0ns   Processes 0   Roles none
```

The cause is that profiling is opt-in via `LINEPROFILER_PROFILE=1`; without it `enabled`
resolves to false and every call is a no-op. `start()`'s docstring does mention it
("Respects the same `LINEPROFILER_PROFILE`... defaults... a disabled profiler is a
near-free no-op"), but that reads as a note about *overhead*, not as "this is off by
default and your run will vanish". The fix on my side was `acc.start(..., enabled=True)`.

The design is right for library code left in place permanently. It is wrong for the
first-run experience of a profiling script, where the user has explicitly asked to
profile *right now*.

Suggestions, in order of preference:

- Make an **explicit** `run_dir=` argument imply intent: if the caller named a directory,
  record into it regardless of the env var. The env var then governs only the
  no-argument, leave-it-in-production case.
- Failing that, **warn once on `stop()`** when a profiler was installed, phases were
  entered, and nothing was written: `"lineprofiler: profiling disabled (set
  LINEPROFILER_PROFILE=1 or pass enabled=True); no data written to <run_dir>"`. A silent
  no-op that produces zero artifacts is the one case that should never be quiet.
- Have `lineprofiler report` on an empty/missing run directory say *why* it is empty
  rather than printing a well-formed report of nothing. `Runtime 0ns Processes 0` looks
  like a successful measurement of a fast program.

## 2. Distribution name and import name differ, with no alias

The package is `pip install with-line-profiler` but imports as `lineprofiler`. `import
with_line_profiler` fails. That is a legitimate choice, but it cost me a wrong first
guess and it is the kind of thing people hit before they reach the docs. Worth stating in
the first line of the README/PyPI description ("installs as `with-line-profiler`, imports
as `lineprofiler`"). A `with_line_profiler.py` shim that re-exports would remove the
problem entirely.

## 3. No `__main__`, so `python -m lineprofiler` fails

The console script `lineprofiler` works, but:

```
$ python -m lineprofiler report run/
No module named lineprofiler.__main__; 'lineprofiler' is a package and cannot be directly executed
```

`python -m` is the reliable way to hit the CLI from inside a script when you cannot
assume PATH (virtualenvs, `poetry run`, subprocess calls). I had to resolve the script
path manually:

```python
CLI = str(Path(sys.executable).parent / "lineprofiler")
```

Adding a three-line `__main__.py` that calls the existing CLI entry point would fix this.

## 4. Programmatic rendering is less discoverable than the CLI

`merge_run()` / `render()` / `report_as_dict()` exist and are exported, but the path from
"I have a `run_dir`" to "I have an HTML file on disk" is not obvious from
`dir(lineprofiler.accounting)` — `report` and `trace` are *modules*, so
`acc.report(...)` raises `TypeError: module is not callable`, which is a mildly
misleading dead end when you are guessing. I ended up shelling out to the CLI for all
four outputs because that was the documented route.

A small set of top-level helpers would cover the common case:

```python
acc.write_report(run_dir, path, format="html")
acc.write_trace(run_dir, path, format="html")
```

## 5. Smaller notes

- **`to_html` is on `LineProfiler`, but the accounting layer renders via the CLI.** Two
  different mechanisms for "give me an HTML file" is a papercut. Worth aligning.
- **The findings text over-reads a single-process run.** With one process and no
  concurrency, every phase is trivially "100% blocked ... nothing was being produced
  while this blocked — a stall rather than a queue". For a genuinely serial program that
  phrasing suggests a problem that does not exist. Consider suppressing wait/stall
  findings when the run has one lane, or wording them as "single process: no concurrency
  to overlap with".
- **`Source 403e0224 (+dirty: 9 files, diff sha f2f576)` in the report header is
  excellent** — recording the exact tree state next to the numbers is precisely what you
  want when a profile is committed to a repo. Please keep it.
- **The phase/percentage table is the most useful output by a distance.** For our case it
  immediately showed env construction at 73.8% of a short run, which is the actionable
  fact. The per-line view is a good complement but the phase table is what I put in the
  README.

## What worked well

- `project_folder=` scoping on `LineProfiler` keeps pandas/pandapower internals collapsed
  and the output readable, with no configuration beyond one path.
- `phase()` as a context manager nests naturally and needed no restructuring of the code
  under test.
- The resource block (peak RSS, growth rate attributed per phase, page-cache vs disk
  reads) gave us the memory numbers for free; I expected to have to measure those
  separately.
- Zero measurable effect on the code under test: profiling stayed entirely in the script,
  nothing had to be imported into the library.
