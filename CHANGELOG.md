# Changelog

## 0.3.0

**Our method changed; grades under `grade_schema` 1.0.0 are not comparable to
earlier reports.** This release rewrites how the letter grade is computed. It
is not a tuning pass -- it is a different, more conservative definition of
"letter-eligible," implemented in response to two confirmed check-validity
bugs. If you have grades from an earlier version, re-grade before comparing;
a lower or higher number does not mean the underlying data changed.

### Grading model (`grade_schema` 1.0.0)

- Every check result now carries a `claim_type`: `DEFECT`, `RISK`,
  `CHARACTERISTIC`, `BY_DESIGN`, or `NOT_ASSESSED`. **Only `DEFECT` findings
  can move the letter grade.** Everything else is reported but never scored.
- The letter grade's aggregation changed from "start at 100, deduct fixed
  points per WARN/FAIL across every check" to **DEFECT-only defect density,
  worst-class-bounded**: for each DEFECT-class check, `defective findings /
  gradable findings`; the dataset's score is `100 * (1 - worst density)`
  across those check classes -- the worst class, never an average. See
  `grading.py`'s `integrity_score` for the exact function and worked
  examples.
- Idle/stall, LDJ smoothness, gripper chatter, joint-limit saturation, and
  action-state consistency (all of `episode_quality.py`) and cross-modal
  timestamp skew (`hygiene.py`) are retyped `RISK` and moved to a new
  `training_value` report section -- they were never sourced correctness
  claims, only proxy training-quality signals, and used to be scored as if
  they were.
- Topic frequency/drops, tf-tree integrity, and message-schema/dim
  consistency remain `DEFECT`-eligible -- they're objective, gate-surviving
  checks against the data's own declared or observed reference, not a
  convention.
- Calibration's own top-level verdict is unaffected; it was already excluded
  from the letter and stays that way.

### Fixed

- **The "vacuous A" gap: a dataset where no DEFECT-class check produced any
  gradable finding used to score a perfect 100/A.** Root cause: each
  DEFECT-eligible check abstains (`NOT_ASSESSED`) when its own precondition
  isn't met (no topics with enough messages to analyze a rate on, no tf
  concept, fewer than 2 episodes to compare schemas across) -- and the
  aggregation filtered `NOT_ASSESSED` findings out entirely rather than
  tracking "no check ran at all" as its own state, so `max(density,
  default=0.0)` silently read total silence as "nothing failed." A dataset
  with zero episodes, or one small/sparse enough that every DEFECT check
  individually abstains, now gets the letter `"NOT ASSESSED"` (score 0, not
  a graded F either -- an empty gradable set must never auto-pass *or*
  auto-fail) instead. A dataset where DEFECT checks actually ran and
  passed -- even on degenerate content -- is unaffected: real evidence
  (density 0.0 included) still earns its letter; only genuine silence
  triggers this. `--min-grade` now treats `NOT ASSESSED` as failing any
  requested bar, including `F`.
- **Content-plausibility flag now surfaces next to the grade, not just
  buried in the dataset-level checks table.** `plausibility.robot_data`
  firing (this doesn't look like robot-learning data) is an index-curation
  gate that must never silently coexist with a confident-looking letter --
  it's now a dedicated `content_plausibility` JSON field and a
  `CONTENT PLAUSIBILITY FLAG` line directly under `Overall grade` in the
  terminal report.
- **Per-check tables no longer render RISK-class findings identically to
  DEFECT ones.** A RISK FAIL/WARN sitting right under an `A` grade could read
  as if it had caused that grade. Every check row now carries a `Claim`
  column (`DEFECT`/`risk`/`characteristic`/`not assessed`/`by design`),
  styled distinctly from the severity column, with a footnote reiterating
  that only `DEFECT` moves the letter.
- `dataset_dim_stats`'s distinct-value tracking no longer truncates each
  episode's contribution to its lowest few unique values before unioning --
  it now unions each episode's full distinct-value set into the running
  tracker (still stopping once the cap is confirmed crossed), removing a
  truncation-order dependency that could otherwise hide part of a
  continuous joint's true range behind repeatedly-revisited low values.

- **`action_state_consistency` no longer hard-FAILs delta/velocity/torque/
  tokenized-action datasets.** Confirmed on a 50,415-episode BridgeData2 run:
  this check was FAILing ~99% of episodes because BridgeData2's actions are
  zero-centered deltas, not absolute state, and the check compared them
  unconditionally. It now only runs the element-wise comparison when a
  trajectory declares `action_space == "absolute"`; every other value
  (including the honest default of undeclared/`None`) returns
  `NOT_ASSESSED` naming the missing declaration instead of failing. A
  state/action dim-count mismatch is now `CHARACTERISTIC` ("architectural,
  not a defect") instead of an automatic FAIL.
- **`joint_limit_saturation` no longer uses each episode's own min/max as the
  limit proxy.** That made low-variance dims (a binary 0/1 gripper flag; any
  short episode that only visits a narrow slice of a joint's true range)
  read as "100% saturated" on every episode, by construction. The reference
  range is now computed once across the WHOLE dataset before per-episode
  grading starts, and binary/discrete dims (fewer than 5 distinct values
  dataset-wide) and near-constant dims are excluded outright. This check
  remains `RISK`-only: there is no declared-joint-limits field in this
  codebase, so it's always a proxy, never a `DEFECT`.

### Added

- `Trajectory.action_space: str | None` (`"absolute" | "delta" | "velocity" |
  "torque" | "tokenized" | None`) -- declared action-space semantics, gating
  `action_state_consistency`'s element-wise comparison. No ingest adapter in
  this release populates it (LeRobot's core spec has no such slot); it stays
  `None` (undeclared) until a format or dataset's own metadata states one.
- `--json` report: `grade_schema`, `defect_classes` (per-DEFECT-check-id
  density behind `overall`), `training_value` (per-RISK-check-id severity
  counts), `content_plausibility` (`{"flagged", "detail"}`). Report
  `schema_version` bumped 3 -> 5; nothing existing was removed or renamed.
  `overall.grade` can now also be the string `"NOT ASSESSED"`.
- Golden-corpus regression test (`tests/test_golden_corpus.py`): grades every
  checked-in fixture dataset and asserts the full expected outcome (grade,
  defect densities, per-check claim types), so an intentional check change
  must force a conscious edit to the expectations, not a silent drift.
- CI guard test asserting only `DEFECT`-claim findings can move
  `integrity_score`, introspecting the aggregation function directly with
  synthetic results of every claim type rather than depending on a real
  dataset happening to exercise the boundary.

## 0.2.2 and earlier

See git history.
