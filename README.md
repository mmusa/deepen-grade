# deepen-grade

A free, local, `pip install`-able CLI that answers **"is this robot dataset
training-grade?"** in one graded report.

`deepen grade` reads `.mcap` (ROS 2), `.bag` (ROS 1), `.db3` (ROS 2 sqlite), or
a [LeRobot](https://huggingface.co/docs/lerobot) dataset (local path or a
HuggingFace `repo-id`), and runs three layers of checks -- **hygiene**,
**episode quality**, and **calibration sanity** -- entirely on your machine.
Nothing uploads by default. Every number in the report is traced to a named,
public source: an open standard, an ROS REP, or a peer-reviewed paper.

```
$ deepen grade my_episode.mcap

╭─ deepen-grade report ──────────────────────────────╮
│ my_episode.mcap                                    │
│ format: mcap   episodes: 1                          │
╰──────────────────────────────────────────────────────╯
Overall grade: A (100/100)   grade_schema 1.0.0 -- DEFECT-class findings only, see training_value below
Calibration:   NOT ASSESSED
               no calibration metadata found anywhere in this dataset -- calibration
               quality is unknown, not good; verification requires the deep audit

         Per-episode grades
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Episode     ┃ Grade ┃ Score ┃ Task ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━┩
│ my_episode  │ A     │ 100   │ -    │
└─────────────┴───────┴───────┴──────┘

           Checks -- my_episode
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┳...
│ Check                   ┃ Res. ┃ Claim        ┃ Summary
│ Topic frequency & drops │ PASS │ DEFECT       │ 4 topics, no drops detected
│ Cross-modal sync skew   │ WARN │ risk         │ max skew 28.4ms exceeds 15ms budget
│ tf-tree integrity       │ PASS │ DEFECT       │ 6 frames, single-parent tree
│ Idle / stall detection  │ PASS │ risk         │ 3.1% of episode idle
│ LDJ smoothness          │ PASS │ risk         │ LDJ = -14.2
│ Joint-limit saturation  │ PASS │ risk         │ worst dim 'joint_3' saturated 4.2%
│ Gripper chatter         │ PASS │ risk         │ 2 direction reversals (0.31 Hz)
│ Action-state consistency│ N/A  │ not assessed │ action-space semantics undeclared
│ Calibration sanity      │ INFO │ not assessed │ NOT ASSESSED -- no camera calibration metadata found
└──────────────────────────────────────────────────────┘
   Claim column: only DEFECT rows can move "Overall grade" above -- risk/
   characteristic/not assessed/by design never do (see training_value).

    Training-value profile (RISK -- never graded)
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━┓
┃ Check                   ┃ Pass ┃ Warn ┃ Fail ┃ N/A ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━┩
│ Cross-modal sync skew   │ 0    │ 1    │ 0    │ 0   │
│ Idle / stall detection  │ 1    │ 0    │ 0    │ 0   │
│ LDJ smoothness          │ 1    │ 0    │ 0    │ 0   │
│ Joint-limit saturation  │ 1    │ 0    │ 0    │ 0   │
│ Gripper chatter         │ 1    │ 0    │ 0    │ 0   │
└─────────────────────────────────────────────────────┘

╭─ Funnel ─────────────────────────────────────────────╮
│ What this can't tell you locally:                    │
│   - Whether camera calibration is geometrically       │
│     correct (not just present) -- targetless          │
│     verification is Deepen's sealed, patented audit.  │
│   - A signed acceptance certificate for a data buy.   │
│   - Robot Dataset Quality Index leaderboard entry.    │
│   - Influence-function trainability scoring.          │
│                                                        │
│ Run the full sealed audit: https://deepen-robograde.pages.dev │
╰────────────────────────────────────────────────────────╯

Deepen AI -- https://deepen.ai   |   Full audit: https://deepen-robograde.pages.dev
```

## Install

The base install is deliberately light (numpy, click, rich -- no torch, no
ROS). Add the extra(s) for the formats you actually use:

```bash
pip install deepen-grade                    # base: no format readers yet
pip install "deepen-grade[mcap]"            # .mcap (ROS 2 default)
pip install "deepen-grade[ros]"             # .bag (ROS 1) / .db3 (ROS 2 sqlite)
pip install "deepen-grade[lerobot]"         # LeRobot local path or HF repo-id
pip install "deepen-grade[all]"             # everything
```

For the latest development version: `pip install "deepen-grade[all] @ git+https://github.com/mmusa/deepen-grade"`

`[ros]` uses the pure-Python [`rosbags`](https://pypi.org/project/rosbags/)
library -- no ROS installation required. `[lerobot]` reads the LeRobot
parquet/JSON dataset format directly (`pandas` + `pyarrow` +
`huggingface_hub`) -- it deliberately does **not** depend on the `lerobot`
training package, which pulls in torch and is much heavier than a CLI needs.
Video files are never downloaded or decoded: every check operates on
recorded state/action arrays and timestamps, not pixels.

If you invoke `deepen grade` on a format whose extra isn't installed, you get
a clear one-line fix (`pip install "deepen-grade[...]"`), not a stack trace.

## Usage

```bash
deepen grade my_episode.mcap
deepen grade my_ros1.bag
deepen grade path/to/ros2_bag/            # directory containing metadata.yaml + *.db3
deepen grade path/to/lerobot_dataset/     # local LeRobotDataset v2/v3 directory
deepen grade some-org/some-dataset        # HuggingFace repo-id (downloaded, no video)

deepen grade my_episode.mcap --json                 # machine-readable, for CI
deepen grade my_episode.mcap --json -o report.json
deepen grade my_episode.mcap --min-grade B          # exit 1 if grade < B (CI gate)

deepen grade some-org/big-dataset --sample 200 --seed 0    # grade a random 200-episode sample
deepen grade some-org/big-dataset --max-episodes 200       # grade the first 200 episodes
deepen grade some-org/container-repo --subdataset team_a/session1   # grade one nested dataset only
deepen grade some-org/gated-dataset --hf-token hf_xxx       # or just set HF_TOKEN
deepen grade my_episode.mcap --quiet                        # no HF progress bars, no CLI chatter
```

### Container repos: datasets nested inside a repo

Some Hub repos aren't a single LeRobot dataset at the root -- they nest one or
more complete LeRobot datasets one or two directories down (a
per-contributor directory, an `ImitationLearning/` wrapper, etc). `deepen
grade` detects this automatically (no `meta/info.json` at the root, but one
exists one or two levels down) and grades every nested dataset together,
prefixing each episode ID with its sub-dataset's relative path
(`team_a/session1::episode_000042`). A warning in the report lists every
sub-dataset found. Use `--subdataset <relpath>` to grade just one of them --
that behaves identically to grading it as a standalone dataset.

### Sampling large corpora: `--sample` / `--max-episodes`

Some Hub repos have 100k+ episodes; downloading and reading the whole thing
can OOM a laptop or blow a CI time budget long before grading starts.
`--sample N` grades a uniform random sample of N episodes (seeded via
`--seed`, default `0`, so the same command always picks the same episodes);
`--max-episodes N` grades the first N episodes in natural order instead.
Either way, only the parquet files that actually contain a selected episode
are ever read -- episodes are always materialized one file at a time, never
concatenated into one corpus-sized DataFrame, so peak memory is bounded by
the largest single file, not the corpus. A `"sampling"` block appears in the
JSON report (`{"mode": "sample"|"head", "n", "seed", "episodes_total"}`) and
the terminal report prints a `GRADED ON A SAMPLE` banner; grade letters
themselves are unaffected.

### Partial reports: `--json -o` writes incrementally

With `--json -o report.json`, the report is written to disk incrementally --
at a progress-proportional interval (roughly every 10% of graded episodes,
never more often than every 200), not just at the end -- via a write-to-temp-
file-then-rename so the file on disk is always a complete, parseable JSON
document, never a half-written one. An in-progress write has `"partial":
true`; the final write sets it to `false`. This means a crash or CI timeout
mid-run still leaves a valid, gradeable-so-far report behind instead of
nothing.

### HuggingFace downloads: retries and rate limits

Repo-id downloads retry up to 3 times with exponential backoff on a dropped
connection or other transient Hub error. An HTTP 429 (rate limited) fails
immediately with an actionable message instead of retrying into the same
wall: anonymous downloads have a lower rate limit than authenticated ones, so
pass `--hf-token` (or set `HF_TOKEN`) to raise it.

## What it checks (and what each check cites)

Every check result carries a `claim_type` -- see **Grading** below for what
that means and why only some of these checks can move the letter.

### A. Hygiene -- deterministic, applies to any recording
| Check | What it measures | Claim type | Cited source |
|---|---|---|---|
| Topic frequency & drops | per-topic observed rate vs. that topic's own median interval, gaps > 3x it | DEFECT | ISO/WD 26264-1 companion (arXiv:2606.19769) |
| Cross-modal timestamp skew | ms skew between vision/proprioception/lidar streams | RISK | arXiv:2606.19769 |
| tf-tree integrity | single-parent tree, no static/dynamic conflicts, no cycles | DEFECT | REP 105 |
| Message-schema & state/action-dim consistency | consistent types/dims across a dataset's episodes | DEFECT | arXiv:2606.19769, ASAM OpenLABEL |

Cross-modal skew is RISK, not DEFECT, because its ms budget
(`SKEW_WARN_MS`/`SKEW_FAIL_MS` in `checks/hygiene.py`) is a Deepen convention
parameterized by sensor class, not a sourced standard -- see the module
docstring.

### B. Episode quality -- published metric implementations, all training-value (RISK)
| Check | What it measures | Claim type | Cited source |
|---|---|---|---|
| Idle / stall | near-zero normalized action/state velocity runs | RISK | DQAF (arXiv:2605.26349), SCIZOR (arXiv:2505.22626) |
| Smoothness | log-dimensionless-jerk of the speed profile | RISK | Balasubramanian et al. 2015 (IEEE TBME), DQAF |
| Joint-limit saturation | time spent near the DATASET-WIDE observed value range | RISK | DQAF |
| Gripper chatter | direction-reversal rate of gripper position | RISK | DQAF |
| Action-state consistency | normalized tracking error between commanded action and state | CHARACTERISTIC (dim mismatch) / NOT_ASSESSED (undeclared action space) / RISK (declared absolute) | DQAF, SCIZOR |

Every episode-quality check is a training-value proxy -- see **Grading**
below for why none of them can move the letter, and why that's the fix, not
a regression.

**On thresholds:** the papers above define *what to measure*; the specific
pass/warn/fail cutoffs (e.g. "stall_frac > 30% is a FAIL") are deepen-grade's
own conservative, documented defaults -- see
[`src/deepen_grade/checks/episode_quality.py`](src/deepen_grade/checks/episode_quality.py)
for every constant in one place. No universal numeric cutoff across every
robot embodiment and control rate exists in the literature; these are sane
starting points, not claims about what the cited papers themselves recommend.

### C. Calibration -- its own verdict, never part of the grade
deepen-grade checks that intrinsics/extrinsics are **present** and not
**obviously broken** (missing, NaN, an untouched identity/zero default).
That's it. It does not verify that a calibration is *geometrically correct*
-- doing that from a recording alone (reprojection, epipolar consistency,
drift over a session) is Deepen's patented targetless calibration
verification, which stays sealed and server-side.

Because a well-formed but geometrically *wrong* calibration passes every
local check, calibration is deliberately **excluded from the letter grade**
and reported as its own top-level verdict instead:

- `NOT ASSESSED` -- no calibration metadata found; quality is unknown, not good.
- `PRESENT -- ACCURACY NOT VERIFIED` -- metadata present and structurally sound.
- `STRUCTURALLY BROKEN` -- metadata present but obviously broken (NaN/unset default).

See
[`src/deepen_grade/checks/calibration_sanity.py`](src/deepen_grade/checks/calibration_sanity.py)
for the firewall this module enforces on itself.

Calibration metadata itself is read from whatever the source format actually
carries: for `.mcap`/`.bag`/`.db3`, from a `sensor_msgs/CameraInfo` message on
each camera topic (intrinsics) paired with a matching `/tf_static` transform
by frame_id (extrinsics, left `None` -- never invented -- if no matching
transform was published); for LeRobot, from the `meta/calibration.json`
sidecar convention or (narrowly) a DROID-style `meta/cam2base_extrinsics.json`
/ `meta/info.json["calibration"]` block.

### D. Plausibility -- does this even look like robot data?
A structurally valid LeRobot/mcap/bag file can still contain something that
isn't robot-learning data at all -- a non-robotics dataset reshaped to fit the
schema, for instance. `Robot-data plausibility` is a cheap, dataset-level
heuristic check that flags this: no state/action trajectory anywhere, a state
that never actually changes across any episode, or no usable timestamps.
It's always `INFO` (or `N/A` when nothing looks wrong), never affects the
score, and is deliberately **uncited** -- "does this look like robot data" is
not a metric any of the papers above define. See
[`src/deepen_grade/checks/plausibility.py`](src/deepen_grade/checks/plausibility.py).

## Grading: DEFECT-only letter, training_value profile, grade_schema

`grade_schema` (currently `"1.0.0"`, carried on every report) is a rewrite of
how the letter is computed, not just a version bump:

**The letter is DEFECT-only.** Every check result carries a `claim_type` --
`DEFECT`, `RISK`, `CHARACTERISTIC`, `BY_DESIGN`, or `NOT_ASSESSED` (see
[`checks/base.py`](src/deepen_grade/checks/base.py)'s `ClaimType`). Only
`DEFECT` findings can ever move the letter. `RISK` findings (idle/jerk/
chatter/joint-limit/action-state/cross-modal-skew -- every episode-quality
check, plus cross-modal skew) are proxy training-quality signals whose
thresholds are conventions, not sourced correctness claims; they're reported
in the report's `training_value` section instead (counts per check, zero
letter impact). `CHARACTERISTIC` (e.g. action/state dims differing by
architecture), `BY_DESIGN`, and `NOT_ASSESSED` (a required gate didn't hold --
e.g. an undeclared action-space semantics field) never touch either.

**Aggregation is defect DENSITY, worst-class-bounded -- not a penalty
average.** For every `DEFECT`-class check_id, deepen-grade computes that
check's defect density: `defective findings (WARN or FAIL) / gradable
findings (anything but N/A)`. The dataset's 0-100 integrity score is
`100 * (1 - worst_density)`, where `worst_density` is the MAXIMUM density
across DEFECT check classes -- never their average. This is deliberate: a
single systemic defect (e.g. every episode's tf-tree is broken) must not be
diluted by four other clean classes into a passing grade. The exact
functional form and worked edge cases live in
[`grading.py`](src/deepen_grade/grading.py)'s `integrity_score` docstring, in
the code, not just here. Grade bands are the same `GRADE_BANDS` table as
before (A >= 90, B >= 75, C >= 60, D >= 40, F otherwise), now applied to this
density-derived score instead of a WARN/FAIL penalty sum.

Calibration keeps its own separate top-level verdict, unaffected by any of
this -- see section C above.

**The NOT ASSESSED floor.** An empty gradable set must never auto-pass
(GRADING_TAXONOMY_V1.md section 3). If a dataset is small or sparse enough
that every DEFECT-eligible check individually abstains -- no topics with
enough messages for topic frequency, no tf concept, fewer than 2 episodes to
compare schemas across -- `defect_classes` comes back with no gradable
entries anywhere, and the dataset's letter is the string `"NOT ASSESSED"`
(not a letter band, not a 0-100 score's worth of confidence) instead of the
vacuous 100/A that a bare `max(density, default=0.0)` would otherwise hand
back. The report's `warnings` names every gate that didn't hold. A dataset
where DEFECT checks actually RAN and PASSED -- even trivially, even with
degenerate/garbage content -- is unaffected: real gradable evidence (density
0.0 included) always earns its letter; only genuine silence triggers this.
`--min-grade` treats `NOT ASSESSED` as failing any requested bar, including
`F`, since "couldn't be graded at all" must never read as "cleared the bar."

**Content-plausibility flag.** `plausibility.robot_data` (section D above) is
an index-curation gate, not a letter input -- but a confident-looking grade
must never silently coexist with "this doesn't look like robot-learning
data." When it fires, both `--json` (`content_plausibility: {"flagged",
"detail"}`, top-level, next to `overall`) and the terminal report (a
`CONTENT PLAUSIBILITY FLAG` line directly under `Overall grade`) surface it
prominently rather than leaving it to be found only in the dataset-level
checks table.

**Two checks were fixed as part of this rewrite** (both confirmed on a
50k+-episode BridgeData2 run before the fix):
- `action_state_consistency` used to hard-FAIL any dataset whose action
  space is a delta/velocity/torque/tokenized convention (the majority of
  modern manipulation datasets, e.g. BridgeData2's zero-centered deltas),
  because it compared action to state unconditionally. It now only runs the
  element-wise comparison when a trajectory *declares* `action_space ==
  "absolute"`; every other value (including the honest default of
  undeclared/`None` -- no ingest adapter in this codebase populates it yet)
  returns `NOT_ASSESSED` naming the missing declaration. A dim-count
  mismatch is now `CHARACTERISTIC` ("architectural, not a defect"), never a
  FAIL.
- `joint_limit_saturation` used to use each *episode's own* min/max as the
  limit proxy, so any low-variance dim (a binary 0/1 gripper flag; a short
  episode that only visits a narrow slice of a joint's true range) read as
  permanently "at the limit." It now computes the reference range across
  the WHOLE dataset (aggregated once before per-episode grading starts),
  and excludes binary/discrete dims (fewer than 5 distinct values
  dataset-wide) and near-constant dims outright. It remains RISK-only: there
  is no declared-joint-limits field in this codebase yet, so this is always
  a proxy, never a DEFECT.

**Terminal styling.** Per-check tables carry a `Claim` column (`DEFECT` /
`risk` / `characteristic` / `not assessed` / `by design`), styled distinctly
from the PASS/WARN/FAIL `Result` column, with a footnote reiterating that
only `DEFECT` rows can move the letter above -- a RISK-class FAIL sitting
right under an `A` grade must never read as if it caused that grade.

**`--json` report additions (schema_version 5):** `grade_schema`,
`defect_classes` (the density behind `overall`, per DEFECT check_id),
`training_value` (RISK-class severity counts, per check_id), and
`content_plausibility` (`{"flagged", "detail"}`). Nothing existing was
removed or renamed. `overall.grade` can now also be the string
`"NOT ASSESSED"` (see the floor, above).

**A note on comparability:** grades produced under `grade_schema` earlier
than `1.0.0` are not comparable to grades under `1.0.0` -- the method changed,
not the data. See [CHANGELOG.md](CHANGELOG.md).

## What this can't tell you locally

Every report ends with a funnel section, because it's true, not because it's
a marketing footer:

- Whether calibration is geometrically **correct**, not just present.
- A signed, reproducible acceptance certificate for a data purchase.
- Entry in the public Robot Dataset Quality Index leaderboard.
- Influence-function trainability scoring (which episodes help your policy).

Full sealed audit: **https://deepen-robograde.pages.dev**

## A self-assessment, not a certification

deepen-grade runs on your machine, on your data, with nothing checked by
Deepen -- so its report is a **self-assessment**, and it says so on its face
(`report_type: "self-assessment"` in `--json`). There is deliberately no
claimable "verified" badge here: a trust mark Deepen never saw would be an
honor-system claim, which is exactly what this tool exists to replace.
Signed, third-party attestation -- a report a counterparty can rely on --
is the sealed deep audit at [deepen-robograde.pages.dev](https://deepen-robograde.pages.dev).

## CI: fail the data PR on a bad episode

```bash
deepen grade path/to/dataset --json --min-grade B
```
exits non-zero if the overall grade is worse than `B`. A ready-to-use
composite GitHub Action is in [`action/`](action/action.yml); see
[`examples/ci-gate.yml`](examples/ci-gate.yml) (copy it into `.github/workflows/` in your repo)
for a full workflow that runs it on every PR touching a dataset directory.

```yaml
- uses: mmusa/deepen-grade@v0
  with:
    path: path/to/dataset
    min-grade: B
    extras: lerobot   # mcap | ros | lerobot | all
```

## Privacy

Nothing uploads by default -- full stop. `--share-report` is an explicit
opt-in flag for a future sealed/signed report from deepen-robograde.pages.dev; **it
is not implemented in this release** (see
[`src/deepen_grade/report/share.py`](src/deepen_grade/report/share.py)) and
performs no network call. Use `--share-report --share-report-dry-run
payload.json` to inspect exactly what the eventual payload would contain.

## Development

```bash
git clone https://github.com/mmusa/deepen-grade
cd deepen-grade
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0 -- see [LICENSE](LICENSE).

---

Built by [Deepen AI](https://deepen.ai), the data infrastructure layer for
physical AI. `deepen-grade` is the free, always-local doorway; the full
sealed audit (targetless calibration verification, acceptance certification)
lives at [deepen-robograde.pages.dev](https://deepen-robograde.pages.dev).
