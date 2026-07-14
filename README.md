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
Overall grade: B (82/100)   self-assessment -- hygiene & episode quality only
Calibration:   NOT ASSESSED
               no calibration metadata found anywhere in this dataset -- calibration
               quality is unknown, not good; verification requires the deep audit

         Per-episode grades
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Episode     ┃ Grade ┃ Score ┃ Task ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━┩
│ my_episode  │ B     │ 82    │ -    │
└─────────────┴───────┴───────┴──────┘

           Checks -- my_episode
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳...
│ Topic frequency & drops │ PASS │ 4 topics, no drops detected
│ Cross-modal sync skew   │ WARN │ max skew 28.4ms exceeds 15ms budget
│ tf-tree integrity       │ PASS │ 6 frames, single-parent tree
│ Idle / stall detection  │ PASS │ 3.1% of episode idle
│ LDJ smoothness          │ PASS │ LDJ = -14.2
│ Joint-limit saturation  │ PASS │ worst dim 'joint_3' saturated 4.2%
│ Gripper chatter         │ PASS │ 2 direction reversals (0.31 Hz)
│ Action-state consistency│ PASS │ consistency score 0.94
│ Calibration sanity      │ INFO │ NOT ASSESSED -- no camera calibration metadata found
└──────────────────────────────────────────────────────┘

╭─ Funnel ─────────────────────────────────────────────╮
│ What this can't tell you locally:                    │
│   - Whether camera calibration is geometrically       │
│     correct (not just present) -- targetless          │
│     verification is Deepen's sealed, patented audit.  │
│   - A signed acceptance certificate for a data buy.   │
│   - Robot Dataset Quality Index leaderboard entry.    │
│   - Influence-function trainability scoring.          │
│                                                        │
│ Run the full sealed audit: https://evaluate.deepen.ai │
╰────────────────────────────────────────────────────────╯

Deepen AI -- https://deepen.ai   |   Full audit: https://evaluate.deepen.ai
```

## Install

The base install is deliberately light (numpy, click, rich -- no torch, no
ROS). Add the extra(s) for the formats you actually use:

```bash
pip install deepen-grade                    # base: no format readers yet
pip install "deepen-grade[mcap]"             # .mcap (ROS 2 default)
pip install "deepen-grade[ros]"              # .bag (ROS 1) / .db3 (ROS 2 sqlite)
pip install "deepen-grade[lerobot]"          # LeRobot local path or HF repo-id
pip install "deepen-grade[all]"              # everything
```

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
```

## What it checks (and what each check cites)

### A. Hygiene -- deterministic, applies to any recording
| Check | What it measures | Cited source |
|---|---|---|
| Topic frequency & drops | per-topic observed rate, gaps > 3x median interval | ISO/WD 26264-1 companion (arXiv:2606.19769) |
| Cross-modal timestamp skew | ms skew between vision/proprioception/lidar streams | arXiv:2606.19769 |
| tf-tree integrity | single-parent tree, no static/dynamic conflicts, no cycles | REP 105 |
| Message-schema & state/action-dim consistency | consistent types/dims across a dataset's episodes | arXiv:2606.19769, ASAM OpenLABEL |

### B. Episode quality -- published metric implementations
| Check | What it measures | Cited source |
|---|---|---|
| Idle / stall | near-zero normalized action/state velocity runs | DQAF (arXiv:2605.26349), SCIZOR (arXiv:2505.22626) |
| Smoothness | log-dimensionless-jerk of the speed profile | Balasubramanian et al. 2015 (IEEE TBME), DQAF |
| Joint-limit saturation | time spent near the episode's own observed value range | DQAF |
| Gripper chatter | direction-reversal rate of gripper position | DQAF |
| Action-state consistency | normalized tracking error between commanded action and state | DQAF, SCIZOR |

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

## What this can't tell you locally

Every report ends with a funnel section, because it's true, not because it's
a marketing footer:

- Whether calibration is geometrically **correct**, not just present.
- A signed, reproducible acceptance certificate for a data purchase.
- Entry in the public Robot Dataset Quality Index leaderboard.
- Influence-function trainability scoring (which episodes help your policy).

Full sealed audit: **https://evaluate.deepen.ai**

## A self-assessment, not a certification

deepen-grade runs on your machine, on your data, with nothing checked by
Deepen -- so its report is a **self-assessment**, and it says so on its face
(`report_type: "self-assessment"` in `--json`). There is deliberately no
claimable "verified" badge here: a trust mark Deepen never saw would be an
honor-system claim, which is exactly what this tool exists to replace.
Signed, third-party attestation -- a report a counterparty can rely on --
is the sealed deep audit at [evaluate.deepen.ai](https://evaluate.deepen.ai).

## CI: fail the data PR on a bad episode

```bash
deepen grade path/to/dataset --json --min-grade B
```
exits non-zero if the overall grade is worse than `B`. A ready-to-use
composite GitHub Action is in [`action/`](action/action.yml); see
[`examples/ci-gate.yml`](examples/ci-gate.yml) (copy it into `.github/workflows/` in your repo)
for a full workflow that runs it on every PR touching a dataset directory.

```yaml
- uses: deepen-ai/deepen-grade@v0
  with:
    path: path/to/dataset
    min-grade: B
    extras: lerobot   # mcap | ros | lerobot | all
```

## Privacy

Nothing uploads by default -- full stop. `--share-report` is an explicit
opt-in flag for a future sealed/signed report from evaluate.deepen.ai; **it
is not implemented in this release** (see
[`src/deepen_grade/report/share.py`](src/deepen_grade/report/share.py)) and
performs no network call. Use `--share-report --share-report-dry-run
payload.json` to inspect exactly what the eventual payload would contain.

## Development

```bash
git clone https://github.com/deepen-ai/deepen-grade
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
lives at [evaluate.deepen.ai](https://evaluate.deepen.ai).
