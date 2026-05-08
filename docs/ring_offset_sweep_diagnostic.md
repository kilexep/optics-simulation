# Ring-offset risk-guided pattern sweep diagnostic

> Surrogate ray-backtracking diagnostic; **not a fire-prevention proof**,
> **not a manufacturing-ready validation**, and **not a PET-bottle safety
> claim**. The phrases "best", "improving", and "safe" do not appear in
> the runtime output unless the holdout metrics actually support them.

## 1. Research question

Earlier risk-guided pattern sweeps showed that placing Gaussian dimples
directly on top of detector-hotspot contribution bins tends to
*strengthen* the same caustic instead of scattering it (the patterned
hotspot rays backtrack to the **same** shell-surface bins as the
baseline; `candidate_only_bins == 0`).

The ring-offset diagnostic asks a narrower question:

> **Q.** If we transform the aggregate hotspot risk map so that
> probability mass is moved into an annular ring around each active
> bin, and then sample dimples from the *transformed* map, does the
> patterned shell reduce the worst-case max-temperature surrogate
> across the evaluation grid — or does it merely redistribute the
> caustic and create new ones elsewhere?

This is a **diagnostic** question. The answer is not "yes" or "no"; the
answer is a vector of axis-specific outcomes (max-T delta,
threshold-count delta, guardrail pass ratio, hotspot overlap fraction)
on a holdout subset, with explicit flags that catch over-fitting to the
selection conditions.

## 2. Scope (what this diagnostic does and does not do)

| Does | Does not |
|------|----------|
| Apply a discrete ring transform to a caller-supplied risk map | Optimize pattern parameters |
| Build a risk-guided patterned PET-water setup per candidate | Repair, smooth, or remesh the input STL |
| Run the legacy four-step PET-water angle/distance scan | Model ignition, pyrolysis, or CFD |
| Compute per-pixel thermal-risk metrics and guardrail comparisons | Calibrate W/m² to physical absolute irradiance |
| Split the comparison into in-sample (selection-condition) and holdout subsets | Treat in-sample improvement as generalized improvement |
| Report contribution-map overlap (shared / candidate-only / jaccard) | Use `new_bin_fraction` alone as a quality axis |
| Pareto-filter candidates on three discriminative axes on the holdout subset | Pretend a single composite score is a primary verdict |
| Flag `no_improving_candidate_in_holdout` when every candidate worsens worst-case max-T | Claim "best overall" or "manufacturing-ready" |

## 3. Why selection / holdout split matters

Stage 1 of the pipeline picks a small set of "worst" baseline
conditions and uses them to build the multi-condition aggregate
hotspot risk map. Re-evaluating each candidate **only** on the same
conditions over-credits patterns that happen to move that one specific
caustic but break behavior elsewhere on the evaluation grid.

The diagnostic therefore splits each candidate's comparison entries:

- **in_sample_diagnostics** — entries whose `(angle_degrees,
  detector_distance)` matches a selected condition.
- **holdout_diagnostics** — every other condition on the schedule.
- **full_diagnostics** — all entries on the schedule (selection ∪
  holdout).

Ranking defaults to the **holdout** basis. In-sample diagnostics are
displayed only as a **leakage-warning channel**: if the in-sample top
candidate diverges from the holdout top, the demo prints an explicit
leakage-warning line. In-sample negative deltas paired with positive
holdout deltas trigger a second leakage warning — "in-sample
improvement does NOT imply holdout improvement".

## 4. Why we do not generalize from in-sample improvement

In a known empirical run (this surrogate, the local PET STL
`data/raw/stl/pet_bottle.stl`, 2 selected conditions on an 18×8
schedule), several ring candidates show in-sample
`worst_delta_max_temperature_k` in the −20 K range while their holdout
worst delta is +25 K to +48 K. The selection conditions are by
definition the angles/distances where the baseline already had its
worst caustic; redistributing energy at those particular conditions is
the easy case. The holdout is the test that actually matters for any
"improvement" claim, and on the holdout no candidate achieves a
strictly negative worst delta.

## 5. Overlap and coverage metrics: how to read them

Each candidate reports six overlap-related numbers, computed against
the baseline contribution map at the candidate's worst-worsened
holdout entry:

| Metric | Meaning | Red flag |
|--------|---------|----------|
| `baseline_nonzero_bins` | Hotspot footprint of the baseline | — |
| `candidate_nonzero_bins` | Hotspot footprint of the candidate | — |
| `source_overlap_bin_count` | Bins active in **both** | Very high together with high `retained_source_bin_fraction` and zero `new_bin_fraction` ⇒ pattern did not move the hotspot |
| `source_overlap_fraction`, `retained_source_bin_fraction` | Same value, two names for the fraction of baseline bins retained | High value paired with positive Δmax T means the pattern strengthened the existing hotspot |
| `candidate_only_bin_count`, `new_bin_fraction` | New hotspot bins introduced by the candidate | High value with positive Δmax T ⇒ pattern created new caustics |
| `jaccard_overlap` | Symmetric overlap = shared / union | 1.0 ⇒ no movement; near 0 ⇒ very different distribution |

### `new_bin_fraction` is *not* a single-axis quality measure

A candidate with `new_bin_fraction == 0` may have *failed to move* the
hotspot rather than *successfully containing* it. A candidate with
`new_bin_fraction > 0` may be redistributing energy usefully or may be
creating fresh caustics. The diagnostic therefore reports
`source_overlap_fraction`, `retained_source_bin_fraction`,
`candidate_only_bin_count`, `jaccard_overlap`, and `new_bin_fraction`
**together**; none of them is part of the Pareto axis tuple.

Coverage / complexity metrics:

- `risk_support_expansion_ratio = transformed_risk_active_count /
  source_risk_active_count` — how aggressively the ring transform
  spread the source risk. A ratio of 6–8 is typical for
  `inner_radius_px = 1`, `outer_radius_px = 3..7`.
- `moved_vertex_fraction` — fraction of body-mask vertices the
  patterned mesh actually displaced. Useful to detect "the pattern
  barely touched the surface" vs "the pattern deformed half the body
  region".

## 6. `no_improving_candidate_*` flags

Two flags are set automatically:

- `no_improving_candidate_under_current_sweep` — `True` iff every
  candidate's **full-grid** `worst_delta_max_temperature_k` is strictly
  positive.
- `no_improving_candidate_in_holdout` — `True` iff every candidate's
  **holdout** `worst_delta_max_temperature_k` is strictly positive.

When `no_improving_candidate_in_holdout` is `True` the demo switches
its phrasing from "Best by ..." to **"Least-bad by ..."**, prints a
banner stating no candidate is described as Best overall or Improving,
and only emits "Pareto non-dominated diagnostic set" / "Least-bad by
holdout worst Δmax temperature" / "Least-bad by holdout pass ratio".

## 7. Pareto vs composite score

- **Pareto non-dominated set (primary)**. Three lower-is-better axes
  on the holdout subset:
  `worst_delta_max_temperature_k`, `best_delta_threshold_count`,
  `-pass_ratio`. Non-discriminative axes (range zero across
  candidates) are auto-dropped and reported in
  `non_discriminative_metrics`. The Pareto filter is the diagnostic
  ordering — read this first.
- **Composite score (tie-break hint, NOT primary)**. Min-max-normalized
  weighted sum on the surviving Pareto axes. Use it only as a
  convenience score to break ties **within** the Pareto set. The
  composite cannot override the `no_improving_candidate_*` flags or
  promote a non-Pareto candidate.

If the threshold-count axis is non-discriminative (every candidate has
the same value on holdout), the demo does not print a "least-bad by
holdout threshold-count delta" winner — it prints
"non-discriminative". Likewise for the other two axes.

## 8. Ring radius units

`inner_radius_px` and `outer_radius_px` are integer **risk-map bin**
units. The aggregate risk map has shape
`(nv, nu) = contribution_resolution`, where:

- `nv` indexes height bins (axis 0); `v ∈ [0, 1]` is non-wrapping.
- `nu` indexes circumferential bins (axis 1); `u ∈ [0, 1)` wraps
  around.

These are **not**:

- mesh-space vertex hops,
- detector-grid pixels,
- world-space millimeters.

The default `contribution_resolution = (32, 64)` means
`outer_radius_px = 7` covers ~22% of the height range and ~11% of the
circumference. If you change `contribution_resolution`, the meaning of
the same `inner_radius_px` / `outer_radius_px` changes proportionally.

## 9. Local STL asset handling

The actual demo defaults to `data/raw/stl/pet_bottle.stl`, which is an
**untracked, local-only input**. The committed test suite does **not**
depend on it — every subprocess test exports a synthetic STL into
`tmp_path` and passes that explicit path to the demo via `--mesh`.

If the demo cannot find the requested STL it prints a clear message
and exits with a non-zero status. The demo header also discloses, when
the default path is used, that `data/raw/stl/pet_bottle.stl` is a
local-only input.

To run the diagnostic without the local STL:

1. Provide your own `--mesh path/to/your.stl`, or
2. Use the synthetic-fixture invocation pattern in
   [`tests/test_examples_actual_stl_ring_offset_sweep_demo.py`](
   ../tests/test_examples_actual_stl_ring_offset_sweep_demo.py)
   as a reference (export
   `create_subdivided_synthetic_bottle_body(...)` to a tmp file and
   pass that path).

## 10. Known negative result on the local STL

With:

- 8 ring candidates on `inner ∈ {1, 2, 3} × outer ∈ {3, 5, 7}`,
- the fixed pattern shape
  `pattern_count=20, sigma=0.03, max_depth=0.05, seed=42`,
- 3 selected worst conditions at the default schedule
  (`angle_count=18, angle_step=20, detector_count=8,
  detector_start=100, detector_spacing=40`),
- subdivision_iterations = 1,

the diagnostic reports
`no_improving_candidate_in_holdout == True`. Every candidate's holdout
`worst_delta_max_temperature_k` is strictly positive (range ≈ +24 to
+48 K). Several candidates show **in-sample** negative deltas in the
−20 K range; the demo flags this as a leakage divergence and prints an
explicit warning that in-sample improvement does NOT imply holdout
improvement.

Pareto non-dominated set on holdout: `ring_in1_out3` and `ring_in1_out5`.
None of them is described as Best overall or Improving.

This is a negative result for **this specific configuration**; it is
not a blanket judgment on ring-offset patterns in general. Other
pattern shapes, other selection rules, other ring grids, and other
mesh resolutions are not covered.

## 11. How to run

Local actual STL (requires `data/raw/stl/pet_bottle.stl`):

```bash
python examples/run_actual_stl_ring_offset_sweep_demo.py \
    --mesh data/raw/stl/pet_bottle.stl \
    --target-height 225.6 --target-diameter 72.1 \
    --wall-thickness 0.3 --inner-offset-mode auto \
    --subdivision-iterations 1 \
    --max-conditions 3 --hotspot-top-percent 10 \
    --angle-count 18 --angle-step 20 \
    --detector-count 8 --detector-start 100 \
    --detector-spacing 40 \
    --sample-count-y 11 --sample-count-z 21 \
    --detector-resolution 40 \
    --pattern-count 20 --pattern-sigma 0.03 \
    --pattern-max-depth 0.05 --pattern-seed 42 \
    --duration 60 --dt 0.5
```

Tests (do not require any local STL):

```bash
python -m pytest
```

To verify clean-checkout reproducibility, temporarily move the local
STL out of the working tree and re-run `python -m pytest` — the suite
must still pass.

## 12. Public API

All public symbols are exported through
`optics_simulation.optics`:

| Symbol | Stage |
|--------|-------|
| `RING_RADIUS_UNITS_DESCRIPTION` | units note |
| `RingOffsetSweepCandidateSpec` | stage 2 input |
| `ConditionSubsetDiagnostics` | stage 5 |
| `RingOffsetSweepEntry`, `RingOffsetSweepResult` | stage 5 output |
| `summarize_thermal_risk_comparison_subset` | stage 5 helper |
| `split_selected_vs_holdout_conditions` | stage 5 helper |
| `compute_holdout_diagnostics` | stage 5 convenience |
| `rank_ring_offset_sweep_pareto_indices` | stage 5 ranking |
| `rank_ring_offset_sweep_by_composite_score` | stage 5 tie-break |
| `run_actual_stl_ring_offset_sweep` | stage 4+5 orchestration |
| `validate_ring_offset_sweep_result` | invariant gate |

The canonical ring transform itself lives in
[`optics_simulation.contribution.risk_transform`](
../src/optics_simulation/contribution/risk_transform.py); the sweep
module is an orchestration layer and does not duplicate the transform
mathematics.
