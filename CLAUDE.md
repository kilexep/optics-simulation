# CLAUDE.md — optics-simulation

## Project summary
This is a research-grade Python project for STL-based PET bottle caustic simulation and pattern optimization.

The project studies how a transparent PET bottle can concentrate transmitted light into caustic hotspots, then designs surface indentation patterns that reduce local optical and simplified thermal risk metrics.

Do not frame the project as complete wildfire prevention. Frame results as reduction of PET-bottle caustic and local thermal-risk metrics.

## Current repository state
The repository already has:
- Python package scaffold under `src/optics_simulation/`
- tests under `tests/`
- research config files under `config/`
- config explanation under `docs/config_overview.md`
- ignored output folders: `runs/`, `results/`

Before implementing, inspect:
- `config/*.yaml`
- `docs/config_overview.md`
- `pyproject.toml`
- existing tests

## Most important rule
Do not hardcode research constants in Python modules.

Use config files for:
- STL path
- units
- refractive indices
- ray counts
- detector geometry
- angle scan ranges
- pattern parameters
- optimization weights
- validation conditions
- output paths

If a value belongs to an experiment condition, it belongs in `config/*.yaml`.

## Research mechanism
Use this physical chain when explaining or implementing:

surface depth pattern δ(u,v)
→ local surface normal dispersion
→ transmitted angular redistribution / BTDF entropy
→ reduced caustic peak C99 and excess energy Eexceed
→ reduced Tmax and dT/dt in a simplified absorber model

Use "BTDF-based transmitted angular redistribution" instead of vague "diffuse reflection".

## Primary pattern model
Final pattern family:

δ(u,v) = dmax * clip( Σ Ai * exp(-||r-ri||² / (2σi²)), 0, 1 )

where:
- r = (u,v)
- u ∈ [0,1): circumferential coordinate
- v ∈ [0,1]: height coordinate
- ri = dimple centers
- Ai = dimple depth coefficient
- σi = dimple radius
- dmax = maximum indentation depth

Dimple centers are sampled from the caustic contribution map:

P(ri at u,v) ∝ Rrisk_norm(u,v) + ε
||ri - rj|| ≥ rmin

Residual update:

Rfinal(u,v) = α * R0(u,v) + (1 - α) * Rresidual(u,v)

Default α comes from config unless changed.

## Intended pipeline
Implement incrementally in this order:

1. config loader and validation
2. STL ingestion and mesh quality report
3. baseline no-pattern 3D ray tracing scaffold
4. angle scan and risk-angle detection
5. hotspot ray backtracking and contribution map
6. pattern generation from Rrisk
7. optimization loop
8. residual hotspot refinement
9. high-resolution validation
10. simplified thermal evaluation
11. formula export and patterned STL export
12. hold-out validation and ablation tests
13. report generation

Do not skip directly to optimization before STL ingestion and baseline validation exist.

## Comparison groups
Use these labels consistently:
- G0: no-pattern STL
- G1: uniform dimple
- G2: regular grid/groove
- G3: fully random dimple
- G4: uniform quasi-random dimple
- G5: caustic-adaptive quasi-random dimple
- G6: residual-updated caustic-adaptive pattern

## Primary metrics
Optical:
- C99 = mean(top 1% of irradiance pixels) / Iincident
- Cmax = Imax / Iincident, secondary only
- Eexceed = integral of max(I(x,y)-Ith, 0)
- Ahot = area where I(x,y) > Ith
- BTDF entropy = -sum(pi log pi)
- NDI = std(local normal deviation angle)

Thermal:
- Tmax
- dT/dt
- Ahot,T

Use objective weights from `config/optimization.yaml`.

## Safety and claim boundaries
- Do not model ignition.
- Do not optimize for ignition.
- Do not provide instructions for starting fires.
- Use simplified absorber-plate thermal metrics only.
- Keep claims limited to caustic and local thermal-risk metric reduction.
- Always mention when a model is simplified or approximate.

## Code style
- Keep modules small and testable.
- Use typed functions where practical.
- Prefer dataclasses or clear dictionaries for structured objects.
- Add or update tests for each new module.
- Do not generate large result files inside Git-tracked paths.
- Store generated artifacts under `runs/` or `results/`.

## Git and workflow rules
Before coding:
1. Inspect relevant files.
2. Explain the plan.
3. Wait for approval if the change touches more than 3 files or changes architecture.

After coding:
1. Run `python -m pytest`.
2. Run `git status --short`.
3. Summarize changed files.
4. Do not commit automatically unless explicitly asked.

Commit style:
- `docs: ...`
- `config: ...`
- `geom: ...`
- `optics: ...`
- `scan: ...`
- `pattern: ...`
- `opt: ...`
- `val: ...`
- `report: ...`

## Next implementation task
The next task after creating this file is:

Build a config loader module that reads `config/*.yaml`, validates required files exist, and exposes a typed or structured config object.

Do not start ray tracing until config loading and tests are complete.