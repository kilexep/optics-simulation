# Development Reference Sources

This file summarizes external source references Claude should consult before implementing major modules.

## Core development stack

Primary implementation stack:
- `trimesh`: STL loading, mesh QA, bounds, centroid, watertight checks, proximity/thickness utilities
- `Open3D`: future ray-mesh intersection engine via `RaycastingScene`
- `numpy`: vector math, ray arrays, detector maps
- `pandas` / Parquet: future ray history logging
- `SciPy`: sampling, fitting, optimization helpers, thermal ODEs
- `Optuna`: parameter optimization
- `SALib`: sensitivity analysis
- `MLflow`: experiment/run tracking

Validation / high-fidelity references:
- `Raysect`: scientific spectral ray tracing and detector-style validation
- `Mitsuba 3`: Fresnel, dielectric, BSDF/BTDF validation
- `pbrt-v4`: physically based rendering reference, not primary engine

Pattern and manufacturing references:
- `SciPy QMC`: Poisson-disk, Sobol, Halton sampling
- `SciPy RBFInterpolator` / B-spline tools: pattern formula fitting
- `geomdl`: NURBS/B-spline representation
- `CadQuery`: STEP export
- `Blender` / `PyMeshLab`: optional preprocessing, not core runtime dependency

## Architecture guidance

Use this pipeline direction:

STL load
→ mesh QA
→ ray intersection
→ custom Snell/Fresnel
→ detector accumulation
→ ray history logging
→ hotspot backtracking
→ contribution map
→ adaptive pattern generation
→ RBF/B-spline formula fitting
→ optimization
→ simplified thermal validation
→ formula/STL/STEP export

## Implementation priorities

1. Keep `trimesh` for geometry ingestion and QA.
2. Add `Open3D RaycastingScene` later for ray intersection, not during STL ingestion foundation.
3. Keep Snell/Fresnel logic custom and unit-tested.
4. Log ray history in a structured table format before building contribution maps.
5. Treat Raysect/Mitsuba as validation references, not first implementation targets.
6. Keep PyMeshLab/Blender optional because of heavier tooling and licensing implications.
7. Store generated artifacts under `runs/` or `results/`, not in Git.

## Full research report

The full collected reference report should be stored as:

`docs/references/deep-research-report.md`

Claude should read that file before implementing:
- ray tracing
- Snell/Fresnel
- ray history logging
- hotspot backtracking
- pattern fitting
- optimization
- thermal surrogate
- CAD/STEP export

## Quick lookup for deep-research-report.md

Map from CLAUDE.md pipeline step → relevant section(s) in
`docs/references/deep-research-report.md`. Use section headings (H2/H3),
not line numbers, since the report may be edited.

Pipeline steps follow CLAUDE.md "Intended pipeline".

| Step | Topic | Sections in deep-research-report.md |
|---|---|---|
| 1 | Config loader and validation | (no direct section — use CLAUDE.md + existing config loader tests) |
| 2 | STL ingestion and mesh QA | `## 코어 스택 권고` → `### 광선 추적과 교차 연산` (trimesh paragraph), `### STL 전처리와 메쉬 보정` (PyMeshLab/Blender as optional only) |
| 3 | Baseline 3D ray tracing scaffold | `### 광선 추적과 교차 연산` (Open3D `RaycastingScene` paragraph), `## 최소 코드 패턴` → `### ray–mesh intersection` |
| 4 | Angle scan and risk-angle detection | Same as step 3 + `### ray history logging` (parquet schema) |
| 5 | Hotspot ray backtracking and contribution map | `### ray history logging`, `### hotspot backtracking` (Open3D `primitive_ids` → `source_face_id`) |
| 6 | Pattern generation from Rrisk | `## 패턴 생성과 수식·CAD 변환` (Poisson-disk / Sobol / Halton, RBF, B-spline, NURBS) |
| 7 | Optimization loop | `## 최적화·열모델·검증` (Optuna, Nevergrad, MLflow) |
| 8 | Residual hotspot refinement | Steps 5 + 6 (re-run contribution map after pattern application) |
| 9 | High-resolution validation | `### 광선 추적과 교차 연산` (Raysect, Mitsuba 3, pbrt-v4 paragraphs), `### Snell·Fresnel·BSDF·BTDF·Detector 참조` |
| 10 | Simplified thermal evaluation | `## 최적화·열모델·검증` (lumped-capacitance surrogate paragraph + code block). **Note**: ignition modeling/prediction is excluded per CLAUDE.md safety boundary; surrogate produces relative thermal-risk metrics (Tmax, dT/dt, Ahot,T) only |
| 11 | Formula export and patterned STL export | `## 패턴 생성과 수식·CAD 변환` (RBF formula spec + B-spline/NURBS + CadQuery STEP export) |
| 12 | Hold-out validation and ablation tests | `## 최적화·열모델·검증` (pytest + NumPy testing + Hypothesis subsection, the five recommended tests) |
| 13 | Report generation | `## 저장소 구조와 권장 산출물` (mermaid pipeline diagram, recommended artifact table) |

### Topic → section quick-jump

For ad-hoc lookups (not pipeline-aligned):

- **Snell + Fresnel + TIR code skeleton**: `## 최소 코드 패턴` → `### Snell refraction + Fresnel`
- **PET / water / air IOR baseline values**: `### Snell·Fresnel·BSDF·BTDF·Detector 참조` (Mitsuba dielectric preset: PET=1.575, water=1.3330, air=1.00028)
- **Ray history parquet schema**: `### ray history logging`
- **Pattern formula YAML example**: `## 패턴 생성과 수식·CAD 변환` (the embedded YAML block)
- **License classification table**: `## 집행 요약` (last paragraph of executive summary)
- **Recommended artifact list (project files to create)**: `## 저장소 구조와 권장 산출물` → "프로젝트에 추가할 권장 아티팩트" table
- **Per-source reference paragraphs (one-liners with license + URL)**: `## 붙여넣기용 MD 참조 문구`

### Safety reminder

The thermal surrogate code in the report has been edited to exclude
`ignition_event` and ignition threshold logic. Do not reintroduce them.
The thermal model in this project is restricted to relative thermal-risk
metric comparison (Tmax, dT/dt, Ahot,T) per CLAUDE.md safety boundary.