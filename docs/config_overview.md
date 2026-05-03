# Config Overview

This folder contains the research configuration foundation for the optics-simulation project.

## Purpose

The config files define the assumptions and boundaries of the simulation before implementation begins. The goal is to keep geometry, materials, optics, pattern generation, optimization, validation, and output tracking reproducible.

## Files

- `project.yaml`: project identity, claim scope, and safety boundary
- `geometry.yaml`: STL loading, axis alignment, wall model, detector geometry
- `materials.yaml`: refractive indices and simplified absorber material values
- `optics.yaml`: ray-tracing model, ray counts, detector metrics
- `angle_scan.yaml`: full incident-angle scan and risk-angle detection
- `pattern.yaml`: Gaussian dimple field, contribution-map weighting, residual update
- `optimization.yaml`: Optuna settings and robust objective weights
- `thermal.yaml`: simplified absorber-plate thermal model
- `validation.yaml`: comparison groups, hold-out tests, ablation tests
- `tracking.yaml`: run directories and artifact logging policy

## Research logic

The main physical chain is:

surface depth pattern δ(u,v)
→ local surface normal dispersion
→ transmitted angular redistribution / BTDF entropy
→ reduced caustic peak C99 and Eexceed
→ reduced simplified thermal metrics Tmax and dT/dt

## Safety boundary

This project does not model or optimize ignition. The results should be framed as local optical and thermal-risk metric reduction, not direct wildfire prevention.
